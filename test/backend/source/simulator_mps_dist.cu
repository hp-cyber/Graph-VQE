// NVQIR wrappers around the tested custom MPS engine (sharded_mps.cuh).
//
// This source intentionally builds several plugins from one implementation:
//   tensornet-mps-mgpu: the original MPS-chain sharding backend.
//   tensornet-mps-mqpu-shots: block-first logical QPUs; finite-shot diagonal
//                       observations reuse one distribution and shard the
//                       fixed total shot budget over each physical-GPU group.
//   tensornet-mps-mqpu-hamiltonian: block-first logical QPUs; finite-shot
//                       observations shard Hamiltonian terms over each
//                       physical-GPU group.
//   tensornet-mps-mqpu: compatibility alias for the shots implementation.
//   tensornet-mps-mqpu-replay: persistent GPU workers independently replay the
//                       ansatz, then evaluate their Hamiltonian term shards.
//
// IMPORTANT: this file compiles ONLY inside an in-tree CUDA-Q build (it needs
// nvqir/CircuitSimulator.h and cudaq/spin_op.h). It is NOT built by the local
// build.sh. Drop it into runtime/nvqir/cutensornet/ and add it to that dir's
// CMakeLists (see distributed_mps/cpp_backend/CMake_snippet.txt). Spots marked
// [VERIFY] use CUDA-Q APIs whose exact form may need a small fixup for your
// release; the underlying engine is already validated.
//
// Design: replaces cuTensorNet's lazy state with our eager explicit MPS chain.
//   applyGate  -> ShardedMPS::apply1q / applyTwoQubit (SWAP-routed)
//   observe    -> sum_k coeff_k * <P_k>  via ShardedMPS::expectPauli
//   bond dim   <- CUDAQ_MPS_MAX_BOND (same env var as stock tensornet-mps)
//
// CUDAQ_MPS_MQPU_BACKEND is defined only for the tensornet-mps-mqpu plugin.
// The MQPU platform exposes block primaries. Each simulator instance owns one
// disjoint CUDAQ_MPS_HBATCH_GROUPS entry and performs real Hamiltonian batching
// without replaying the ansatz on each helper GPU.
//
// Noise support (quantum trajectory / Monte Carlo wavefunction method):
//   - applyNoiseChannel() stochastically selects one Kraus operator per gate.
//   - Unitary-mixture channels (depolarizing, bit-flip, phase-flip, Pauli):
//     selected with given probabilities; no renormalization needed.
//   - General 1-qubit Kraus (amplitude damping): probability of each operator
//     computed from the local reduced density matrix via <X>,<Y>,<Z> on the
//     target qubit; state renormalized after selection.
//   - observe()/sample() with noise run a trajectory loop: gate history is
//     replayed from |0>^n for each trajectory, each with a fresh Kraus
//     realization. Number of trajectories = numberTrajectories if set, else
//     shots, else DEFAULT_OBSERVE_TRAJECTORIES.

#if defined(CUDAQ_MPS_MQPU_SHOTS_BACKEND)
#define ShardedMPS ShardedMPSMQPUShots
#define SimulatorMPSDist SimulatorMPSMQPUShots
#elif defined(CUDAQ_MPS_MQPU_HAMILTONIAN_BACKEND)
#define ShardedMPS ShardedMPSMQPUHamiltonian
#define SimulatorMPSDist SimulatorMPSMQPUHamiltonian
#elif defined(CUDAQ_MPS_MQPU_STRONG_BACKEND)
#define ShardedMPS ShardedMPSMQPUStrong
#define SimulatorMPSDist SimulatorMPSMQPUStrong
#elif defined(CUDAQ_MPS_MQPU_REPLAY_BACKEND)
#define ShardedMPS ShardedMPSMQPUReplay
#define SimulatorMPSDist SimulatorMPSMQPUReplay
#elif defined(CUDAQ_MPS_MQPU_BACKEND)
// Both custom plugins can be loaded into the same process. Give the MQPU build
// distinct C++ symbols so ELF interposition cannot bind its factory/vtable or
// MPS methods to the already-loaded tensornet-mps-mgpu implementation.
#define ShardedMPS ShardedMPSMQPU
#define SimulatorMPSDist SimulatorMPSMQPU
#endif

#include "sharded_mps.cuh"
#include "nvqir/CircuitSimulator.h"
#include "common/NoiseModel.h"
#include "cudaq/spin_op.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

#if defined(CUDAQ_MPS_MQPU_SHOTS_BACKEND)
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mqpu-shots"
#elif defined(CUDAQ_MPS_MQPU_HAMILTONIAN_BACKEND)
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mqpu-hamiltonian"
#elif defined(CUDAQ_MPS_MQPU_STRONG_BACKEND)
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mqpu-strong"
#elif defined(CUDAQ_MPS_MQPU_REPLAY_BACKEND)
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mqpu-replay"
#elif defined(CUDAQ_MPS_MQPU_BACKEND)
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mqpu"
#else
#define CUDAQ_CUSTOM_MPS_BACKEND_NAME "tensornet-mps-mgpu"
#endif

// Round-robin GPU assignment for single-GPU MPS mode (CUDAQ_MPS_NUM_GPUS=1).
// Each new thread that calls ensureEngine() is assigned the next available GPU
// so that parallel SubspaceVQE blocks land on different physical GPUs.
#ifndef CUDAQ_MPS_MQPU_BACKEND
static std::atomic<int> g_next_gpu_rr{0};
#endif

namespace nvqir {

class SimulatorMPSDist : public nvqir::CircuitSimulatorBase<double> {
  using Base = nvqir::CircuitSimulatorBase<double>;
  using GateApplicationTask = typename Base::GateApplicationTask;

  std::unique_ptr<ShardedMPS> mps;
  std::size_t allocated = 0;     // qubits allocated so far
  int maxBond = 64;
  double absCut = 1e-5, relCut = 1e-5;
  std::vector<int> gpuIds;
  // Physical (CUDA-visible logical) devices used as Hamiltonian workers. The
  // prepared MPS itself resides only on gpuIds[0].
  std::vector<int> hbatchGpuIds;
  bool hbatchTracePrinted = false;

  // ---- Noise / trajectory support ------------------------------------------
  // Each gate applied during circuit execution is recorded so that the full
  // circuit can be replayed (from |0>^n) for each noisy trajectory.
  struct GateRecord {
    std::string operationName;
    std::vector<std::complex<double>> matrix;
    std::vector<std::size_t> controls;
    std::vector<std::size_t> targets;
    std::vector<double> parameters;
  };
  std::vector<GateRecord> m_gateHistory;
  bool m_recordingGates = true;   // false during trajectory replays
  std::mt19937_64 m_rng{std::random_device{}()};

#ifdef CUDAQ_MPS_MQPU_BACKEND
  enum class HBatchMode { CopyPreparedState, ReplayAnsatz };
  enum class HBatchEvaluation { Analytic, ShotSampled };

  struct HBatchJob {
    const ShardedMPS *source = nullptr;
    const std::vector<GateRecord> *history = nullptr;
    const std::vector<std::string> *words = nullptr;
    const std::vector<std::size_t> *activeTerms = nullptr;
    std::vector<std::vector<std::size_t>> *assignments = nullptr;
    std::vector<std::complex<double>> *values = nullptr;
    // The shared-diagonal path partitions one global shot budget over the
    // physical GPU workers. Every worker evaluates every Z/I word from one
    // shared local sample set; the primary combines the shard means weighted
    // by these exact shot counts.
    std::vector<std::vector<std::complex<double>>> *shotShardValues = nullptr;
    const std::vector<int> *shotShardCounts = nullptr;
    std::vector<std::vector<std::string>> *shotShardBitstrings = nullptr;
    bool sharedDiagonalShots = false;
    bool sharedBitstringShots = false;
    HBatchEvaluation evaluation = HBatchEvaluation::Analytic;
    int shots = 0;
    unsigned evaluationSeed = 0;
    bool useHost = false;
    bool forcePreparedStateCopy = false;
    std::size_t activeWorkerCount = 1;
  };

  struct HBatchWorker {
    std::size_t index = 0;
    int device = 0;
    std::thread thread;
    std::unique_ptr<ShardedMPS> state;
    std::vector<std::size_t> siteCapacity;
    cudaStream_t copyStream = nullptr;
    cudaEvent_t replicaReady = nullptr;
    std::size_t observedGeneration = 0;
    std::size_t observedCircuitGeneration = 0;
    std::size_t nextGate = 0;
    std::size_t completedGateCount = 0;
    std::exception_ptr gateFailure;
  };

  HBatchMode hbatchMode =
#ifdef CUDAQ_MPS_MQPU_REPLAY_BACKEND
      HBatchMode::ReplayAnsatz;
#else
      HBatchMode::CopyPreparedState;
#endif
  static constexpr bool hbatchStrongTermBatch =
#if defined(CUDAQ_MPS_MQPU_STRONG_BACKEND) ||                              \
    defined(CUDAQ_MPS_MQPU_STRONG_PIPELINE)
      true;
#else
      false;
#endif
  std::vector<std::unique_ptr<HBatchWorker>> hbatchWorkers;
  std::mutex hbatchMutex;
  std::condition_variable hbatchWorkReady;
  std::condition_variable hbatchWorkDone;
  HBatchJob hbatchJob;
  std::vector<std::exception_ptr> hbatchFailures;
  std::vector<int> hbatchWorkerGpuIds;
  std::size_t hbatchGeneration = 0;
  std::size_t hbatchCircuitGeneration = 0;
  std::atomic<std::size_t> hbatchNextTerm{0};
  std::size_t hbatchCompleted = 0;
  std::size_t hbatchReadyWorkers = 0;
  std::size_t hbatchWorkerQubits = 0;
  bool hbatchStopping = false;
  cudaEvent_t hbatchPrimaryReady = nullptr;
  // Replay-mode workers consume this queue while GPU 0 is preparing the same
  // circuit. By observe() time their persistent MPS replicas are already
  // prepared, so neither state copies nor a serialized full-circuit replay sit
  // on the energy-evaluation critical path.
  std::vector<GateRecord> hbatchGateQueue;
#endif
  // Deterministic MQPU sampling streams. Hamiltonian-term subsets deliberately
  // use matched streams so an all-Z batched energy reproduces one full-H sample
  // set. Optional shot shards offset by QPU to remain statistically independent.
  unsigned sampleSeed = 1234567u;

  // Default trajectories when noise model is set but no explicit count given.
  static constexpr std::size_t DEFAULT_OBSERVE_TRAJECTORIES = 1000;

  // Build the engine once the qubit count is known.
  //
  // tensornet-mps-mqpu:
  //   Each logical CUDA-Q QPU prepares one block MPS on its primary GPU.  The
  //   optional CUDAQ_MPS_HBATCH_GROUPS mapping (for example "0,3;1;2") gives
  //   that primary disjoint helper GPUs for internal Hamiltonian batching.
  //   With one logical QPU and no mapping, all visible GPUs remain one group.
  //
  // tensornet-mps-mgpu, CUDAQ_MPS_NUM_GPUS (default: all visible GPUs):
  //   Set to 1 to use a single GPU per circuit, enabling SubspaceVQE to run
  //   multiple blocks in parallel on different GPUs via ThreadPoolExecutor.
  //   Each new thread gets the next GPU in round-robin order so blocks on
  //   GPU 0, 1, 2, ... run concurrently without NCCL overhead.
  //   Set to N>1 to shard the MPS chain across N GPUs (original behavior).
  void ensureEngine() {
    if (mps) return;
    if (allocated < 2)
      throw std::runtime_error(CUDAQ_CUSTOM_MPS_BACKEND_NAME
                               " needs >= 2 qubits");

    gpuIds.clear();
#ifdef CUDAQ_MPS_MQPU_BACKEND
    int visibleGpuCount = 1;
    CUDA_CHECK(cudaGetDeviceCount(&visibleGpuCount));
    int primaryDevice = 0;
    CUDA_CHECK(cudaGetDevice(&primaryDevice));
    int requestedGpuCount = visibleGpuCount;
    if (const char *requested = std::getenv("CUDAQ_MPS_HBATCH_NGPUS"))
      requestedGpuCount = std::max(1, std::atoi(requested));
    requestedGpuCount = std::min(requestedGpuCount, visibleGpuCount);
    hbatchGpuIds.clear();

    if (const char *configuredGroups =
            std::getenv("CUDAQ_MPS_HBATCH_GROUPS")) {
      const std::string specification(configuredGroups);
      std::vector<bool> assigned(static_cast<std::size_t>(visibleGpuCount),
                                 false);
      bool primaryFound = false;
      bool reservedAsHelper = false;
      std::size_t groupStart = 0;
      while (groupStart <= specification.size()) {
        const auto groupEnd = specification.find(';', groupStart);
        const auto groupText = specification.substr(
            groupStart, groupEnd == std::string::npos
                            ? std::string::npos
                            : groupEnd - groupStart);
        if (groupText.empty())
          throw std::runtime_error(
              "CUDAQ_MPS_HBATCH_GROUPS contains an empty GPU group");

        std::vector<int> group;
        std::size_t itemStart = 0;
        while (itemStart <= groupText.size()) {
          const auto itemEnd = groupText.find(',', itemStart);
          const auto itemText = groupText.substr(
              itemStart, itemEnd == std::string::npos
                             ? std::string::npos
                             : itemEnd - itemStart);
          if (itemText.empty())
            throw std::runtime_error(
                "CUDAQ_MPS_HBATCH_GROUPS contains an empty GPU id");
          std::size_t parsed = 0;
          int device = 0;
          try {
            device = std::stoi(itemText, &parsed);
          } catch (...) {
            throw std::runtime_error(
                "CUDAQ_MPS_HBATCH_GROUPS contains a non-integer GPU id");
          }
          while (parsed < itemText.size() &&
                 std::isspace(static_cast<unsigned char>(itemText[parsed])))
            ++parsed;
          if (parsed != itemText.size() || device < 0 ||
              device >= visibleGpuCount)
            throw std::runtime_error(
                "CUDAQ_MPS_HBATCH_GROUPS contains an invalid GPU id");
          if (assigned[static_cast<std::size_t>(device)])
            throw std::runtime_error(
                "CUDAQ_MPS_HBATCH_GROUPS assigns a GPU more than once");
          assigned[static_cast<std::size_t>(device)] = true;
          group.push_back(device);
          if (itemEnd == std::string::npos)
            break;
          itemStart = itemEnd + 1;
        }

        if (group.front() == primaryDevice) {
          primaryFound = true;
          hbatchGpuIds.assign(
              group.begin(),
              group.begin() + std::min<std::size_t>(
                                  group.size(), requestedGpuCount));
        } else if (std::find(group.begin() + 1, group.end(), primaryDevice) !=
                   group.end()) {
          reservedAsHelper = true;
        }
        if (groupEnd == std::string::npos)
          break;
        groupStart = groupEnd + 1;
      }
      if (!primaryFound) {
        if (reservedAsHelper)
          throw std::runtime_error(
              "logical QPU GPU is reserved as a Hamiltonian-batch helper");
        hbatchGpuIds.push_back(primaryDevice);
      }
    } else {
      int logicalQpuCount = 1;
      if (const char *configuredQpus = std::getenv("CUDAQ_MQPU_NGPUS"))
        logicalQpuCount = std::max(1, std::atoi(configuredQpus));
      if (logicalQpuCount > 1) {
        // Block-parallel default: each logical QPU owns only its selected GPU.
        hbatchGpuIds.push_back(primaryDevice);
      } else {
        // Standalone/benchmark default: one QPU batches across visible GPUs.
        hbatchGpuIds.reserve(requestedGpuCount);
        for (int device = 0; device < requestedGpuCount; ++device)
          hbatchGpuIds.push_back(device);
      }
    }

    // CUDA_VISIBLE_DEVICES remaps the physical mask to dense logical devices.
    // The first entry in a group is always the prepared-state primary.
    CUDA_CHECK(cudaSetDevice(hbatchGpuIds.front()));
    gpuIds.push_back(hbatchGpuIds.front());
#else
    int G_total = 1;
    CUDA_CHECK(cudaGetDeviceCount(&G_total));
    const char* ng_env = std::getenv("CUDAQ_MPS_NUM_GPUS");
    int num_mps_gpus = ng_env ? std::max(1, std::atoi(ng_env)) : G_total;
    num_mps_gpus = std::min(num_mps_gpus, G_total);

    if (num_mps_gpus == 1) {
      // Single-GPU mode: assign this thread a GPU via round-robin so parallel
      // block workers land on different devices.
      thread_local static int t_gpu = -1;
      if (t_gpu < 0)
        t_gpu = g_next_gpu_rr.fetch_add(1) % G_total;
      gpuIds.push_back(t_gpu);
    } else {
      for (int i = 0; i < num_mps_gpus; ++i) gpuIds.push_back(i);
    }
#endif

    if (auto *b = std::getenv("CUDAQ_MPS_MAX_BOND")) maxBond = std::max(1, atoi(b));
    if (auto *a = std::getenv("CUDAQ_MPS_ABS_CUTOFF")) absCut = atof(a);
    if (auto *r = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF")) relCut = atof(r);
    mps = std::make_unique<ShardedMPS>((int)allocated, gpuIds, maxBond, absCut, relCut);
    mps->distributedBoundary = (gpuIds.size() > 1);
#ifdef CUDAQ_MPS_MQPU_BACKEND
    ensureHbatchWorkers();
    beginHbatchCircuit();
#endif
  }

  // controlled-U (1 control c, 1 target t) -> 4x4 row-major [(C*2+T)*4+(c*2+t)]
  static std::vector<std::complex<double>>
  controlledGate(const std::vector<std::complex<double>> &U) {
    std::vector<std::complex<double>> G(16, 0.0);
    for (int C=0;C<2;++C) for(int T=0;T<2;++T) for(int c=0;c<2;++c) for(int t=0;t<2;++t){
      if (C != c) continue;                       // control qubit unchanged
      std::complex<double> v = (c==0) ? (T==t ? 1.0 : 0.0) : U[T*2+t];
      G[(C*2+T)*4 + (c*2+t)] = v;
    }
    return G;
  }

  // CUDA-Q qubit index -> our site index (msb ordering, like Qpp's helper).
  int conv(std::size_t i) const { return (int)(allocated - 1 - i); }

  // ---- Internal gate application (no history recording) --------------------
  // The explicit state argument lets a persistent replay worker build the same
  // ansatz on its own GPU without touching the primary simulator state.
  void applyGateToState(ShardedMPS &state, const GateApplicationTask &task) {
    const auto &m = task.matrix;
    std::vector<std::size_t> ctr, tgt;
    auto stateConv = [&](std::size_t i) {
      return static_cast<std::size_t>(state.n - 1 - i);
    };
    for (auto i : task.controls) ctr.push_back(stateConv(i));
    for (auto i : task.targets)  tgt.push_back(stateConv(i));

    if (ctr.empty() && tgt.size() == 1) {
      std::vector<std::complex<double>> U{m[0], m[1], m[2], m[3]};
      state.apply1q(U, (int)tgt[0]); return;
    }
    if (ctr.empty() && tgt.size() == 2) {
      state.applyTwoQubit(m, (int)tgt[0], (int)tgt[1]); return;
    }
    if (ctr.size() == 1 && tgt.size() == 1) {
      std::vector<std::complex<double>> U{m[0], m[1], m[2], m[3]};
      state.applyTwoQubit(controlledGate(U), (int)ctr[0], (int)tgt[0]); return;
    }
    throw std::runtime_error(
        CUDAQ_CUSTOM_MPS_BACKEND_NAME
        ": >2-qubit or multi-control gates not yet supported "
        "(decompose, or extend applyGate).");
  }

  void applyGateInternal(const GateApplicationTask &task) {
    applyGateToState(*mps, task);
  }

  void replayCircuitOnState(ShardedMPS &state,
                            const std::vector<GateRecord> &history) {
    state.resetZeroInPlace();
    for (const auto &gate : history) {
      GateApplicationTask task(gate.operationName, gate.matrix, gate.controls,
                               gate.targets, gate.parameters);
      applyGateToState(state, task);
    }
  }

  // ---- Kraus channel application helpers -----------------------------------

  // Apply one Kraus channel to the MPS via quantum trajectory selection.
  void applyKrausChannel(const std::vector<std::size_t> &targets,
                         const cudaq::kraus_channel &ch) {
    using cd = std::complex<double>;
    if (ch.is_unitary_mixture()) {
      // Sample one unitary from the mixture with the given probabilities.
      std::discrete_distribution<int> dist(ch.probabilities.begin(),
                                           ch.probabilities.end());
      int idx = dist(m_rng);
      const auto &U = ch.unitary_ops[idx]; // std::vector<cd>, row-major
      if (targets.size() == 1) {
        mps->apply1q(U, conv(targets[0]));
      } else if (targets.size() == 2) {
        mps->applyTwoQubit(U, conv(targets[0]), conv(targets[1]));
      } else {
        throw std::runtime_error(
            CUDAQ_CUSTOM_MPS_BACKEND_NAME
            ": noise channel on >2 qubits not supported");
      }
      // Unitary preserves norm; no renormalization needed.
    } else {
      // General (non-unitary-mixture) Kraus channel.
      if (targets.size() == 1) {
        applyGeneralKraus1q(targets[0], ch.get_ops());
      } else {
        throw std::runtime_error(
            CUDAQ_CUSTOM_MPS_BACKEND_NAME
            ": general (non-unitary-mixture) Kraus channels "
            "on 2-qubit targets are not supported; use depolarization or "
            "Pauli2 channels instead.");
      }
    }
  }

  // Quantum trajectory application of a general 1-qubit Kraus channel.
  // Computes p_m = Tr(K_m†K_m · ρ_q) from the local reduced density matrix,
  // samples m, applies K_m, and renormalizes the MPS.
  void applyGeneralKraus1q(std::size_t cudaq_qubit,
                            const std::vector<cudaq::kraus_op> &ops) {
    using cd = std::complex<double>;
    const int site = conv(cudaq_qubit);

    // Build single-qubit Pauli words for the target qubit.
    std::string wZ(mps->n, 'I'); wZ[site] = 'Z';

    // <Z> always needed for diagonal rho elements.
    double Iz = mps->expectPauli(wZ).real();
    double rho00 = (1.0 + Iz) / 2.0;
    double rho11 = (1.0 - Iz) / 2.0;

    // Check if any K†K has off-diagonal elements (needs <X>, <Y> too).
    bool needOffDiag = false;
    for (const auto &op : ops) {
      const auto &d = op.data;
      cd KdK01 = std::conj(d[0])*d[1] + std::conj(d[2])*d[3];
      if (std::abs(KdK01) > 1e-10) { needOffDiag = true; break; }
    }
    cd rho10{0.0, 0.0};
    if (needOffDiag) {
      std::string wX(mps->n, 'I'); wX[site] = 'X';
      std::string wY(mps->n, 'I'); wY[site] = 'Y';
      double Ix = mps->expectPauli(wX).real();
      double Iy = mps->expectPauli(wY).real();
      // ρ_q = (I + Ix*X + Iy*Y + Iz*Z) / 2
      // ρ[1][0] = (Ix + i*Iy) / 2  (ket=0, bra=1 convention)
      rho10 = cd(Ix, Iy) / 2.0;
    }

    // Compute p_m = Tr(K_m†K_m · ρ_q) for each Kraus op.
    std::vector<double> probs;
    probs.reserve(ops.size());
    for (const auto &op : ops) {
      const auto &d = op.data;
      // K†K diagonal elements
      cd KdK00 = std::conj(d[0])*d[0] + std::conj(d[2])*d[2];
      cd KdK11 = std::conj(d[1])*d[1] + std::conj(d[3])*d[3];
      double p = KdK00.real()*rho00 + KdK11.real()*rho11;
      if (needOffDiag) {
        cd KdK01 = std::conj(d[0])*d[1] + std::conj(d[2])*d[3];
        // 2*Re(K†K[0][1] * ρ[1][0])  (off-diagonal contribution)
        p += 2.0 * (KdK01 * rho10).real();
      }
      probs.push_back(std::max(0.0, p));
    }

    // Normalize (completeness: sum should be 1 up to numerical drift).
    double total = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (total > 1e-15)
      for (auto &p : probs) p /= total;
    else
      std::fill(probs.begin(), probs.end(), 1.0 / probs.size()); // fallback

    // Sample Kraus operator index.
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int idx = dist(m_rng);

    // Apply selected Kraus operator.
    const auto &d = ops[idx].data;
    std::vector<cd> K{cd(d[0]), cd(d[1]), cd(d[2]), cd(d[3])};
    mps->apply1q(K, site);

    // Renormalize state by 1/sqrt(p_idx).
    double nrm = std::sqrt(probs[idx]);
    if (nrm > 1e-15) {
      cd s = 1.0 / nrm;
      std::vector<cd> scaleOp{s, 0.0, 0.0, s};   // diagonal scaling matrix
      mps->apply1q(scaleOp, site);
    }
  }

  // ---- Trajectory replay ---------------------------------------------------
  // Re-execute the saved gate history from |0>^n, applying noise after each
  // gate. Used by observe() and sample() for multi-trajectory averaging.
  // Must be called with m_recordingGates = false and mps already resetZero().
  void replayCircuitWithNoise(const std::vector<GateRecord> &history,
                              cudaq::ExecutionContext *ctx) {
    for (const auto &g : history) {
      GateApplicationTask task(g.operationName, g.matrix, g.controls,
                               g.targets, g.parameters);
      applyGateInternal(task);
      // Manually invoke noise channel lookup (mirrors base class flushGateQueue).
      if (ctx && ctx->noiseModel && !ctx->noiseModel->empty()) {
        auto channels = ctx->noiseModel->get_channels(
            g.operationName, g.targets, g.controls, g.parameters);
        for (const auto &ch : channels)
          applyKrausChannel(g.targets, ch);
      }
    }
  }

  // ---- Pauli word construction for observe ---------------------------------
  struct PauliTermData {
    std::vector<std::string> words;
    std::vector<std::complex<double>> coeffs;
    std::vector<std::string> termIds;
  };

  PauliTermData buildPauliTermData(const cudaq::spin_op &op) const {
    PauliTermData pd;
    const int nq = mps->n;
    for (const auto &term : op) {
      std::string word = term.get_pauli_word(nq);
      // Gates route CUDA-Q qubit q to MPS site conv(q)=n-1-q, so reverse.
      std::reverse(word.begin(), word.end());
      pd.words.push_back(std::move(word));
      pd.coeffs.push_back(term.evaluate_coefficient());
      pd.termIds.push_back(term.get_term_id());
    }
    return pd;
  }

#ifdef CUDAQ_MPS_MQPU_BACKEND
  static unsigned hbatchSeedForWord(unsigned evaluationSeed,
                                    const std::string &word) {
    unsigned hash = 2166136261u ^ evaluationSeed;
    for (unsigned char value : word) {
      hash ^= static_cast<unsigned>(value);
      hash *= 16777619u;
    }
    return hash;
  }

  static unsigned hbatchSeedForShotShard(unsigned evaluationSeed,
                                         std::size_t workerIndex) {
    // SplitMix-style integer mixing gives deterministic, independent streams
    // without making the result depend on Hamiltonian-term ordering.
    unsigned value = evaluationSeed +
                     static_cast<unsigned>(0x9e3779b9U * (workerIndex + 1));
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
  }

  static std::vector<std::complex<double>>
  sampleSharedDiagonalWords(ShardedMPS &state,
                            const std::vector<std::string> &words,
                            int shots, unsigned seed) {
    const auto sampled = state.sampleZExpectationsGpu(words, shots, seed);
    std::vector<std::complex<double>> values(sampled.size());
    for (std::size_t term = 0; term < sampled.size(); ++term)
      values[term] = sampled[term];
    return values;
  }

  std::complex<double> evaluateHbatchTerm(
      ShardedMPS &state, const std::string &word,
      HBatchEvaluation evaluation, int shots, unsigned evaluationSeed,
      bool useHost) {
    if (evaluation == HBatchEvaluation::Analytic)
      return useHost ? state.expectPauliHost(word) : state.expectPauli(word);

    const bool zOnly = std::all_of(
        word.begin(), word.end(),
        [](char value) { return value == 'I' || value == 'Z'; });
    if (!zOnly)
      return state.expectPauli(word);

    // Deliberately evaluate every Hamiltonian term independently. This backend
    // implements Hamiltonian batching and does not fuse diagonal observations.
    const std::vector<std::string> oneWord{word};
    return state.sampleZExpectationsGpu(
        oneWord, shots, hbatchSeedForWord(evaluationSeed, word)).front();
  }

  void evaluateHbatchTermBatch(
      ShardedMPS &state, const std::vector<std::size_t> &terms,
      const std::vector<std::string> &words,
      std::vector<std::complex<double>> &values,
      HBatchEvaluation evaluation, int shots, unsigned evaluationSeed,
      bool useHost) {
    if (evaluation == HBatchEvaluation::Analytic) {
      const char *mode = std::getenv("CUDAQ_MPS_ANALYTIC_MODE");
      if (useHost || (mode && std::string(mode)=="reference")) {
        for (const auto term : terms)
          values[term] = evaluateHbatchTerm(state, words[term], evaluation, shots, evaluationSeed, useHost);
      } else {
        std::vector<std::string> selected;
        for(auto term:terms)selected.push_back(words[term]);
        auto batch = mode && std::string(mode)=="batch"
            ? state.expectPauliBatch(selected) : state.expectPauliShared(selected);
        for(size_t i=0;i<terms.size();++i)values[terms[i]]=batch[i];
      }
      return;
    }

    std::vector<std::size_t> zTerms;
    std::vector<std::string> zWords;
    std::vector<unsigned> zSeeds;
    zTerms.reserve(terms.size());
    zWords.reserve(terms.size());
    zSeeds.reserve(terms.size());
    for (const auto term : terms) {
      const bool zOnly = std::all_of(
          words[term].begin(), words[term].end(),
          [](char value) { return value == 'I' || value == 'Z'; });
      if (zOnly) {
        zTerms.push_back(term);
        zWords.push_back(words[term]);
        zSeeds.push_back(hbatchSeedForWord(evaluationSeed, words[term]));
      } else {
        values[term] = state.expectPauli(words[term]);
      }
    }
    // The Hamiltonian target partitions terms across physical GPUs, but a
    // commuting diagonal partition must still reuse one measurement stream.
    // Every worker deliberately uses the same deterministic stream seed: the
    // prepared replicas therefore produce the same global `shots` bitstrings,
    // while each GPU evaluates only its assigned term masks.  This makes the
    // one-GPU workload identical to the shared-shot target and prevents the
    // old terms*shots sampling explosion.  Other MQPU targets retain their
    // existing independent-term fallback for mixed-basis observations.
#if defined(CUDAQ_MPS_MQPU_HAMILTONIAN_BACKEND)
    const auto zValues = state.sampleZExpectationsGpu(
        zWords, shots, hbatchSeedForShotShard(evaluationSeed, 0));
#else
    const auto zValues = state.sampleZExpectationsIndependentGpu(
        zWords, shots, zSeeds);
#endif
    for (std::size_t index = 0; index < zTerms.size(); ++index)
      values[zTerms[index]] = zValues[index];
  }

  void updatePersistentReplica(HBatchWorker &worker,
                               const ShardedMPS &source) {
    CUDA_CHECK(cudaSetDevice(worker.device));
    CUDA_CHECK(cudaStreamWaitEvent(worker.copyStream, hbatchPrimaryReady, 0));

    for (int siteIndex = 0; siteIndex < source.n; ++siteIndex) {
      const auto &input = source.site[siteIndex];
      auto &output = worker.state->site[siteIndex];
      const std::size_t elements =
          static_cast<std::size_t>(input.Dl) * 2 * input.Dr;
      const std::size_t bytes = sizeof(cdouble) * elements;

      if (worker.siteCapacity[siteIndex] < elements) {
        // Shape growth happens during the first observation. Thereafter the
        // fixed ansatz topology normally reuses these allocations unchanged.
        CUDA_CHECK(cudaStreamSynchronize(worker.copyStream));
        CUDA_CHECK(cudaFree(output.d));
        CUDA_CHECK(cudaMalloc(&output.d, bytes));
        worker.siteCapacity[siteIndex] = elements;
      }

      if (input.dev == worker.device) {
        CUDA_CHECK(cudaMemcpyAsync(output.d, input.d, bytes,
                                   cudaMemcpyDeviceToDevice,
                                   worker.copyStream));
      } else {
        CUDA_CHECK(cudaMemcpyPeerAsync(output.d, worker.device, input.d,
                                       input.dev, bytes,
                                       worker.copyStream));
      }
      output.Dl = input.Dl;
      output.Dr = input.Dr;
      output.dev = worker.device;
    }
    worker.state->seedCounter = source.seedCounter;
    CUDA_CHECK(cudaEventRecord(worker.replicaReady, worker.copyStream));
    // Observation kernels use this worker thread's default CUDA stream. Make
    // that stream depend on the asynchronous replica update without blocking
    // the host worker thread in cudaStreamSynchronize().
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, worker.replicaReady, 0));
  }

  void beginHbatchCircuit() {
    if (hbatchMode != HBatchMode::ReplayAnsatz || hbatchWorkers.empty())
      return;
    {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      hbatchGateQueue.clear();
      ++hbatchCircuitGeneration;
    }
    hbatchWorkReady.notify_all();
  }

  void enqueueHbatchGate(const GateRecord &gate) {
    if (hbatchMode != HBatchMode::ReplayAnsatz || hbatchWorkers.empty())
      return;
    {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      hbatchGateQueue.push_back(gate);
    }
    hbatchWorkReady.notify_all();
  }

  void waitForStreamedReplay() {
    if (hbatchMode != HBatchMode::ReplayAnsatz || hbatchWorkers.empty())
      return;

    std::unique_lock<std::mutex> lock(hbatchMutex);
    const auto expectedCircuit = hbatchCircuitGeneration;
    const auto expectedGates = hbatchGateQueue.size();
    hbatchWorkDone.wait(lock, [&] {
      return std::all_of(
          hbatchWorkers.begin(), hbatchWorkers.end(),
          [&](const auto &worker) {
            return worker->gateFailure ||
                   (worker->observedCircuitGeneration == expectedCircuit &&
                    worker->completedGateCount == expectedGates);
          });
    });
    for (const auto &worker : hbatchWorkers)
      if (worker->gateFailure)
        std::rethrow_exception(worker->gateFailure);
  }

  void hbatchWorkerLoop(HBatchWorker &worker) {
    try {
      CUDA_CHECK(cudaSetDevice(worker.device));
      CUDA_CHECK(cudaStreamCreateWithFlags(&worker.copyStream,
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreateWithFlags(&worker.replicaReady,
                                          cudaEventDisableTiming));
      worker.state = std::make_unique<ShardedMPS>(
          static_cast<int>(hbatchWorkerQubits),
          std::vector<int>{worker.device}, maxBond, absCut, relCut);
      worker.siteCapacity.assign(hbatchWorkerQubits, 2);
    } catch (...) {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      hbatchFailures[worker.index] = std::current_exception();
      ++hbatchReadyWorkers;
      hbatchWorkDone.notify_all();
      return;
    }

    {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      ++hbatchReadyWorkers;
      hbatchWorkDone.notify_all();
    }

    while (true) {
      HBatchJob job;
      GateRecord gate;
      enum class Work { Reset, Gate, Evaluation } work;
      {
        std::unique_lock<std::mutex> lock(hbatchMutex);
        hbatchWorkReady.wait(lock, [&] {
          return hbatchStopping ||
                 (hbatchMode == HBatchMode::ReplayAnsatz &&
                  (worker.observedCircuitGeneration !=
                       hbatchCircuitGeneration ||
                   (!worker.gateFailure &&
                    worker.nextGate < hbatchGateQueue.size()))) ||
                 hbatchGeneration != worker.observedGeneration;
        });
        if (hbatchStopping)
          break;

        if (hbatchMode == HBatchMode::ReplayAnsatz &&
            worker.observedCircuitGeneration != hbatchCircuitGeneration) {
          worker.observedCircuitGeneration = hbatchCircuitGeneration;
          worker.nextGate = 0;
          worker.completedGateCount = 0;
          worker.gateFailure = nullptr;
          work = Work::Reset;
        } else if (hbatchMode == HBatchMode::ReplayAnsatz &&
                   !worker.gateFailure &&
                   worker.nextGate < hbatchGateQueue.size()) {
          gate = hbatchGateQueue[worker.nextGate++];
          work = Work::Gate;
        } else {
          worker.observedGeneration = hbatchGeneration;
          job = hbatchJob;
          work = Work::Evaluation;
        }
      }

      if (work == Work::Reset) {
        try {
          worker.state->resetZeroInPlace();
        } catch (...) {
          std::lock_guard<std::mutex> lock(hbatchMutex);
          worker.gateFailure = std::current_exception();
        }
        hbatchWorkDone.notify_all();
        continue;
      }

      if (work == Work::Gate) {
        try {
          GateApplicationTask task(gate.operationName, gate.matrix,
                                   gate.controls, gate.targets,
                                   gate.parameters);
          applyGateToState(*worker.state, task);
        } catch (...) {
          std::lock_guard<std::mutex> lock(hbatchMutex);
          worker.gateFailure = std::current_exception();
        }
        {
          std::lock_guard<std::mutex> lock(hbatchMutex);
          worker.completedGateCount = worker.nextGate;
        }
        hbatchWorkDone.notify_all();
        continue;
      }

      if (worker.index >= job.activeWorkerCount)
        continue;

      std::exception_ptr failure;
      try {
        if (hbatchMode == HBatchMode::CopyPreparedState ||
            job.forcePreparedStateCopy) {
          updatePersistentReplica(worker, *job.source);
        }

        if (job.sharedBitstringShots) {
          const int workerShots = job.shotShardCounts->at(worker.index);
          job.shotShardBitstrings->at(worker.index) =
              worker.state->sampleBits(
                  workerShots,
                  hbatchSeedForShotShard(job.evaluationSeed, worker.index));
        } else if (job.sharedDiagonalShots) {
          const int workerShots = job.shotShardCounts->at(worker.index);
          job.shotShardValues->at(worker.index) = sampleSharedDiagonalWords(
              *worker.state, *job.words, workerShots,
              hbatchSeedForShotShard(job.evaluationSeed, worker.index));
        } else if (hbatchStrongTermBatch || job.evaluation == HBatchEvaluation::Analytic) {
          evaluateHbatchTermBatch(
              *worker.state, job.assignments->at(worker.index), *job.words,
              *job.values, job.evaluation, job.shots, job.evaluationSeed,
              job.useHost);
        } else {
          while (true) {
            const auto position = hbatchNextTerm.fetch_add(
                1, std::memory_order_relaxed);
            if (position >= job.activeTerms->size())
              break;
            const auto term = (*job.activeTerms)[position];
            // Each worker owns its assignment vector, so recording the dynamic
            // schedule requires no lock and makes load balancing visible in
            // the trace output.
            job.assignments->at(worker.index).push_back(term);
            (*job.values)[term] = evaluateHbatchTerm(
                *worker.state, (*job.words)[term], job.evaluation, job.shots,
                job.evaluationSeed, job.useHost);
          }
        }
      } catch (...) {
        failure = std::current_exception();
      }

      {
        std::lock_guard<std::mutex> lock(hbatchMutex);
        hbatchFailures[worker.index] = failure;
        ++hbatchCompleted;
        hbatchWorkDone.notify_all();
      }
    }

    worker.state.reset();
    if (worker.replicaReady) {
      cudaSetDevice(worker.device);
      cudaEventDestroy(worker.replicaReady);
      worker.replicaReady = nullptr;
    }
    if (worker.copyStream) {
      cudaSetDevice(worker.device);
      cudaStreamDestroy(worker.copyStream);
      worker.copyStream = nullptr;
    }
  }

  void shutdownHbatchWorkers() {
    {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      hbatchStopping = true;
      ++hbatchGeneration;
    }
    hbatchWorkReady.notify_all();
    for (auto &worker : hbatchWorkers)
      if (worker->thread.joinable())
        worker->thread.join();
    hbatchWorkers.clear();
    hbatchFailures.clear();
    hbatchReadyWorkers = 0;
    hbatchCompleted = 0;
    hbatchWorkerQubits = 0;
    hbatchWorkerGpuIds.clear();
    // New workers must not consume the previous job's dangling pointers.
    // Its vectors and source MPS have already been destroyed after observe.
    hbatchGeneration = 0;
    hbatchCircuitGeneration = 0;
    hbatchJob = HBatchJob{};
    hbatchGateQueue.clear();
    hbatchStopping = false;
    if (hbatchPrimaryReady) {
      cudaSetDevice(hbatchGpuIds.empty() ? 0 : hbatchGpuIds.front());
      cudaEventDestroy(hbatchPrimaryReady);
      hbatchPrimaryReady = nullptr;
    }
  }

  void ensureHbatchWorkers() {
    if (!hbatchWorkers.empty()) {
      if (hbatchWorkerQubits == allocated && hbatchWorkerGpuIds == hbatchGpuIds)
        return;
      shutdownHbatchWorkers();
    }
    if (hbatchGpuIds.size() <= 1)
      return;

    if (const char *requestedMode = std::getenv("CUDAQ_MPS_HBATCH_MODE")) {
      const std::string mode(requestedMode);
      if (mode == "copy")
        hbatchMode = HBatchMode::CopyPreparedState;
      else if (mode == "replay")
        hbatchMode = HBatchMode::ReplayAnsatz;
      else
        throw std::runtime_error(
            "CUDAQ_MPS_HBATCH_MODE must be 'copy' or 'replay'");
    }

    hbatchWorkerQubits = allocated;
    hbatchWorkerGpuIds = hbatchGpuIds;
    hbatchFailures.assign(hbatchGpuIds.size(), nullptr);
    hbatchReadyWorkers = 0;
    CUDA_CHECK(cudaSetDevice(hbatchGpuIds.front()));
    CUDA_CHECK(cudaEventCreateWithFlags(&hbatchPrimaryReady,
                                        cudaEventDisableTiming));

    for (std::size_t index = 1; index < hbatchGpuIds.size(); ++index) {
      auto worker = std::make_unique<HBatchWorker>();
      worker->index = index;
      worker->device = hbatchGpuIds[index];
      auto *workerPointer = worker.get();
      worker->thread = std::thread(
          [this, workerPointer] { hbatchWorkerLoop(*workerPointer); });
      hbatchWorkers.push_back(std::move(worker));
    }

    {
      std::unique_lock<std::mutex> lock(hbatchMutex);
      hbatchWorkDone.wait(lock, [&] {
        return hbatchReadyWorkers == hbatchWorkers.size();
      });
    }
    for (std::size_t index = 1; index < hbatchFailures.size(); ++index) {
      if (hbatchFailures[index]) {
        auto failure = hbatchFailures[index];
        shutdownHbatchWorkers();
        std::rethrow_exception(failure);
      }
    }
  }

  std::vector<std::complex<double>> runPersistentHamiltonianBatch(
      const std::vector<std::string> &words, HBatchEvaluation evaluation,
      int shots, unsigned evaluationSeed, bool useHost,
      bool forcePreparedStateCopy) {
    std::vector<std::complex<double>> values(words.size(), 1.0);
    std::vector<std::size_t> activeTerms;
    activeTerms.reserve(words.size());
    for (std::size_t term = 0; term < words.size(); ++term) {
      const bool identity = std::all_of(
          words[term].begin(), words[term].end(),
          [](char value) { return value == 'I'; });
      if (!identity)
        activeTerms.push_back(term);
    }
    if (activeTerms.empty())
      return values;

    // A replay worker cannot reconstruct the primary's stochastic trajectory.
    // Keep noisy-trajectory expectations on that exact primary state.
    if (forcePreparedStateCopy && hbatchMode == HBatchMode::ReplayAnsatz) {
      for (const auto term : activeTerms)
        values[term] = evaluateHbatchTerm(
            *mps, words[term], evaluation, shots, evaluationSeed, useHost);
      return values;
    }

    const std::size_t workerCount = std::min(
        hbatchGpuIds.size(), activeTerms.size());
    if (workerCount == 1) {
      if (hbatchStrongTermBatch || evaluation == HBatchEvaluation::Analytic) {
        evaluateHbatchTermBatch(*mps, activeTerms, words, values, evaluation,
                                shots, evaluationSeed, useHost);
      } else {
        for (const auto term : activeTerms)
          values[term] = evaluateHbatchTerm(
              *mps, words[term], evaluation, shots, evaluationSeed, useHost);
      }
      return values;
    }

    ensureHbatchWorkers();
    waitForStreamedReplay();
    std::vector<std::vector<std::size_t>> assignments(workerCount);
    if (hbatchStrongTermBatch || evaluation == HBatchEvaluation::Analytic)
      for (std::size_t position = 0; position < activeTerms.size(); ++position)
        assignments[position % workerCount].push_back(activeTerms[position]);

    if (hbatchMode == HBatchMode::CopyPreparedState ||
        forcePreparedStateCopy) {
      CUDA_CHECK(cudaSetDevice(mps->site.front().dev));
      CUDA_CHECK(cudaEventRecord(hbatchPrimaryReady, nullptr));
    }

    {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      std::fill(hbatchFailures.begin(), hbatchFailures.end(), nullptr);
      hbatchCompleted = 0;
      hbatchNextTerm.store(0, std::memory_order_relaxed);
      hbatchJob = HBatchJob{};
      hbatchJob.source = mps.get();
      hbatchJob.history = &m_gateHistory;
      hbatchJob.words = &words;
      hbatchJob.activeTerms = &activeTerms;
      hbatchJob.assignments = &assignments;
      hbatchJob.values = &values;
      hbatchJob.evaluation = evaluation;
      hbatchJob.shots = shots;
      hbatchJob.evaluationSeed = evaluationSeed;
      hbatchJob.useHost = useHost;
      hbatchJob.forcePreparedStateCopy = forcePreparedStateCopy;
      hbatchJob.activeWorkerCount = workerCount;
      ++hbatchGeneration;
    }
    hbatchWorkReady.notify_all();

    std::exception_ptr primaryFailure;
    try {
      if (hbatchStrongTermBatch || evaluation == HBatchEvaluation::Analytic) {
        evaluateHbatchTermBatch(
            *mps, assignments.front(), words, values, evaluation, shots,
            evaluationSeed, useHost);
      } else {
        while (true) {
          const auto position = hbatchNextTerm.fetch_add(
              1, std::memory_order_relaxed);
          if (position >= activeTerms.size())
            break;
          const auto term = activeTerms[position];
          assignments.front().push_back(term);
          values[term] = evaluateHbatchTerm(
              *mps, words[term], evaluation, shots, evaluationSeed, useHost);
        }
      }
    } catch (...) {
      primaryFailure = std::current_exception();
    }

    {
      std::unique_lock<std::mutex> lock(hbatchMutex);
      hbatchWorkDone.wait(lock, [&] {
        return hbatchCompleted == workerCount - 1;
      });
    }
    if (primaryFailure)
      std::rethrow_exception(primaryFailure);
    for (std::size_t index = 1; index < workerCount; ++index)
      if (hbatchFailures[index])
        std::rethrow_exception(hbatchFailures[index]);

    if (!hbatchTracePrinted) {
      const char *trace = std::getenv("CUDAQ_MPS_HBATCH_TRACE");
      if (trace && std::atoi(trace) != 0) {
        const char *mode = hbatchStrongTermBatch
                               ? "persistent-copy-term-pipeline"
                               : (hbatchMode == HBatchMode::ReplayAnsatz
                                      ? "persistent-streamed-replay"
                                      : "persistent-copy");
        std::fprintf(stderr,
                     "[%s] Hamiltonian batch: %zu terms, %zu persistent "
                     "physical GPU workers, mode=%s\n",
                     CUDAQ_CUSTOM_MPS_BACKEND_NAME, activeTerms.size(),
                     workerCount, mode);
        for (std::size_t index = 0; index < workerCount; ++index)
          std::fprintf(stderr,
                       "[%s] worker %zu: CUDA device %d, %zu terms\n",
                       CUDAQ_CUSTOM_MPS_BACKEND_NAME, index,
                       hbatchGpuIds[index], assignments[index].size());
      }
      hbatchTracePrinted = true;
    }
    return values;
  }

  std::vector<std::complex<double>> runSharedDiagonalShotBatch(
      const std::vector<std::string> &words, int shots,
      unsigned evaluationSeed) {
    if (words.empty())
      return {};
    if (shots <= 0)
      throw std::runtime_error(
          "shared diagonal sampling requires a positive shot count");

    const std::size_t workerCount = std::min<std::size_t>(
        hbatchGpuIds.size(), static_cast<std::size_t>(shots));
    if (workerCount == 1)
      return sampleSharedDiagonalWords(
          *mps, words, shots,
          hbatchSeedForShotShard(evaluationSeed, 0));

    ensureHbatchWorkers();
    waitForStreamedReplay();

    std::vector<int> shardShots(workerCount, shots / workerCount);
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(shots % workerCount); ++index)
      ++shardShots[index];
    std::vector<std::vector<std::complex<double>>> shardValues(workerCount);
    std::vector<std::vector<std::size_t>> assignments(workerCount);
    std::vector<std::size_t> allTerms(words.size());
    std::iota(allTerms.begin(), allTerms.end(), 0);
    std::vector<std::complex<double>> unusedValues(words.size(), 1.0);

    if (hbatchMode == HBatchMode::CopyPreparedState) {
      CUDA_CHECK(cudaSetDevice(mps->site.front().dev));
      CUDA_CHECK(cudaEventRecord(hbatchPrimaryReady, nullptr));
    }

    {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      std::fill(hbatchFailures.begin(), hbatchFailures.end(), nullptr);
      hbatchCompleted = 0;
      hbatchJob = HBatchJob{};
      hbatchJob.source = mps.get();
      hbatchJob.history = &m_gateHistory;
      hbatchJob.words = &words;
      hbatchJob.activeTerms = &allTerms;
      hbatchJob.assignments = &assignments;
      hbatchJob.values = &unusedValues;
      hbatchJob.shotShardValues = &shardValues;
      hbatchJob.shotShardCounts = &shardShots;
      hbatchJob.sharedDiagonalShots = true;
      hbatchJob.evaluation = HBatchEvaluation::ShotSampled;
      hbatchJob.shots = shots;
      hbatchJob.evaluationSeed = evaluationSeed;
      hbatchJob.activeWorkerCount = workerCount;
      ++hbatchGeneration;
    }
    hbatchWorkReady.notify_all();

    std::exception_ptr primaryFailure;
    try {
      shardValues.front() = sampleSharedDiagonalWords(
          *mps, words, shardShots.front(),
          hbatchSeedForShotShard(evaluationSeed, 0));
    } catch (...) {
      primaryFailure = std::current_exception();
    }

    {
      std::unique_lock<std::mutex> lock(hbatchMutex);
      hbatchWorkDone.wait(lock, [&] {
        return hbatchCompleted == workerCount - 1;
      });
    }
    if (primaryFailure)
      std::rethrow_exception(primaryFailure);
    for (std::size_t index = 1; index < workerCount; ++index)
      if (hbatchFailures[index])
        std::rethrow_exception(hbatchFailures[index]);

    std::vector<std::complex<double>> values(words.size(), 0.0);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
      if (shardValues[worker].size() != words.size())
        throw std::runtime_error(
            "shared diagonal sampling worker returned the wrong term count");
      const double weight =
          static_cast<double>(shardShots[worker]) / shots;
      for (std::size_t term = 0; term < words.size(); ++term)
        values[term] += weight * shardValues[worker][term];
    }

    if (!hbatchTracePrinted) {
      const char *trace = std::getenv("CUDAQ_MPS_HBATCH_TRACE");
      if (trace && std::atoi(trace) != 0) {
        std::fprintf(stderr,
                     "[%s] shared diagonal sampling: %zu terms, %d total "
                     "shots, %zu physical GPU shot shards\n",
                     CUDAQ_CUSTOM_MPS_BACKEND_NAME, words.size(), shots,
                     workerCount);
        for (std::size_t index = 0; index < workerCount; ++index)
          std::fprintf(stderr,
                       "[%s] worker %zu: CUDA device %d, %d shots, %zu "
                       "shared terms\n",
                       CUDAQ_CUSTOM_MPS_BACKEND_NAME, index,
                       hbatchGpuIds[index], shardShots[index], words.size());
      }
      hbatchTracePrinted = true;
    }
    return values;
  }

  std::vector<std::string> runSharedBitstringShotBatch(
      int shots, unsigned evaluationSeed) {
    if (shots <= 0)
      throw std::runtime_error(
          "shared bitstring sampling requires a positive shot count");

    const std::size_t workerCount = std::min<std::size_t>(
        hbatchGpuIds.size(), static_cast<std::size_t>(shots));
    if (workerCount == 1)
      return mps->sampleBits(
          shots, hbatchSeedForShotShard(evaluationSeed, 0));

    ensureHbatchWorkers();
    waitForStreamedReplay();

    std::vector<int> shardShots(workerCount, shots / workerCount);
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(shots % workerCount); ++index)
      ++shardShots[index];
    std::vector<std::vector<std::string>> shardBitstrings(workerCount);

    // sample() needs an exact copy of the final prepared state. This also
    // keeps the result correct when the optional replay backend is selected.
    CUDA_CHECK(cudaSetDevice(mps->site.front().dev));
    CUDA_CHECK(cudaEventRecord(hbatchPrimaryReady, nullptr));

    {
      std::lock_guard<std::mutex> lock(hbatchMutex);
      std::fill(hbatchFailures.begin(), hbatchFailures.end(), nullptr);
      hbatchCompleted = 0;
      hbatchJob = HBatchJob{};
      hbatchJob.source = mps.get();
      hbatchJob.shotShardCounts = &shardShots;
      hbatchJob.shotShardBitstrings = &shardBitstrings;
      hbatchJob.sharedBitstringShots = true;
      hbatchJob.evaluationSeed = evaluationSeed;
      hbatchJob.forcePreparedStateCopy = true;
      hbatchJob.activeWorkerCount = workerCount;
      ++hbatchGeneration;
    }
    hbatchWorkReady.notify_all();

    std::exception_ptr primaryFailure;
    try {
      shardBitstrings.front() = mps->sampleBits(
          shardShots.front(),
          hbatchSeedForShotShard(evaluationSeed, 0));
    } catch (...) {
      primaryFailure = std::current_exception();
    }

    {
      std::unique_lock<std::mutex> lock(hbatchMutex);
      hbatchWorkDone.wait(lock, [&] {
        return hbatchCompleted == workerCount - 1;
      });
    }
    if (primaryFailure)
      std::rethrow_exception(primaryFailure);
    for (std::size_t index = 1; index < workerCount; ++index)
      if (hbatchFailures[index])
        std::rethrow_exception(hbatchFailures[index]);

    std::vector<std::string> bitstrings;
    bitstrings.reserve(static_cast<std::size_t>(shots));
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
      if (shardBitstrings[worker].size() !=
          static_cast<std::size_t>(shardShots[worker]))
        throw std::runtime_error(
            "shared bitstring sampling worker returned the wrong shot count");
      bitstrings.insert(bitstrings.end(), shardBitstrings[worker].begin(),
                        shardBitstrings[worker].end());
    }

    if (!hbatchTracePrinted) {
      const char *trace = std::getenv("CUDAQ_MPS_HBATCH_TRACE");
      if (trace && std::atoi(trace) != 0) {
        std::fprintf(stderr,
                     "[%s] shared bitstring sampling: %d total shots, %zu "
                     "persistent physical GPU shot shards\n",
                     CUDAQ_CUSTOM_MPS_BACKEND_NAME, shots, workerCount);
        for (std::size_t index = 0; index < workerCount; ++index)
          std::fprintf(stderr,
                       "[%s] worker %zu: CUDA device %d, %d shots\n",
                       CUDAQ_CUSTOM_MPS_BACKEND_NAME, index,
                       hbatchGpuIds[index], shardShots[index]);
      }
      hbatchTracePrinted = true;
    }
    return bitstrings;
  }
#endif

public:
  SimulatorMPSDist() = default;
  ~SimulatorMPSDist() override {
#ifdef CUDAQ_MPS_MQPU_BACKEND
    shutdownHbatchWorkers();
#endif
  }

  // Bring both base addQubitsToState overloads into scope (we only specialize
  // the (count, ptr) one) to avoid a partial-override error.
  using nvqir::CircuitSimulatorBase<double>::addQubitsToState;

  // ---- required CircuitSimulatorBase overrides ----
  void addQubitToState() override { addQubitsToState(1); }

  void addQubitsToState(std::size_t count, const void * = nullptr) override {
    if (mps) throw std::runtime_error(
        CUDAQ_CUSTOM_MPS_BACKEND_NAME
        ": adding qubits after gates is not supported");
    allocated += count;
  }

  void deallocateStateImpl() override {
    mps.reset(); allocated = 0;
    m_gateHistory.clear(); m_recordingGates = true;
  }

  void setToZeroState() override {
    if (mps) mps->resetZero();
    // Clear history: a new circuit is being set up from |0>.
    if (m_recordingGates) m_gateHistory.clear();
  }

  // Apply gate and record to history for later trajectory replay.
  void applyGate(const GateApplicationTask &task) override {
    ensureEngine();
    applyGateInternal(task);
    if (m_recordingGates) {
      GateRecord record{
          task.operationName,
          std::vector<std::complex<double>>(task.matrix.begin(), task.matrix.end()),
          task.controls, task.targets,
          std::vector<double>(task.parameters.begin(), task.parameters.end())};
      m_gateHistory.push_back(record);
#ifdef CUDAQ_MPS_MQPU_BACKEND
      enqueueHbatchGate(record);
#endif
    }
  }

  // ---- Noise channel application ------------------------------------------
  // Called by the base class gate loop after each gate when a noise model is
  // active. Stochastically applies one Kraus operator from each matching
  // channel to the current MPS state.
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override {
    auto *ctx = cudaq::getExecutionContext();
    if (!ctx || !ctx->noiseModel || ctx->noiseModel->empty()) return;

    auto channels = ctx->noiseModel->get_channels(
        std::string(gateName), targets, controls, params);
    for (const auto &ch : channels)
      applyKrausChannel(targets, ch);
  }

  // ---- Advertise analytic observe (handled even with noise via trajectories)
  bool canHandleObserve() override { return true; }

  // ---- observe() -----------------------------------------------------------
  // 1. No noise, no shots -> analytic contractions (cost depends on terms/bonds).
  // 2. No noise, shots > 0 → shot noise: sample bitstrings from |psi|^2,
  //    estimate each Z/I Pauli term by parity average (mimics standard CUDA-Q
  //    shot-based observe). X/Y terms fall back to analytic.
  // 3. Noise model set     → multi-trajectory loop (quantum Monte Carlo).
  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    ensureEngine();
    auto *ctx = cudaq::getExecutionContext();
    const bool hasNoise = ctx && ctx->noiseModel && !ctx->noiseModel->empty();
    // The C++ async/parallel observe API uses -1 for analytic execution. Since
    // ExecutionContext stores shots as size_t, that sentinel arrives here as
    // SIZE_MAX and must not be interpreted as an enormous shot request.
    const bool finiteShots =
        ctx && ctx->shots > 0 &&
        ctx->shots != std::numeric_limits<std::size_t>::max();
    const bool useShots = !hasNoise && finiteShots;

    auto pd = buildPauliTermData(op);

    static const bool useHost = []{
      const char *e = std::getenv("CUDAQ_MPS_EXPECT_HOST");
      return e && atoi(e) != 0;
    }();

    auto computeExpVals = [&]() -> std::vector<std::complex<double>> {
#ifdef CUDAQ_MPS_MQPU_BACKEND
      return runPersistentHamiltonianBatch(
          pd.words, HBatchEvaluation::Analytic, 0, 0, useHost, hasNoise);
#else
      std::vector<std::complex<double>> values(pd.words.size(), 1.0);
      for (std::size_t term = 0; term < pd.words.size(); ++term)
        values[term] = useHost ? mps->expectPauliHost(pd.words[term])
                               : mps->expectPauli(pd.words[term]);
      return values;
#endif
    };

    // Helper: build and return an observe_result from per-term expectation values.
    auto makeResult = [&](const std::vector<std::complex<double>> &ev)
        -> cudaq::observe_result {
      std::complex<double> total = 0.0;
      std::vector<cudaq::ExecutionResult> results;
      for (size_t k = 0; k < pd.words.size(); ++k) {
        std::complex<double> contrib = pd.coeffs[k] * ev[k];
        total += contrib;
        // Per-term raw data stores <P_k>, not c_k<P_k>. CUDA-Q uses the term
        // ID to reapply c_k when an async result is serialized/reconstructed.
        results.emplace_back(
            cudaq::ExecutionResult({}, pd.termIds[k], ev[k].real()));
      }
      cudaq::sample_result data(total.real(), results);
      return cudaq::observe_result(total.real(), op, data);
    };

    // --- Path 1: exact analytic (no noise, no shots) ---
    if (!hasNoise && !useShots)
      return makeResult(computeExpVals());

    // --- Path 2: finite-shot observation (no noise model) ---
    // The shots backend treats a diagonal Hamiltonian as one commuting
    // measurement group, reusing one sample set for every Z/I word and
    // partitioning one fixed shot budget over its physical-GPU group. The
    // Hamiltonian backend instead distributes terms and applies `nshots` to
    // every non-identity term, matching conventional Hamiltonian batching.
    if (useShots) {
      const int nshots = ctx->shots;
      const unsigned evaluationSeed = sampleSeed++;
#ifdef CUDAQ_MPS_MQPU_BACKEND
#if defined(CUDAQ_MPS_MQPU_HAMILTONIAN_BACKEND)
      auto ev = runPersistentHamiltonianBatch(
          pd.words, HBatchEvaluation::ShotSampled, nshots, evaluationSeed,
          false, false);
#else
      const bool allDiagonal = std::all_of(
          pd.words.begin(), pd.words.end(), [](const std::string &word) {
            return std::all_of(word.begin(), word.end(),
                               [](char value) {
                                 return value == 'I' || value == 'Z';
                               });
          });
      auto ev = allDiagonal
                    ? runSharedDiagonalShotBatch(pd.words, nshots,
                                                 evaluationSeed)
                    : runPersistentHamiltonianBatch(
                          pd.words, HBatchEvaluation::ShotSampled, nshots,
                          evaluationSeed, false, false);
#endif
#else
      std::vector<std::complex<double>> ev(pd.words.size(), 1.0);
      for (std::size_t term = 0; term < pd.words.size(); ++term) {
        const auto &word = pd.words[term];
        const bool identity = std::all_of(
            word.begin(), word.end(), [](char value) { return value == 'I'; });
        if (identity)
          continue;
        const bool zOnly = std::all_of(
            word.begin(), word.end(),
            [](char value) { return value == 'I' || value == 'Z'; });
        if (!zOnly) {
          ev[term] = mps->expectPauli(word);
          continue;
        }
        unsigned termSeed = 2166136261u ^ evaluationSeed;
        for (unsigned char value : word) {
          termSeed ^= static_cast<unsigned>(value);
          termSeed *= 16777619u;
        }
        ev[term] = mps->sampleZExpectationsGpu(
            std::vector<std::string>{word}, nshots, termSeed).front();
      }
#endif
      return makeResult(ev);
    }

    // --- Path 3: noisy trajectory loop ---
    std::size_t numTraj = DEFAULT_OBSERVE_TRAJECTORIES;
    if (ctx->numberTrajectories.has_value())
      numTraj = ctx->numberTrajectories.value();
    else if (finiteShots)
      numTraj = ctx->shots;

    auto savedHistory = m_gateHistory;
    m_recordingGates = false;

    std::vector<std::complex<double>> termExpVals(pd.words.size(), 0.0);
    for (std::size_t traj = 0; traj < numTraj; ++traj) {
      mps->resetZero();
      replayCircuitWithNoise(savedHistory, ctx);
      auto ev = computeExpVals();
      for (size_t k = 0; k < pd.words.size(); ++k)
        termExpVals[k] += ev[k] / static_cast<double>(numTraj);
    }

    m_recordingGates = true;
    mps->resetZero();
    for (const auto &g : savedHistory) {
      GateApplicationTask task(g.operationName, g.matrix, g.controls,
                               g.targets, g.parameters);
      applyGateInternal(task);
    }

    return makeResult(termExpVals);
  }

  bool measureQubit(const std::size_t) override {
    throw std::runtime_error(CUDAQ_CUSTOM_MPS_BACKEND_NAME
                             ": mid-circuit measureQubit not "
                             "implemented; use observe() or sample().");
  }
  void resetQubit(const std::size_t) override {
    throw std::runtime_error(CUDAQ_CUSTOM_MPS_BACKEND_NAME
                             ": resetQubit not yet implemented.");
  }

  // ---- sample() ------------------------------------------------------------
  // Without noise: exact MPS perfect sampling from |psi|^2.
  // With noise: one trajectory per shot — each shot re-runs the circuit from
  // |0> with an independent Kraus realization before sampling one bitstring.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    ensureEngine();
    auto *ctx = cudaq::getExecutionContext();
    const bool hasNoise = ctx && ctx->noiseModel && !ctx->noiseModel->empty();
    int nshots = shots > 0 ? shots : 1000;

    if (!hasNoise) {
      // Split one fixed shot budget over the backend-managed physical GPU
      // group. Every helper samples an exact replica of the prepared state;
      // concatenating the shards is statistically identical to one GPU.
#ifdef CUDAQ_MPS_MQPU_BACKEND
      std::vector<std::string> bs =
          runSharedBitstringShotBatch(nshots, sampleSeed++);
#else
      std::vector<std::string> bs = mps->sampleBits(nshots, sampleSeed++);
#endif
      cudaq::CountsDictionary counts;
      std::string key; key.reserve(qubits.size());
      for (const auto &full : bs) {
        key.clear();
        for (auto q : qubits) key.push_back(full[(int)q]);
        counts[key]++;
      }
      return cudaq::ExecutionResult(counts);
    }

    // --- Noisy path: one trajectory per shot ---
    auto savedHistory = m_gateHistory;
    m_recordingGates = false;

    cudaq::CountsDictionary counts;
    std::string key; key.reserve(qubits.size());
    for (int sh = 0; sh < nshots; ++sh) {
      mps->resetZero();
      replayCircuitWithNoise(savedHistory, ctx);
      auto bs = mps->sampleBits(1, sampleSeed++);   // one bitstring per trajectory
      key.clear();
      for (auto q : qubits) key.push_back(bs[0][(int)q]);
      if (includeSequentialData)
        counts[key]++;
      else
        counts[key]++;
    }

    m_recordingGates = true;
    // Restore deterministic final state (noise-free).
    mps->resetZero();
    for (const auto &g : savedHistory) {
      GateApplicationTask task(g.operationName, g.matrix, g.controls,
                               g.targets, g.parameters);
      applyGateInternal(task);
    }

    return cudaq::ExecutionResult(counts);
  }

  // Lift the base class's 63-qubit limit. The base computes a "state dimension"
  // of 2^n and throws at n>=64 (overflow). An MPS never materializes 2^n, and
  // this value is only used for FLOP/memory accounting, so return a
  // non-overflowing placeholder for large n -- enabling proteins >63 qubits.
  std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits < 64 ? (1ULL << numQubits) : numQubits;
  }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }
  std::string name() const override { return CUDAQ_CUSTOM_MPS_BACKEND_NAME; }

  CircuitSimulator *clone() override {
    thread_local static auto sim = std::make_unique<SimulatorMPSDist>();
    return sim.get();
  }
};

} // namespace nvqir

#if defined(CUDAQ_MPS_MQPU_SHOTS_BACKEND)
/// Register fixed-budget shared-distribution shot sharding.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist,
                         tensornet_mps_mqpu_shots)
#elif defined(CUDAQ_MPS_MQPU_HAMILTONIAN_BACKEND)
/// Register finite-shot Hamiltonian-term batching.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist,
                         tensornet_mps_mqpu_hamiltonian)
#elif defined(CUDAQ_MPS_MQPU_STRONG_BACKEND)
/// Register persistent prepared-state workers with a device-side term pipeline.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist, tensornet_mps_mqpu_strong)
#elif defined(CUDAQ_MPS_MQPU_REPLAY_BACKEND)
/// Register persistent independent-ansatz workers under a separate target.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist, tensornet_mps_mqpu_replay)
#elif defined(CUDAQ_MPS_MQPU_BACKEND)
/// Register the block-first, group-batched custom MPS under the MQPU name.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist, tensornet_mps_mqpu)
#else
/// Register the original sharded custom MPS backend.
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist, tensornet_mps_mgpu)
#endif

#undef CUDAQ_CUSTOM_MPS_BACKEND_NAME
#ifdef CUDAQ_MPS_MQPU_BACKEND
#undef SimulatorMPSDist
#undef ShardedMPS
#endif
