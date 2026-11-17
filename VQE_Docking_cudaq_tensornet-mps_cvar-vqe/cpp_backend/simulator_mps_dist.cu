// Milestone 3: NVQIR backend "tensornet-mps-mgpu" wrapping the tested
// multi-GPU sharded-MPS engine (sharded_mps.cuh).
//
// IMPORTANT: this file compiles ONLY inside an in-tree CUDA-Q build (it needs
// nvqir/CircuitSimulator.h and cudaq/spin_op.h). It is NOT built by the local
// build.sh. Drop it into runtime/nvqir/cutensornet/ and add it to that dir's
// CMakeLists (see distributed_mps/cpp_backend/CMake_snippet.txt). Spots marked
// [VERIFY] use CUDA-Q APIs whose exact form may need a small fixup for your
// release; the underlying engine is already validated.
//
// Design: replaces cuTensorNet's lazy single-device state with our eager
// explicit chain sharded across GPUs.
//   applyGate  -> ShardedMPS::apply1q / applyTwoQubit (SWAP-routed)
//   observe    -> sum_k coeff_k * <P_k>  via ShardedMPS::expectPauli
//   bond dim   <- CUDAQ_MPS_MAX_BOND (same env var as stock tensornet-mps)

#include "sharded_mps.cuh"
#include "nvqir/CircuitSimulator.h"
#include "cudaq/spin_op.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>

// Round-robin GPU assignment for single-GPU MPS mode (CUDAQ_MPS_NUM_GPUS=1).
// Each new thread that calls ensureEngine() is assigned the next available GPU
// so that parallel SubspaceVQE blocks land on different physical GPUs.
static std::atomic<int> g_next_gpu_rr{0};

namespace nvqir {

class SimulatorMPSDist : public nvqir::CircuitSimulatorBase<double> {
  using Base = nvqir::CircuitSimulatorBase<double>;
  using GateApplicationTask = typename Base::GateApplicationTask;

  std::unique_ptr<ShardedMPS> mps;
  std::size_t allocated = 0;     // qubits allocated so far
  int maxBond = 64;
  double absCut = 1e-5, relCut = 1e-5;
  std::vector<int> gpuIds;

  // Build the engine once the qubit count is known.
  //
  // CUDAQ_MPS_NUM_GPUS (default: all visible GPUs):
  //   Set to 1 to use a single GPU per circuit, enabling SubspaceVQE to run
  //   multiple blocks in parallel on different GPUs via ThreadPoolExecutor.
  //   Each new thread gets the next GPU in round-robin order so blocks on
  //   GPU 0, 1, 2, ... run concurrently without NCCL overhead.
  //   Set to N>1 to shard the MPS chain across N GPUs (original behavior).
  void ensureEngine() {
    if (mps) return;
    if (allocated < 2)
      throw std::runtime_error("tensornet-mps-mgpu needs >= 2 qubits");

    int G_total = 1; cudaGetDeviceCount(&G_total);

    const char* ng_env = std::getenv("CUDAQ_MPS_NUM_GPUS");
    int num_mps_gpus = ng_env ? std::max(1, std::atoi(ng_env)) : G_total;
    num_mps_gpus = std::min(num_mps_gpus, G_total);

    gpuIds.clear();
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

    if (auto *b = std::getenv("CUDAQ_MPS_MAX_BOND")) maxBond = std::max(1, atoi(b));
    if (auto *a = std::getenv("CUDAQ_MPS_ABS_CUTOFF")) absCut = atof(a);
    if (auto *r = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF")) relCut = atof(r);
    mps = std::make_unique<ShardedMPS>((int)allocated, gpuIds, maxBond, absCut, relCut);
    mps->distributedBoundary = (gpuIds.size() > 1);
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

public:
  SimulatorMPSDist() = default;

  // Bring both base addQubitsToState overloads into scope (we only specialize
  // the (count, ptr) one) to avoid a partial-override error.
  using nvqir::CircuitSimulatorBase<double>::addQubitsToState;

  // ---- required CircuitSimulatorBase overrides ----
  void addQubitToState() override { addQubitsToState(1); }

  void addQubitsToState(std::size_t count, const void * = nullptr) override {
    if (mps) throw std::runtime_error(
        "tensornet-mps-mgpu: adding qubits after gates is not supported");
    allocated += count;
  }

  void deallocateStateImpl() override { mps.reset(); allocated = 0; }

  void setToZeroState() override { if (mps) mps->resetZero(); }

  void applyGate(const GateApplicationTask &task) override {
    ensureEngine();
    const auto &m = task.matrix;                  // row-major
    std::vector<std::size_t> ctr, tgt;
    for (auto i : task.controls) ctr.push_back(conv(i));   // [VERIFY] endianness
    for (auto i : task.targets)  tgt.push_back(conv(i));

    if (ctr.empty() && tgt.size() == 1) {
      std::vector<std::complex<double>> U{m[0], m[1], m[2], m[3]};
      mps->apply1q(U, (int)tgt[0]); return;
    }
    if (ctr.empty() && tgt.size() == 2) {
      mps->applyTwoQubit(m, (int)tgt[0], (int)tgt[1]); return;
    }
    if (ctr.size() == 1 && tgt.size() == 1) {
      std::vector<std::complex<double>> U{m[0], m[1], m[2], m[3]};
      mps->applyTwoQubit(controlledGate(U), (int)ctr[0], (int)tgt[0]); return;
    }
    throw std::runtime_error(
        "tensornet-mps-mgpu: >2-qubit or multi-control gates not yet supported "
        "(decompose, or extend applyGate).");
  }

  // Advertise analytic observe so CUDA-Q calls observe() directly instead of
  // falling back to shot-based sampling (which would hit our sample() stub).
  bool canHandleObserve() override { return true; }

  // VQE energy path: <H> = sum_k coeff_k <P_k>. All terms are evaluated together
  // through the multi-GPU segment pipeline (expectPauliBatch) so the GPUs work
  // on different terms concurrently.
  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    ensureEngine();
    const int nq = mps->n;
    std::vector<std::string> words, termIds;
    std::vector<std::complex<double>> coeffs;
    for (const auto &term : op) {
      std::string word = term.get_pauli_word(nq);  // length-nq "IXYZ", CUDA-Q qubit order
      // Gates route CUDA-Q qubit q to MPS site conv(q)=n-1-q, so the Pauli word
      // (indexed by CUDA-Q qubit) must be reversed to index MPS sites.
      std::reverse(word.begin(), word.end());
      words.push_back(std::move(word));
      coeffs.push_back(term.evaluate_coefficient());
      termIds.push_back(term.get_term_id());
    }
    // A/B knob for confirming GPU acceleration: CUDAQ_MPS_EXPECT_HOST=1 routes
    // the expectation through the reference host (CPU, O(chi^4)) path instead of
    // the GPU pipeline. Default = GPU pipeline.
    static const bool useHost = []{ const char* e = std::getenv("CUDAQ_MPS_EXPECT_HOST");
                                    return e && atoi(e) != 0; }();
    std::vector<std::complex<double>> ev;
    if (useHost) { ev.resize(words.size());
      for (size_t k = 0; k < words.size(); ++k) ev[k] = mps->expectPauliHost(words[k]); }
    else ev = mps->expectPauliBatch(words);
    std::complex<double> total = 0.0;
    std::vector<cudaq::ExecutionResult> results;
    for (size_t k = 0; k < words.size(); ++k) {
      std::complex<double> contrib = coeffs[k] * ev[k];
      total += contrib;
      results.emplace_back(cudaq::ExecutionResult({}, termIds[k], contrib.real()));
    }
    cudaq::sample_result data(total.real(), results);
    return cudaq::observe_result(total.real(), op, data);
  }

  unsigned sampleSeed = 1234567u;

  bool measureQubit(const std::size_t) override {
    throw std::runtime_error("tensornet-mps-mgpu: mid-circuit measureQubit not "
                             "implemented; use observe() or sample().");
  }
  void resetQubit(const std::size_t) override {
    throw std::runtime_error("tensornet-mps-mgpu: resetQubit not yet implemented.");
  }
  // Exact MPS sampling: draw `shots` bitstrings from |psi|^2 and tally counts.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    ensureEngine();
    int nshots = shots > 0 ? shots : 1000;
    std::vector<std::string> bs = mps->sampleBits(nshots, sampleSeed++);  // qubit-0-leftmost
    cudaq::CountsDictionary counts;
    std::string key; key.reserve(qubits.size());
    for (const auto &full : bs) {
      key.clear();
      for (auto q : qubits) key.push_back(full[(int)q]);  // select/order requested qubits
      counts[key]++;
    }
    cudaq::ExecutionResult result(counts);
    return result;
  }

  // Lift the base class's 63-qubit limit. The base computes a "state dimension"
  // of 2^n and throws at n>=64 (overflow). An MPS never materializes 2^n, and
  // this value is only used for FLOP/memory accounting, so return a
  // non-overflowing placeholder for large n -- enabling proteins >63 qubits.
  std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits < 64 ? (1ULL << numQubits) : numQubits;
  }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }
  std::string name() const override { return "tensornet-mps-mgpu"; }

  CircuitSimulator *clone() override {
    thread_local static auto sim = std::make_unique<SimulatorMPSDist>();
    return sim.get();
  }
};

} // namespace nvqir

/// Register under target name "tensornet-mps-mgpu".
NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPSDist, tensornet_mps_mgpu)
