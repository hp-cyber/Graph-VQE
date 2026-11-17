#!/usr/bin/env bash
# =============================================================================
# One-shot build of CUDA-Q + the multi-GPU memory-pooled MPS backend
# (tensornet-mps-mgpu), installed ENTIRELY under THIS directory.
#
# Start state: this directory contains only ./VQE_Docking and this script.
# Usage (inside a CUDA-Q devdeps container, or a server with a consistent
# clang/gcc<=13 toolchain; conda env provides python + cuQuantum + cmake/ninja):
#
#     conda activate cudaq-env
#     ./setup_local_cudaq.sh
#
# Everything (LLVM, prerequisites, CUDA-Q, our backend) installs under
# ./cudaq_local -- nothing touches /usr/local or your home dir.
#
# After it finishes:
#     ./run.sh                       # runs VQE_Docking/Main.py on tensornet-mps-mgpu
#
# Knobs (env): CUDA_VISIBLE_DEVICES (default 2,3,4,5), CUDAQ_MPS_MAX_BOND,
#              CUDAQ_VERSION_TAG / CUDAQ_COMMIT (CUDA-Q source revision).
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---- config -----------------------------------------------------------------
CUDAQ_COMMIT="${CUDAQ_COMMIT:-91ab3092e76dab8887d1fdf0c99a2478ca90581c}"   # == 0.14.2
CUDAQ_VERSION_TAG="${CUDAQ_VERSION_TAG:-0.14.2}"
CUDAQ_REPO="${CUDAQ_REPO:-https://github.com/NVIDIA/cuda-quantum.git}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

LOCAL="$PROJECT_DIR/cudaq_local"          # all installs live here
SRC="$PROJECT_DIR/cuda-quantum-src"       # CUDA-Q source tree
BACKEND="$SRC/runtime/nvqir/cutensornet"  # where our backend files go
mkdir -p "$LOCAL"

# ---- environment detection: conda (bare metal) vs devdeps container ---------
# Bare-metal mode: conda cudaq-env provides python/cuQuantum, and we build LLVM.
# Container mode  : the devdeps image already has LLVM + cuQuantum + toolchain,
#                   so we skip install_prerequisites and reuse them.
if [ -z "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate cudaq-env 2>/dev/null || true
fi
command -v git >/dev/null || { echo "ERROR: git required" >&2; exit 1; }
command -v python  >/dev/null 2>&1 || alias python=python3
PY="$(command -v python || command -v python3)"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo /usr)"

# Is LLVM already provided (devdeps container)?  Then we're in PREBUILT mode.
PREBUILT=0
for d in "${LLVM_INSTALL_PREFIX:-}" /opt/llvm /usr/lib/llvm; do
  [ -n "$d" ] && [ -x "$d/bin/llvm-config" ] && { PREBUILT=1; LLVM_PREBUILT="$d"; break; }
done

echo ">> install root:    $LOCAL"
echo ">> CUDA-Q source:   $SRC  @ ${CUDAQ_COMMIT}"
echo ">> python:          $PY   (site: $SITE)"
echo ">> mode:            $([ "$PREBUILT" = 1 ] && echo "PREBUILT/container (LLVM @ $LLVM_PREBUILT)" || echo "bare-metal (will build LLVM)")"

# ---- 1. fetch CUDA-Q source (shallow, pinned) -------------------------------
if [ ! -d "$SRC/.git" ]; then
  echo ">> cloning CUDA-Q ..."
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$CUDAQ_REPO"
  if ! git -C "$SRC" fetch --depth 1 origin "$CUDAQ_COMMIT" 2>/dev/null; then
    git -C "$SRC" fetch --depth 1 origin "$CUDAQ_VERSION_TAG"
  fi
  git -C "$SRC" checkout -q FETCH_HEAD
else
  echo ">> CUDA-Q source already present."
fi

# ---- 2. write the multi-GPU MPS backend into the cutensornet dir ------------
echo ">> writing tensornet-mps-mgpu backend sources ..."
cat > "$BACKEND/dist_svd.cuh" <<'MGPU_DIST_SVD_CUH_EOF'
// Reusable distributed randomized truncated SVD (cuBLAS/cuSOLVER/NCCL).
// Header form so both the SVD self-test and the sharded-MPS engine can use it.
// Works for G>=1 GPUs: with G==1 it is a single-GPU randomized truncated SVD
// (NCCL allreduce is skipped).  All GPU matrices are COLUMN-MAJOR.
#pragma once

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <algorithm>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s){fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)
#define CUSOLVER_CHECK(x) do { cusolverStatus_t s=(x); if(s){fprintf(stderr,"cuSOLVER %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)
#define NCCL_CHECK(x) do { ncclResult_t r=(x); if(r!=ncclSuccess){fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r));exit(1);} }while(0)
#endif

using cdouble = cuDoubleComplex;
static const cdouble C_ONE  = make_cuDoubleComplex(1.0, 0.0);
static const cdouble C_ZERO = make_cuDoubleComplex(0.0, 0.0);

struct GpuCtx {
  int dev;
  cudaStream_t stream;
  cublasHandle_t blas;
  cusolverDnHandle_t solver;
  ncclComm_t comm;   // unused when the call uses a single GPU
};

static inline void allreduceSum(std::vector<GpuCtx>& ctx, std::vector<cdouble*>& bufs,
                                size_t count) {
  if (ctx.size() == 1) return;   // single GPU: buffer already holds the full sum
  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < ctx.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    NCCL_CHECK(ncclAllReduce(bufs[i], bufs[i], count * 2, ncclDouble, ncclSum,
                             ctx[i].comm, ctx[i].stream));
  }
  NCCL_CHECK(ncclGroupEnd());
  for (auto& c : ctx) { CUDA_CHECK(cudaSetDevice(c.dev)); CUDA_CHECK(cudaStreamSynchronize(c.stream)); }
}

static inline void qrColumns(GpuCtx& g, int m, int l, const cdouble* A, cdouble* Qout) {
  CUDA_CHECK(cudaSetDevice(g.dev));
  CUDA_CHECK(cudaMemcpyAsync(Qout, A, sizeof(cdouble)*(size_t)m*l, cudaMemcpyDeviceToDevice, g.stream));
  cdouble* tau; CUDA_CHECK(cudaMalloc(&tau, sizeof(cdouble)*l));
  int lwork = 0; CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(g.solver, m, l, Qout, m, &lwork));
  int lwork2 = 0; CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(g.solver, m, l, l, Qout, m, tau, &lwork2));
  if (lwork2 > lwork) lwork = lwork2;
  cdouble* work; CUDA_CHECK(cudaMalloc(&work, sizeof(cdouble)*lwork));
  int* info;     CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnZgeqrf(g.solver, m, l, Qout, m, tau, work, lwork, info));
  CUSOLVER_CHECK(cusolverDnZungqr(g.solver, m, l, l, Qout, m, tau, work, lwork, info));
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  CUDA_CHECK(cudaFree(tau)); CUDA_CHECK(cudaFree(work)); CUDA_CHECK(cudaFree(info));
}

static inline void heig(GpuCtx& g, int l, cdouble* C, double* w_host) {
  CUDA_CHECK(cudaSetDevice(g.dev));
  double* w; CUDA_CHECK(cudaMalloc(&w, sizeof(double)*l));
  int lwork=0;
  CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(g.solver, CUSOLVER_EIG_MODE_VECTOR,
                  CUBLAS_FILL_MODE_LOWER, l, C, l, w, &lwork));
  cdouble* work; CUDA_CHECK(cudaMalloc(&work, sizeof(cdouble)*lwork));
  int* info; CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnZheevd(g.solver, CUSOLVER_EIG_MODE_VECTOR,
                  CUBLAS_FILL_MODE_LOWER, l, C, l, w, work, lwork, info));
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  CUDA_CHECK(cudaMemcpy(w_host, w, sizeof(double)*l, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(w)); CUDA_CHECK(cudaFree(work)); CUDA_CHECK(cudaFree(info));
}

static inline cdouble* peerCopy(int srcDev, const cdouble* src, int dstDev, size_t n) {
  cdouble* dst; CUDA_CHECK(cudaSetDevice(dstDev)); CUDA_CHECK(cudaMalloc(&dst, sizeof(cdouble)*n));
  CUDA_CHECK(cudaMemcpyPeer(dst, dstDev, src, srcDev, sizeof(cdouble)*n));
  return dst;
}

__global__ static void scaleRowsKernel(cdouble* B, int r, int ncols, const double* inv) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= r*ncols) return;
  int row = idx % r;
  B[idx].x *= inv[row]; B[idx].y *= inv[row];
}

struct SvdResult {
  std::vector<cdouble> U;   // m x k  (host, column-major)
  std::vector<double>  S;   // k
  std::vector<cdouble> Vh;  // k x n  (host, column-major, gathered)
  int k;
};

// Distributed (or single-GPU) randomized truncated SVD of a column-sharded M.
static inline SvdResult distSvd(std::vector<GpuCtx>& ctx, std::vector<cdouble*>& blocks,
                  const std::vector<int>& ng, int m, int n,
                  int maxBond, double absCut, double relCut, int oversample,
                  unsigned seed) {
  int G = (int)ctx.size();
  int root = 0;
  int l = std::min(std::min(maxBond + oversample, m), n);

  std::vector<cdouble*> Y(G);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int i = 0; i < G; ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    std::vector<cdouble> omh((size_t)ng[i]*l);
    for (auto& z : omh) z = make_cuDoubleComplex(nd(rng), nd(rng));
    cdouble* om; CUDA_CHECK(cudaMalloc(&om, sizeof(cdouble)*omh.size()));
    CUDA_CHECK(cudaMemcpy(om, omh.data(), sizeof(cdouble)*omh.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&Y[i], sizeof(cdouble)*(size_t)m*l));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_N, CUBLAS_OP_N, m, l, ng[i],
                  &C_ONE, blocks[i], m, om, ng[i], &C_ZERO, Y[i], m));
    CUDA_CHECK(cudaStreamSynchronize(ctx[i].stream));
    CUDA_CHECK(cudaFree(om));
  }
  allreduceSum(ctx, Y, (size_t)m*l);

  std::vector<cdouble*> Q(G);
  CUDA_CHECK(cudaSetDevice(ctx[root].dev));
  CUDA_CHECK(cudaMalloc(&Q[root], sizeof(cdouble)*(size_t)m*l));
  qrColumns(ctx[root], m, l, Y[root], Q[root]);
  for (int i = 0; i < G; ++i) if (i != root)
    Q[i] = peerCopy(ctx[root].dev, Q[root], ctx[i].dev, (size_t)m*l);

  std::vector<cdouble*> B(G), Cpart(G);
  for (int i = 0; i < G; ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    CUDA_CHECK(cudaMalloc(&B[i], sizeof(cdouble)*(size_t)l*ng[i]));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_C, CUBLAS_OP_N, l, ng[i], m,
                  &C_ONE, Q[i], m, blocks[i], m, &C_ZERO, B[i], l));
    CUDA_CHECK(cudaMalloc(&Cpart[i], sizeof(cdouble)*(size_t)l*l));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_N, CUBLAS_OP_C, l, l, ng[i],
                  &C_ONE, B[i], l, B[i], l, &C_ZERO, Cpart[i], l));
    CUDA_CHECK(cudaStreamSynchronize(ctx[i].stream));
  }
  allreduceSum(ctx, Cpart, (size_t)l*l);

  std::vector<double> w(l);
  heig(ctx[root], l, Cpart[root], w.data());
  std::vector<double> S(l);
  for (int i = 0; i < l; ++i) S[i] = std::sqrt(std::max(0.0, w[l-1-i]));
  double smax = S.empty()?0.0:S[0];
  double thresh = std::max(absCut, relCut*smax);
  int k = 0; for (int i=0;i<l;++i) if (S[i] > thresh) ++k;
  if (k < 1) k = 1; if (k > maxBond) k = maxBond; if (k > l) k = l;

  std::vector<cdouble> Cvec((size_t)l*l);
  CUDA_CHECK(cudaSetDevice(ctx[root].dev));
  CUDA_CHECK(cudaMemcpy(Cvec.data(), Cpart[root], sizeof(cdouble)*(size_t)l*l, cudaMemcpyDeviceToHost));
  std::vector<cdouble> Uhat((size_t)l*k);
  for (int c = 0; c < k; ++c) { int src = l-1-c;
    for (int rrow = 0; rrow < l; ++rrow) Uhat[(size_t)c*l + rrow] = Cvec[(size_t)src*l + rrow]; }
  cdouble* dUhat; CUDA_CHECK(cudaMalloc(&dUhat, sizeof(cdouble)*(size_t)l*k));
  CUDA_CHECK(cudaMemcpy(dUhat, Uhat.data(), sizeof(cdouble)*(size_t)l*k, cudaMemcpyHostToDevice));

  SvdResult res; res.k = k; res.S.assign(S.begin(), S.begin()+k);
  cdouble* dU; CUDA_CHECK(cudaMalloc(&dU, sizeof(cdouble)*(size_t)m*k));
  CUBLAS_CHECK(cublasZgemm(ctx[root].blas, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l,
                &C_ONE, Q[root], m, dUhat, l, &C_ZERO, dU, m));
  res.U.resize((size_t)m*k);
  CUDA_CHECK(cudaMemcpy(res.U.data(), dU, sizeof(cdouble)*(size_t)m*k, cudaMemcpyDeviceToHost));

  std::vector<double> invS(k); for (int i=0;i<k;++i) invS[i] = S[i]>0? 1.0/S[i] : 0.0;
  res.Vh.assign((size_t)k*n, C_ZERO);
  int colOffset = 0;
  for (int i = 0; i < G; ++i) {
    CUDA_CHECK(cudaSetDevice(ctx[i].dev));
    cdouble* UhatI = (i==root)? dUhat : peerCopy(ctx[root].dev, dUhat, ctx[i].dev, (size_t)l*k);
    double* dInv; CUDA_CHECK(cudaMalloc(&dInv, sizeof(double)*k));
    CUDA_CHECK(cudaMemcpy(dInv, invS.data(), sizeof(double)*k, cudaMemcpyHostToDevice));
    cdouble* Vhg; CUDA_CHECK(cudaMalloc(&Vhg, sizeof(cdouble)*(size_t)k*ng[i]));
    CUBLAS_CHECK(cublasZgemm(ctx[i].blas, CUBLAS_OP_C, CUBLAS_OP_N, k, ng[i], l,
                  &C_ONE, UhatI, l, B[i], l, &C_ZERO, Vhg, k));
    int threads=256, blocks_=((k*ng[i])+threads-1)/threads;
    scaleRowsKernel<<<blocks_,threads,0,ctx[i].stream>>>(Vhg, k, ng[i], dInv);
    CUDA_CHECK(cudaStreamSynchronize(ctx[i].stream));
    std::vector<cdouble> Vhh((size_t)k*ng[i]);
    CUDA_CHECK(cudaMemcpy(Vhh.data(), Vhg, sizeof(cdouble)*(size_t)k*ng[i], cudaMemcpyDeviceToHost));
    for (int c=0;c<ng[i];++c) for (int r=0;r<k;++r)
      res.Vh[(size_t)(colOffset+c)*k + r] = Vhh[(size_t)c*k + r];
    colOffset += ng[i];
    CUDA_CHECK(cudaFree(dInv)); CUDA_CHECK(cudaFree(Vhg));
    if (i!=root) CUDA_CHECK(cudaFree(UhatI));
  }
  for (int i=0;i<G;++i){ CUDA_CHECK(cudaSetDevice(ctx[i].dev)); CUDA_CHECK(cudaFree(Y[i])); CUDA_CHECK(cudaFree(Q[i])); CUDA_CHECK(cudaFree(B[i])); CUDA_CHECK(cudaFree(Cpart[i])); }
  CUDA_CHECK(cudaSetDevice(ctx[root].dev)); CUDA_CHECK(cudaFree(dUhat)); CUDA_CHECK(cudaFree(dU));
  return res;
}
MGPU_DIST_SVD_CUH_EOF

cat > "$BACKEND/sharded_mps.cuh" <<'MGPU_SHARDED_MPS_CUH_EOF'
// Sharded-MPS engine (header): eager explicit chain across GPUs.
// Used by the self-test (sharded_mps.cu) and the NVQIR wrapper (milestone 3).
//  - chain sharded by contiguous segments across GPUs (memory pooling)
//  - 1q gates local; adjacent 2q gates via custom contraction + truncated SVD
//  - arbitrary-pair 2q gates via SWAP routing
//  - <psi|Pauli-string|psi> expectation (for VQE energy)
#pragma once
#include "dist_svd.cuh"
#include <string>
#include <algorithm>
#include <cstdlib>

__global__ static void k_apply1q(const cdouble* A, const cdouble* U, cdouble* out,
                                 int Dl, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int total = Dl*2*Dr; if (idx>=total) return;
  int c = idx % Dr; int t = idx/Dr; int x = t%2; int l = t/2;
  cdouble acc = make_cuDoubleComplex(0.0,0.0);
  for (int i=0;i<2;++i) acc = cuCadd(acc, cuCmul(U[x*2+i], A[(l*2+i)*Dr + c]));
  out[(l*2+x)*Dr + c] = acc;
}

__global__ static void k_buildM(cdouble* M, const cdouble* A, const cdouble* B,
                               const cdouble* G, int Dl, int Dc, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int total = Dl*2*2*Dr; if (idx>=total) return;
  int r = idx % Dr; int t = idx/Dr; int J = t%2; t/=2; int I = t%2; int l = t/2;
  cdouble acc = make_cuDoubleComplex(0.0,0.0);
  for (int i=0;i<2;++i) for (int j=0;j<2;++j) {
    cdouble g = G[(I*2+J)*4 + (i*2+j)];
    if (g.x==0.0 && g.y==0.0) continue;
    cdouble s = make_cuDoubleComplex(0.0,0.0);
    for (int c=0;c<Dc;++c) s = cuCadd(s, cuCmul(A[(l*2+i)*Dc + c], B[(c*2+j)*Dr + r]));
    acc = cuCadd(acc, cuCmul(g, s));
  }
  int row = l*2+I, col = J*Dr + r, m = 2*Dl;
  M[row + (size_t)col*m] = acc;
}

// ---- GPU-side Pauli expectation (stage 1) ----------------------------------
// Left-to-right transfer: E_{s+1} = A_s^dagger (P_s) E_s A_s, kept on-device.
// Two O(chi^3) sweeps per site (vs the old host O(chi^4) quadruple loop), all
// on the GPU that owns the site; the environment E is peer-copied across GPU
// boundaries. Indexing mirrors the validated host version exactly.
//   k_transL:  R[a,ip,bp] = sum_ap E[a,ap] A[ap,ip,bp]          (contract left bond)
//   k_applyP:  S[a,i,bp]  = sum_ip P[i,ip] R[a,ip,bp]           (apply Pauli)
//   k_transR:  newE[b,bp] = sum_{a,i} conj(A[a,i,b]) S[a,i,bp]  (contract bra)
__global__ static void k_transL(const cdouble* E, const cdouble* A, cdouble* R,
                                int Dl, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= Dl*2*Dr) return;
  int bp = idx % Dr; int t = idx / Dr; int ip = t % 2; int a = t / 2;
  cdouble acc = make_cuDoubleComplex(0,0);
  for (int ap = 0; ap < Dl; ++ap)
    acc = cuCadd(acc, cuCmul(E[a*Dl + ap], A[(ap*2+ip)*Dr + bp]));
  R[(a*2+ip)*Dr + bp] = acc;
}
__global__ static void k_applyP(const cdouble* R, cdouble* S, int Dl, int Dr,
                                cdouble p00, cdouble p01, cdouble p10, cdouble p11) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= Dl*2*Dr) return;
  int bp = idx % Dr; int t = idx / Dr; int i = t % 2; int a = t / 2;
  cdouble r0 = R[(a*2+0)*Dr + bp], r1 = R[(a*2+1)*Dr + bp];
  cdouble pi0 = (i==0)?p00:p10, pi1 = (i==0)?p01:p11;
  S[(a*2+i)*Dr + bp] = cuCadd(cuCmul(pi0, r0), cuCmul(pi1, r1));
}
__global__ static void k_transR(const cdouble* A, const cdouble* S, cdouble* newE,
                                int Dl, int Dr) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= Dr*Dr) return;
  int bp = idx % Dr; int b = idx / Dr;
  cdouble acc = make_cuDoubleComplex(0,0);
  int rows = Dl*2;
  for (int t = 0; t < rows; ++t)
    acc = cuCadd(acc, cuCmul(cuConj(A[t*Dr + b]), S[t*Dr + bp]));
  newE[b*Dr + bp] = acc;
}
// 2x2 Pauli matrix elements P[i][ip] (bra index i, ket index ip)
static inline void pauliMat(char p, cdouble& p00, cdouble& p01,
                            cdouble& p10, cdouble& p11) {
  p00 = p01 = p10 = p11 = make_cuDoubleComplex(0,0);
  switch (p) {
    case 'X': p01 = make_cuDoubleComplex(1,0);  p10 = make_cuDoubleComplex(1,0);  break;
    case 'Y': p01 = make_cuDoubleComplex(0,-1); p10 = make_cuDoubleComplex(0,1);  break;
    case 'Z': p00 = make_cuDoubleComplex(1,0);  p11 = make_cuDoubleComplex(-1,0); break;
    default:  p00 = make_cuDoubleComplex(1,0);  p11 = make_cuDoubleComplex(1,0);  break; // I
  }
}

struct Site { cdouble* d; int Dl, Dr, dev; };   // physical dim = 2

// Per-thread persistent GPU context pool: NCCL communicators + cuBLAS / cuSOLVER
// handles + streams, created ONCE per thread for the (fixed) GPU set and reused
// by every ShardedMPS on that thread. Previously these were created per circuit
// execution and never destroyed, leaking NCCL communicators (and their proxy
// threads) until pthread_create failed with EAGAIN ("Resource temporarily
// unavailable") partway through a long VQE run. thread_local keeps it correct
// for the multi-threaded SubspaceVQE path (NCCL comms aren't safe to share
// across threads), while bounding total comms to one set per worker thread.
static std::vector<GpuCtx>& shardedGpuPool(const std::vector<int>& gpu_ids) {
  static thread_local std::vector<GpuCtx> pool;
  if (pool.empty()) {
    int G = (int)gpu_ids.size();
    pool.resize(G);
    std::vector<int> devs = gpu_ids;
    ncclComm_t* comms = new ncclComm_t[G];
    NCCL_CHECK(ncclCommInitAll(comms, G, devs.data()));
    for (int i = 0; i < G; ++i) {
      pool[i].dev = gpu_ids[i]; CUDA_CHECK(cudaSetDevice(pool[i].dev));
      CUDA_CHECK(cudaStreamCreate(&pool[i].stream));
      CUBLAS_CHECK(cublasCreate(&pool[i].blas)); CUBLAS_CHECK(cublasSetStream(pool[i].blas, pool[i].stream));
      CUSOLVER_CHECK(cusolverDnCreate(&pool[i].solver)); CUSOLVER_CHECK(cusolverDnSetStream(pool[i].solver, pool[i].stream));
      pool[i].comm = comms[i];
    }
    delete[] comms;
  }
  return pool;
}

struct ShardedMPS {
  int n;
  std::vector<int> gpu_ids;
  std::vector<Site> site;
  std::vector<int> owner;
  std::vector<GpuCtx> ctx;
  int maxBond; double absCut, relCut; unsigned seedCounter = 1;
  bool distributedBoundary = false;   // true -> cross-GPU bonds use G>1 distSvd

  ShardedMPS(int nqubits, std::vector<int> gids, int maxB, double aC, double rC)
      : n(nqubits), gpu_ids(gids), maxBond(maxB), absCut(aC), relCut(rC) {
    int G = gpu_ids.size();
    // Share the persistent comms/handles (copies handle values; not owned here).
    ctx = shardedGpuPool(gpu_ids);
    owner.resize(n);
    for (int s=0;s<n;++s) owner[s] = gpu_ids[(long long)s*G/n];
    site.resize(n);
    for (int s=0;s<n;++s){ int dev=owner[s]; CUDA_CHECK(cudaSetDevice(dev));
      cdouble h[2]={make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0)};
      cdouble* d; CUDA_CHECK(cudaMalloc(&d, sizeof(cdouble)*2));
      CUDA_CHECK(cudaMemcpy(d,h,sizeof(cdouble)*2,cudaMemcpyHostToDevice));
      site[s]={d,1,1,dev}; }
  }

  // Free only this instance's site tensors. The comms/cuBLAS/cuSOLVER handles in
  // ctx belong to the persistent pool (shardedGpuPool) and are reused, so they
  // must NOT be destroyed here. No CUDA_CHECK (never throw from a destructor).
  ~ShardedMPS(){
    for (auto& st : site) if (st.d){ cudaSetDevice(st.dev); cudaFree(st.d); st.d=nullptr; }
  }

  GpuCtx& ctxOf(int dev){ for(auto&c:ctx) if(c.dev==dev) return c; return ctx[0]; }

  void resetZero(){
    for (int s=0;s<n;++s){ CUDA_CHECK(cudaSetDevice(site[s].dev)); CUDA_CHECK(cudaFree(site[s].d));
      cdouble h[2]={make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0)};
      cdouble* d; CUDA_CHECK(cudaMalloc(&d,sizeof(cdouble)*2));
      CUDA_CHECK(cudaMemcpy(d,h,sizeof(cdouble)*2,cudaMemcpyHostToDevice));
      site[s]={d,1,1,owner[s]}; }
  }

  void apply1q(const std::vector<std::complex<double>>& U, int s){
    int dev=site[s].dev; CUDA_CHECK(cudaSetDevice(dev));
    cdouble* Ud; CUDA_CHECK(cudaMalloc(&Ud,sizeof(cdouble)*4));
    CUDA_CHECK(cudaMemcpy(Ud,U.data(),sizeof(cdouble)*4,cudaMemcpyHostToDevice));
    int Dl=site[s].Dl, Dr=site[s].Dr; size_t sz=(size_t)Dl*2*Dr;
    cdouble* out; CUDA_CHECK(cudaMalloc(&out,sizeof(cdouble)*sz));
    int th=256, bl=(Dl*2*Dr+th-1)/th;
    k_apply1q<<<bl,th>>>(site[s].d, Ud, out, Dl, Dr); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(site[s].d)); CUDA_CHECK(cudaFree(Ud)); site[s].d=out;
  }

  // adjacent 2q gate on (s,s+1); G4 row-major 4x4 [(I*2+J)*4+(i*2+j)]
  void apply2q(const std::vector<std::complex<double>>& G4, int s){
    int ldev=site[s].dev, rdev=site[s+1].dev, work=ldev;
    int Dl=site[s].Dl, Dc=site[s].Dr, Dr=site[s+1].Dr;
    CUDA_CHECK(cudaSetDevice(work));
    cdouble* A = site[s].d;
    cdouble* B = (rdev==work)? site[s+1].d : peerCopy(rdev, site[s+1].d, work, (size_t)Dc*2*Dr);
    cdouble* Gd; CUDA_CHECK(cudaMalloc(&Gd,sizeof(cdouble)*16));
    CUDA_CHECK(cudaMemcpy(Gd,G4.data(),sizeof(cdouble)*16,cudaMemcpyHostToDevice));
    int m=2*Dl, ncol=2*Dr;
    cdouble* M; CUDA_CHECK(cudaMalloc(&M,sizeof(cdouble)*(size_t)m*ncol));
    int th=256, bl=(Dl*2*2*Dr+th-1)/th;
    k_buildM<<<bl,th>>>(M,A,B,Gd,Dl,Dc,Dr); CUDA_CHECK(cudaDeviceSynchronize());

    // Single-GPU truncated SVD. (Stage 3 -- a distributed-SVD path scattering M
    // across all GPUs -- was implemented and validated to 1.4e-15, but on PCIe
    // (no NVLink) at chi<=512 the per-gate NCCL communication outweighed the
    // O(chi^3/G) compute saving, so it is DISABLED. The multi-GPU distSvd routine
    // remains in dist_svd.cuh if revisited on NVLink / very large chi.)
    std::vector<GpuCtx> one{ctxOf(work)};
    std::vector<cdouble*> blk{M}; std::vector<int> ng{ncol};
    SvdResult r = distSvd(one, blk, ng, m, ncol, maxBond, absCut, relCut, 10, seedCounter++);
    int k=r.k;

    std::vector<cdouble> hA((size_t)Dl*2*k);
    for (int l=0;l<Dl;++l) for(int I=0;I<2;++I) for(int c=0;c<k;++c)
      hA[(size_t)(l*2+I)*k + c] = r.U[(size_t)c*m + (l*2+I)];
    std::vector<cdouble> hB((size_t)k*2*Dr);
    for (int c=0;c<k;++c) for(int J=0;J<2;++J) for(int rr=0;rr<Dr;++rr){
      cdouble v = r.Vh[(size_t)(J*Dr+rr)*k + c];
      hB[(size_t)(c*2+J)*Dr + rr] = make_cuDoubleComplex(r.S[c]*v.x, r.S[c]*v.y);
    }
    CUDA_CHECK(cudaFree(site[s].d));
    CUDA_CHECK(cudaSetDevice(ldev)); cdouble* dA; CUDA_CHECK(cudaMalloc(&dA,sizeof(cdouble)*hA.size()));
    CUDA_CHECK(cudaMemcpy(dA,hA.data(),sizeof(cdouble)*hA.size(),cudaMemcpyHostToDevice));
    site[s].d=dA; site[s].Dl=Dl; site[s].Dr=k;
    CUDA_CHECK(cudaFree(site[s+1].d));
    CUDA_CHECK(cudaSetDevice(rdev)); cdouble* dB; CUDA_CHECK(cudaMalloc(&dB,sizeof(cdouble)*hB.size()));
    CUDA_CHECK(cudaMemcpy(dB,hB.data(),sizeof(cdouble)*hB.size(),cudaMemcpyHostToDevice));
    site[s+1].d=dB; site[s+1].Dl=k; site[s+1].Dr=Dr;
    CUDA_CHECK(cudaSetDevice(work)); CUDA_CHECK(cudaFree(M)); CUDA_CHECK(cudaFree(Gd));
    if (rdev!=work) CUDA_CHECK(cudaFree(B));
  }

  static std::vector<std::complex<double>> swapMat(){
    std::vector<std::complex<double>> S(16,0.0);
    for(int I=0;I<2;++I)for(int J=0;J<2;++J)for(int i=0;i<2;++i)for(int j=0;j<2;++j)
      S[(I*2+J)*4+(i*2+j)] = (I==j && J==i)?1.0:0.0;
    return S;
  }
  static std::vector<std::complex<double>> swapOperands(const std::vector<std::complex<double>>& G){
    std::vector<std::complex<double>> H(16);
    for(int I=0;I<2;++I)for(int J=0;J<2;++J)for(int i=0;i<2;++i)for(int j=0;j<2;++j)
      H[(I*2+J)*4+(i*2+j)] = G[(J*2+I)*4+(j*2+i)];
    return H;
  }

  // arbitrary-pair 2q gate on qubits (a,b) via SWAP routing
  void applyTwoQubit(const std::vector<std::complex<double>>& G4, int a, int b){
    if (a==b) throw std::runtime_error("2q gate on same qubit");
    if (a>b){ applyTwoQubit(swapOperands(G4), b, a); return; }
    static const std::vector<std::complex<double>> SW = swapMat();
    for (int x=b; x>a+1; --x) apply2q(SW, x-1);   // bring qubit b down to a+1
    apply2q(G4, a);
    for (int x=a+1; x<b; ++x) apply2q(SW, x);      // restore order
  }

  // GPU-side <psi|P|psi> for a Pauli string (exact, any chi). Environment E is
  // kept on-device; two O(chi^3) kernel sweeps per site on the owning GPU, with
  // E peer-copied across GPU boundaries.
  std::complex<double> expectPauli(const std::string& paulis){
    int Edev = site[0].dev;
    CUDA_CHECK(cudaSetDevice(Edev));
    cdouble* E; CUDA_CHECK(cudaMalloc(&E, sizeof(cdouble)));
    cdouble one = make_cuDoubleComplex(1,0);
    CUDA_CHECK(cudaMemcpy(E, &one, sizeof(cdouble), cudaMemcpyHostToDevice));
    int dimE = 1;
    const int TH = 256;
    for (int s = 0; s < n; ++s) {
      int dev = site[s].dev, Dl = site[s].Dl, Dr = site[s].Dr;
      if (Edev != dev) {  // bring environment to the GPU that owns this site
        CUDA_CHECK(cudaSetDevice(dev));
        cdouble* E2; CUDA_CHECK(cudaMalloc(&E2, sizeof(cdouble)*(size_t)dimE*dimE));
        CUDA_CHECK(cudaMemcpyPeer(E2, dev, E, Edev, sizeof(cdouble)*(size_t)dimE*dimE));
        CUDA_CHECK(cudaSetDevice(Edev)); CUDA_CHECK(cudaFree(E));
        E = E2; Edev = dev;
      }
      CUDA_CHECK(cudaSetDevice(dev));
      cdouble *R, *S, *newE;
      CUDA_CHECK(cudaMalloc(&R,    sizeof(cdouble)*(size_t)Dl*2*Dr));
      CUDA_CHECK(cudaMalloc(&S,    sizeof(cdouble)*(size_t)Dl*2*Dr));
      CUDA_CHECK(cudaMalloc(&newE, sizeof(cdouble)*(size_t)Dr*Dr));
      char p = paulis.empty() ? 'I' : paulis[s];
      cdouble p00,p01,p10,p11; pauliMat(p, p00,p01,p10,p11);
      k_transL<<<(Dl*2*Dr+TH-1)/TH, TH>>>(E, site[s].d, R, Dl, Dr);
      k_applyP<<<(Dl*2*Dr+TH-1)/TH, TH>>>(R, S, Dl, Dr, p00,p01,p10,p11);
      k_transR<<<(Dr*Dr+TH-1)/TH, TH>>>(site[s].d, S, newE, Dl, Dr);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(R)); CUDA_CHECK(cudaFree(S)); CUDA_CHECK(cudaFree(E));
      E = newE; dimE = Dr;  // Edev stays = dev
    }
    cdouble res; CUDA_CHECK(cudaSetDevice(Edev));
    CUDA_CHECK(cudaMemcpy(&res, E, sizeof(cdouble), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(E));
    return std::complex<double>(res.x, res.y);
  }

  // Stage 2: batched Pauli expectations via a GPU-segment pipeline. Each
  // contiguous GPU segment is a pipeline stage; the per-term environment flows
  // left->right, so after the pipeline fills, every segment (GPU) is working on
  // a different term concurrently. Returns <psi|P_k|psi> for each word.
  std::vector<std::complex<double>> expectPauliBatch(const std::vector<std::string>& words){
    int T = (int)words.size();
    std::vector<std::complex<double>> result(T, std::complex<double>(0,0));
    if (T == 0) return result;

    // ---- contiguous segments from owner[] (one pipeline stage per segment) --
    struct Seg { int dev, s0, s1, dIn, dOut, maxRS, maxB; };
    std::vector<Seg> segs;
    for (int s = 0; s < n; ) {
      int d = site[s].dev, s0 = s;
      while (s < n && site[s].dev == d) ++s;
      Seg sg; sg.dev = d; sg.s0 = s0; sg.s1 = s;
      sg.dIn = site[s0].Dl; sg.dOut = site[s-1].Dr;
      sg.maxRS = 0; sg.maxB = std::max(sg.dIn, sg.dOut);
      for (int k = s0; k < s; ++k) {
        sg.maxRS = std::max(sg.maxRS, site[k].Dl*2*site[k].Dr);
        sg.maxB  = std::max(sg.maxB, std::max(site[k].Dl, site[k].Dr));
      }
      segs.push_back(sg);
    }
    int G = (int)segs.size();

    // ---- preallocate per-segment scratch + boundary buffers (reused) --------
    std::vector<cdouble*> inbuf(G), outbuf(G), Rb(G), Sb(G), Eb0(G), Eb1(G);
    for (int g = 0; g < G; ++g) { auto& sg = segs[g]; CUDA_CHECK(cudaSetDevice(sg.dev));
      CUDA_CHECK(cudaMalloc(&inbuf[g],  sizeof(cdouble)*(size_t)sg.dIn*sg.dIn));
      CUDA_CHECK(cudaMalloc(&outbuf[g], sizeof(cdouble)*(size_t)sg.dOut*sg.dOut));
      CUDA_CHECK(cudaMalloc(&Rb[g],     sizeof(cdouble)*(size_t)sg.maxRS));
      CUDA_CHECK(cudaMalloc(&Sb[g],     sizeof(cdouble)*(size_t)sg.maxRS));
      CUDA_CHECK(cudaMalloc(&Eb0[g],    sizeof(cdouble)*(size_t)sg.maxB*sg.maxB));
      CUDA_CHECK(cudaMalloc(&Eb1[g],    sizeof(cdouble)*(size_t)sg.maxB*sg.maxB));
    }
    // every term enters segment 0 with the 1x1 identity environment; set once.
    { cdouble one = make_cuDoubleComplex(1,0); CUDA_CHECK(cudaSetDevice(segs[0].dev));
      CUDA_CHECK(cudaMemcpy(inbuf[0], &one, sizeof(cdouble), cudaMemcpyHostToDevice)); }

    const int TH = 256;
    // launch one segment's transfer (async on its GPU's default stream; no sync)
    auto segTransfer = [&](int g, const std::string& word){
      auto& sg = segs[g]; CUDA_CHECK(cudaSetDevice(sg.dev));
      cdouble* E = inbuf[g]; int pp = 0;
      for (int s = sg.s0; s < sg.s1; ++s) {
        int Dl = site[s].Dl, Dr = site[s].Dr; bool last = (s == sg.s1 - 1);
        cdouble* dst = last ? outbuf[g] : (pp ? Eb1[g] : Eb0[g]);
        char p = word.empty() ? 'I' : word[s];
        cdouble p00,p01,p10,p11; pauliMat(p, p00,p01,p10,p11);
        k_transL<<<(Dl*2*Dr+TH-1)/TH, TH>>>(E, site[s].d, Rb[g], Dl, Dr);
        k_applyP<<<(Dl*2*Dr+TH-1)/TH, TH>>>(Rb[g], Sb[g], Dl, Dr, p00,p01,p10,p11);
        k_transR<<<(Dr*Dr+TH-1)/TH, TH>>>(site[s].d, Sb[g], dst, Dl, Dr);
        E = dst; pp ^= 1;
      }
    };

    // ---- wavefront: at step t, segment g handles term (t-g) ----------------
    for (int t = 0; t < T + G - 1; ++t) {
      for (int g = 0; g < G; ++g) { int k = t - g; if (k < 0 || k >= T) continue; segTransfer(g, words[k]); }
      for (int g = 0; g < G; ++g) { int k = t - g; if (k < 0 || k >= T) continue;
        CUDA_CHECK(cudaSetDevice(segs[g].dev)); CUDA_CHECK(cudaDeviceSynchronize()); }
      for (int g = 0; g < G; ++g) { int k = t - g; if (k < 0 || k >= T) continue;
        if (g == G-1) { cdouble r; CUDA_CHECK(cudaSetDevice(segs[g].dev));
          CUDA_CHECK(cudaMemcpy(&r, outbuf[g], sizeof(cdouble), cudaMemcpyDeviceToHost));
          result[k] = std::complex<double>(r.x, r.y);
        } else { size_t sz = sizeof(cdouble)*(size_t)segs[g].dOut*segs[g].dOut;
          CUDA_CHECK(cudaMemcpyPeer(inbuf[g+1], segs[g+1].dev, outbuf[g], segs[g].dev, sz)); }
      }
    }
    for (int g = 0; g < G; ++g) { CUDA_CHECK(cudaSetDevice(segs[g].dev));
      cudaFree(inbuf[g]); cudaFree(outbuf[g]); cudaFree(Rb[g]); cudaFree(Sb[g]); cudaFree(Eb0[g]); cudaFree(Eb1[g]); }
    return result;
  }

  // Reference host implementation (O(chi^4)); kept for validation/fallback.
  std::complex<double> expectPauliHost(const std::string& paulis){
    using cd = std::complex<double>;
    auto P = [&](char p, int i, int ip)->cd{
      switch(p){ case 'I': return i==ip?cd(1,0):cd(0,0);
        case 'X': return i!=ip?cd(1,0):cd(0,0);
        case 'Y': return (i==0&&ip==1)?cd(0,-1):((i==1&&ip==0)?cd(0,1):cd(0,0));
        case 'Z': return i==ip?(i==0?cd(1,0):cd(-1,0)):cd(0,0);
        default: return i==ip?cd(1,0):cd(0,0);} };
    // E: (chiL x chiL) transfer; start 1x1 = [1]
    std::vector<cd> E(1, cd(1,0)); int dimE=1;
    for (int s=0;s<n;++s){
      int Dl=site[s].Dl, Dr=site[s].Dr;
      std::vector<cdouble> h((size_t)Dl*2*Dr);
      CUDA_CHECK(cudaSetDevice(site[s].dev));
      CUDA_CHECK(cudaMemcpy(h.data(),site[s].d,sizeof(cdouble)*h.size(),cudaMemcpyDeviceToHost));
      auto A=[&](int a,int i,int c)->cd{ cdouble z=h[(a*2+i)*Dr+c]; return cd(z.x,z.y); };
      char p = paulis.empty()? 'I' : paulis[s];
      std::vector<cd> newE((size_t)Dr*Dr, cd(0,0));
      for (int b=0;b<Dr;++b) for(int bp=0;bp<Dr;++bp){ cd acc(0,0);
        for (int a=0;a<Dl;++a) for(int ap=0;ap<Dl;++ap){ cd e=E[(size_t)a*dimE+ap]; if(e==cd(0,0))continue;
          for(int i=0;i<2;++i) for(int ip=0;ip<2;++ip){ cd pv=P(p,i,ip); if(pv==cd(0,0))continue;
            acc += std::conj(A(a,i,b)) * pv * A(ap,ip,bp) * e; } }
        newE[(size_t)b*Dr+bp]=acc; }
      E.swap(newE); dimE=Dr;
    }
    return E[0];   // 1x1
  }

  std::vector<std::complex<double>> toStatevector(){
    std::vector<std::complex<double>> psi;
    int Dr0=site[0].Dr;
    std::vector<cdouble> h((size_t)1*2*Dr0);
    CUDA_CHECK(cudaSetDevice(site[0].dev));
    CUDA_CHECK(cudaMemcpy(h.data(),site[0].d,sizeof(cdouble)*h.size(),cudaMemcpyDeviceToHost));
    int rows=2, bond=Dr0; psi.resize((size_t)rows*bond);
    for(int i=0;i<2;++i) for(int c=0;c<Dr0;++c) psi[(size_t)i*bond+c]={h[(0*2+i)*Dr0+c].x,h[(0*2+i)*Dr0+c].y};
    for(int s=1;s<n;++s){ int Dc=site[s].Dl, Dr=site[s].Dr;
      std::vector<cdouble> hs((size_t)Dc*2*Dr);
      CUDA_CHECK(cudaSetDevice(site[s].dev));
      CUDA_CHECK(cudaMemcpy(hs.data(),site[s].d,sizeof(cdouble)*hs.size(),cudaMemcpyDeviceToHost));
      std::vector<std::complex<double>> np((size_t)rows*2*Dr, 0.0);
      for(int rIdx=0;rIdx<rows;++rIdx) for(int j=0;j<2;++j) for(int rr=0;rr<Dr;++rr){
        std::complex<double> acc=0;
        for(int c=0;c<Dc;++c) acc += psi[(size_t)rIdx*bond+c]*std::complex<double>(hs[(c*2+j)*Dr+rr].x,hs[(c*2+j)*Dr+rr].y);
        np[((size_t)rIdx*2+j)*Dr+rr]=acc; }
      rows*=2; bond=Dr; psi.swap(np); }
    return psi;
  }

  // Exact MPS perfect sampling (Ferris-Vidal): returns `shots` bitstrings in
  // CUDA-Q qubit order (qubit 0 = leftmost). Site tensors are pulled to host
  // (tiny at the chi used here); precompute right environments, then sample
  // left->right from the exact conditional P(s_k | s_0..s_{k-1}). O(shots*n*chi^2).
  std::vector<std::string> sampleBits(int shots, unsigned seed){
    using cd = std::complex<double>;
    std::vector<std::vector<cd>> A(n); std::vector<int> Dl(n), Dr(n);
    for (int k=0;k<n;++k){ Dl[k]=site[k].Dl; Dr[k]=site[k].Dr;
      std::vector<cdouble> h((size_t)Dl[k]*2*Dr[k]);
      CUDA_CHECK(cudaSetDevice(site[k].dev));
      CUDA_CHECK(cudaMemcpy(h.data(), site[k].d, sizeof(cdouble)*h.size(), cudaMemcpyDeviceToHost));
      A[k].resize(h.size());
      for (size_t t=0;t<h.size();++t) A[k][t]=cd(h[t].x, h[t].y);
    }
    // right environments R[k] (Dl_k x Dl_k); R[n] = [1].
    std::vector<std::vector<cd>> R(n+1);
    R[n].assign(1, cd(1,0));
    for (int k=n-1;k>=0;--k){
      int dl=Dl[k], dr=Dr[k];
      std::vector<cd> T((size_t)dl*2*dr, cd(0,0));   // T[a,i,b'] = sum_b A[a,i,b] R[k+1][b,b']
      for (int a=0;a<dl;++a) for (int i=0;i<2;++i) for (int bp=0;bp<dr;++bp){
        cd s(0,0); for (int b=0;b<dr;++b) s += A[k][(a*2+i)*dr+b]*R[k+1][(size_t)b*dr+bp];
        T[(a*2+i)*dr+bp]=s; }
      std::vector<cd> Rk((size_t)dl*dl, cd(0,0));
      for (int a=0;a<dl;++a) for (int ap=0;ap<dl;++ap){
        cd s(0,0); for (int i=0;i<2;++i) for (int bp=0;bp<dr;++bp) s += T[(a*2+i)*dr+bp]*std::conj(A[k][(ap*2+i)*dr+bp]);
        Rk[(size_t)a*dl+ap]=s; }
      R[k].swap(Rk);
    }
    std::mt19937_64 rng(seed); std::uniform_real_distribution<double> uni(0.0,1.0);
    std::vector<std::string> out; out.reserve(shots);
    for (int sh=0; sh<shots; ++sh){
      std::vector<cd> v(1, cd(1,0));   // left vector over Dl_0 = 1
      std::string bits(n, '0');
      for (int k=0;k<n;++k){
        int dl=Dl[k], dr=Dr[k];
        std::vector<cd> w0(dr,cd(0,0)), w1(dr,cd(0,0));   // w_i[b] = sum_a v[a] A[a,i,b]
        for (int a=0;a<dl;++a){ cd va=v[a];
          for (int b=0;b<dr;++b){ w0[b]+=va*A[k][(a*2+0)*dr+b]; w1[b]+=va*A[k][(a*2+1)*dr+b]; } }
        // P(prefix,s_k=i) = sum_{b,bp} w[b] R[k+1][b,bp] conj(w[bp]), where
        // R[b,bp] = sum_tail t[b] conj(t[bp]). The conj must be on the SECOND
        // factor: MPS tensors are complex (from the SVD) even for real states.
        auto quad=[&](std::vector<cd>& w)->double{ cd s(0,0);
          for (int b=0;b<dr;++b){ cd t(0,0); for (int bp=0;bp<dr;++bp) t += R[k+1][(size_t)b*dr+bp]*std::conj(w[bp]);
            s += w[b]*t; } return s.real(); };
        double p0=quad(w0), p1=quad(w1), tot=p0+p1; if (tot<=0){ p0=p1=0.5; tot=1.0; }
        int si = (uni(rng) < p0/tot) ? 0 : 1;
        bits[k] = si ? '1' : '0';                         // MPS site order
        std::vector<cd>& ws = si ? w1 : w0;
        double nrm = std::sqrt(si ? p1 : p0); if (nrm>0) for (auto& z : ws) z/=nrm;   // renormalize (avoid underflow)
        v.swap(ws);
      }
      std::reverse(bits.begin(), bits.end());   // MPS site k = CUDA-Q qubit n-1-k -> reverse to qubit order
      out.push_back(std::move(bits));
    }
    return out;
  }
};
MGPU_SHARDED_MPS_CUH_EOF

cat > "$BACKEND/simulator_mps_dist.cu" <<'MGPU_SIMULATOR_MPS_DIST_CU_EOF'
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
#include <cstdlib>
#include <memory>

namespace nvqir {

class SimulatorMPSDist : public nvqir::CircuitSimulatorBase<double> {
  using Base = nvqir::CircuitSimulatorBase<double>;
  using GateApplicationTask = typename Base::GateApplicationTask;

  std::unique_ptr<ShardedMPS> mps;
  std::size_t allocated = 0;     // qubits allocated so far
  int maxBond = 64;
  double absCut = 1e-5, relCut = 1e-5;
  std::vector<int> gpuIds;

  // Build the engine once the qubit count is known (VQE allocates all qubits
  // before applying gates, so a single lazy build is sufficient).
  void ensureEngine() {
    if (mps) return;
    if (allocated < 2)
      throw std::runtime_error("tensornet-mps-mgpu needs >= 2 qubits");
    int G = 1; cudaGetDeviceCount(&G);
    gpuIds.clear(); for (int i = 0; i < G; ++i) gpuIds.push_back(i);
    if (auto *b = std::getenv("CUDAQ_MPS_MAX_BOND")) maxBond = std::max(1, atoi(b));
    if (auto *a = std::getenv("CUDAQ_MPS_ABS_CUTOFF")) absCut = atof(a);
    if (auto *r = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF")) relCut = atof(r);
    mps = std::make_unique<ShardedMPS>((int)allocated, gpuIds, maxBond, absCut, relCut);
    mps->distributedBoundary = true;   // cross-GPU bonds use G>1 distSvd
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
MGPU_SIMULATOR_MPS_DIST_CU_EOF

cat > "$BACKEND/tensornet-mps-mgpu.yml" <<'MGPU_TENSORNET-MPS-MGPU_YML_EOF'
# ============================================================================ #
# Target config for the multi-GPU memory-pooled MPS backend.                   #
# Drop into runtime/nvqir/cutensornet/ alongside tensornet-mps.yml.            #
# ============================================================================ #

name: tensornet-mps-mgpu
description: "Multi-GPU memory-pooled matrix product state simulator (sharded chain + distributed truncated SVD)"
gpu-requirements: true

target-arguments:
  - key: option
    required: false
    type: option-flags
    help-string: "Comma-separated options. Supported: 'fp64' (default)."

configuration-matrix:
  - name: multi-gpu-fp64
    option-flags: [fp64]
    default: true
    config:
      nvqir-simulation-backend: tensornet-mps-mgpu
      preprocessor-defines: ["-D CUDAQ_SIMULATION_SCALAR_FP64"]
MGPU_TENSORNET-MPS-MGPU_YML_EOF


# ---- 3. patch the cutensornet CMakeLists to build our backend ---------------
python3 - "$BACKEND/CMakeLists.txt" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
if "tensornet-mps-mgpu" in src:
    print("   CMakeLists already patched"); sys.exit(0)
marker = "nvqir_create_cutn_plugin(tensornet-mps TRUE ${BASE_TENSOR_BACKEND_SRS} simulator_mps_fp64_register.cpp)"
if marker not in src:
    print("   ERROR: expected CMake marker not found", file=sys.stderr); sys.exit(1)
patch = marker + "\n" + "\n".join([
  "  # --- multi-GPU memory-pooled MPS backend (added by setup_local_cudaq.sh) ---",
  "  if (NOT DEFINED NCCL_ROOT AND DEFINED ENV{NCCL_ROOT})",
  "    set(NCCL_ROOT $ENV{NCCL_ROOT})",
  "  endif()",
  "  nvqir_create_cutn_plugin(tensornet-mps-mgpu TRUE simulator_mps_dist.cu)",
  "  target_include_directories(nvqir-tensornet-mps-mgpu PRIVATE ${CMAKE_SOURCE_DIR}/runtime ${NCCL_ROOT}/include)",
  "  target_link_libraries(nvqir-tensornet-mps-mgpu PRIVATE CUDA::cublas CUDA::cusolver ${NCCL_ROOT}/lib/libnccl.so)",
])
src = src.replace(marker, patch, 1)
open(path, "w").write(src)
print("   patched CMakeLists.txt")
PYEOF

# ---- 3b. fix const_iterator symbol localization (so `import cudaq` works) ----
# conda's default CXXFLAGS inject -fvisibility-inlines-hidden. That makes the
# inline members of sum_op<matrix_handler>::const_iterator HIDDEN in TUs that
# implicitly instantiate them (e.g. super_op.cpp); the linker's weak-merge then
# localizes the symbol in libcudaq-operator.so, so the python bindings
# (_quakeDialects) fail at import with an undefined-symbol error. Override the
# flag for the operator library and mark the iterator types default-visible.
python3 - "$SRC/runtime/cudaq/operators/CMakeLists.txt" "$SRC/runtime/cudaq/operators.h" <<'PYEOF'
import sys
cml, hdr = sys.argv[1], sys.argv[2]
s = open(cml).read()
if "-fno-visibility-inlines-hidden" not in s:
    anchor = '-Wno-ctad-maybe-unsupported")'
    if anchor in s:
        s = s.replace(anchor, '-Wno-ctad-maybe-unsupported -fno-visibility-inlines-hidden")', 1)
        open(cml, "w").write(s); print("   patched operators/CMakeLists.txt (inlines-hidden override)")
    else:
        print("   ERROR: CXX_FLAGS anchor not found in operators/CMakeLists.txt", file=sys.stderr); sys.exit(1)
else:
    print("   operators/CMakeLists.txt already patched")
h = open(hdr).read()
if 'struct const_iterator {' in h and '__attribute__((visibility("default"))) const_iterator' not in h:
    h = h.replace('struct const_iterator {',
                  'struct __attribute__((visibility("default"))) const_iterator {')
    open(hdr, "w").write(h); print("   patched operators.h (const_iterator default visibility)")
else:
    print("   operators.h already patched / pattern absent")
PYEOF

# ---- 4. install prefixes + dependency locations -----------------------------
export CUDAQ_INSTALL_PREFIX="$LOCAL/cudaq"   # our build always installs here
export CUDAQ_BUILD_TESTS=FALSE
export CUDACXX="${CUDACXX:-$(command -v nvcc || echo "$CUDA_HOME/bin/nvcc")}"
[ -n "${CUDA_HOME:-}" ] || export CUDA_HOME="${CONDA_PREFIX:-/usr/local/cuda}"
mkdir -p "$CUDAQ_INSTALL_PREFIX"

if [ "$PREBUILT" = 1 ]; then
  # Container: reuse the image's LLVM + cuQuantum + cuTensor (don't rebuild).
  export LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-$LLVM_PREBUILT}"
  : "${CUQUANTUM_INSTALL_PREFIX:=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}}"
  : "${CUTENSOR_INSTALL_PREFIX:=${CUTENSOR_INSTALL_PREFIX:-/opt/nvidia/cutensor}}"
  [ -d "$CUQUANTUM_INSTALL_PREFIX" ] || CUQUANTUM_INSTALL_PREFIX="$SITE/cuquantum"
  [ -d "$CUTENSOR_INSTALL_PREFIX" ]  || CUTENSOR_INSTALL_PREFIX="$SITE/cutensor"
  export CUQUANTUM_INSTALL_PREFIX CUTENSOR_INSTALL_PREFIX
else
  # Bare metal: build everything into $LOCAL.
  export LLVM_INSTALL_PREFIX="$LOCAL/llvm"
  export PYBIND11_INSTALL_PREFIX="$LOCAL/pybind11" BLAS_INSTALL_PREFIX="$LOCAL/blas"
  export ZLIB_INSTALL_PREFIX="$LOCAL/zlib" OPENSSL_INSTALL_PREFIX="$LOCAL/ssl"
  export CURL_INSTALL_PREFIX="$LOCAL/curl" AWS_INSTALL_PREFIX="$LOCAL/aws" QRMI_INSTALL_PREFIX="$LOCAL/qrmi"
  export CUQUANTUM_INSTALL_PREFIX="${CUQUANTUM_INSTALL_PREFIX:-$SITE/cuquantum}"
  export CUTENSOR_INSTALL_PREFIX="${CUTENSOR_INSTALL_PREFIX:-$SITE/cutensor}"
  mkdir -p "$LLVM_INSTALL_PREFIX" "$PYBIND11_INSTALL_PREFIX" "$BLAS_INSTALL_PREFIX" \
           "$ZLIB_INSTALL_PREFIX" "$OPENSSL_INSTALL_PREFIX" "$CURL_INSTALL_PREFIX" \
           "$AWS_INSTALL_PREFIX" "$QRMI_INSTALL_PREFIX"
fi

# NCCL for our backend (include/nccl.h + lib/libnccl.so). Find it; else pip-install.
export NCCL_ROOT="$LOCAL/nccl"; mkdir -p "$NCCL_ROOT/include" "$NCCL_ROOT/lib"
_link_nccl() {  # $1 = a prefix containing include/nccl.h + lib/libnccl.so*
  [ -e "$1/include/nccl.h" ] || return 1
  ln -sf "$1/include/nccl.h" "$NCCL_ROOT/include/nccl.h"
  for so in "$1"/lib/libnccl.so*; do [ -e "$so" ] && ln -sf "$so" "$NCCL_ROOT/lib/$(basename "$so")"; done
  [ -e "$NCCL_ROOT/lib/libnccl.so" ] || { v="$(ls "$NCCL_ROOT"/lib/libnccl.so.* 2>/dev/null|head -1)"; [ -n "$v" ] && ln -sf "$v" "$NCCL_ROOT/lib/libnccl.so"; }
  [ -e "$NCCL_ROOT/lib/libnccl.so" ]
}
_link_nccl "$SITE/nvidia/nccl" || _link_nccl /usr/local/cuda || _link_nccl /usr || {
  echo ">> NCCL not found; pip-installing nvidia-nccl-cu12 ..."
  "$PY" -m pip install -q nvidia-nccl-cu12 2>/dev/null || true
  _link_nccl "$SITE/nvidia/nccl" || echo ">> WARNING: NCCL still missing; set NCCL_ROOT manually."
}
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-}:$NCCL_ROOT:${PYBIND11_INSTALL_PREFIX:-}:${ZLIB_INSTALL_PREFIX:-}:${OPENSSL_INSTALL_PREFIX:-}:${CURL_INSTALL_PREFIX:-}:${BLAS_INSTALL_PREFIX:-}:${CMAKE_PREFIX_PATH:-}"

# ---- 5. build CUDA-Q + our backend ------------------------------------------
cd "$SRC"
if [ "$PREBUILT" = 1 ]; then
  echo ">> PREBUILT mode: skipping install_prerequisites (using container LLVM/cuQuantum)."
else
  echo ">> building prerequisites (LLVM etc.) -> $LOCAL  (long; ~1-3h) ..."
  bash scripts/install_prerequisites.sh
fi
echo ">> building CUDA-Q + tensornet-mps-mgpu -> $CUDAQ_INSTALL_PREFIX ..."
bash scripts/build_cudaq.sh

# ---- 5b. install the VQE app's Python dependencies --------------------------
# Needed so Main.py can run wherever cudaq lives (esp. inside the devdeps
# container, which ships no numpy/qiskit/etc.). cudaq comes from the build above.
REQ="$PROJECT_DIR/requirements.txt"
if [ -f "$REQ" ]; then
  echo ">> installing VQE_Docking Python deps from requirements.txt ..."
  "$PY" -m pip install -q -r "$REQ" || echo ">> WARNING: some Python deps failed to install (see above)."
fi

# ---- 6. point VQE_Docking/Main.py at the mgpu target ------------------------
python3 - "$PROJECT_DIR/VQE_Docking/Main.py" "$GPUS" <<'PYEOF'
import re, sys
p, gpus = sys.argv[1], sys.argv[2]
try:
    s = open(p).read()
except FileNotFoundError:
    print("   (Main.py not found; skipping)"); sys.exit(0)
orig = s
# select the multi-GPU MPS target (fp64; mgpu yml has no fp32 config)
s = re.sub(r"cudaq\.set_target\(\s*['\"](?:tensornet-mps|nvidia)['\"][^)]*\)",
           "cudaq.set_target('tensornet-mps-mgpu')", s)
# device list without spaces (CUDA dislikes 'CUDA_VISIBLE_DEVICES=2, 3')
s = re.sub(r'(CUDA_VISIBLE_DEVICES"\s*,\s*")[^"]*(")', r'\g<1>' + gpus + r'\g<2>', s)
if s != orig:
    open(p, "w").write(s)
    print("   patched Main.py -> tensornet-mps-mgpu, CUDA_VISIBLE_DEVICES=%s" % gpus)
else:
    print("   Main.py unchanged (already mgpu, or no set_target found)")
PYEOF

# ---- 7. write run.sh launcher -----------------------------------------------
cat > "$PROJECT_DIR/run.sh" <<RUNEOF
#!/bin/bash
# Run VQE_Docking on the locally-built tensornet-mps-mgpu backend.
# Works under conda (bare metal) or inside the devdeps container (no conda).
set -e
if command -v conda >/dev/null 2>&1; then
  eval "\$(conda shell.bash hook 2>/dev/null)"; conda activate cudaq-env 2>/dev/null || true
fi
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-$GPUS}"
export CUDAQ_MPS_MAX_BOND="\${CUDAQ_MPS_MAX_BOND:-64}"
CQ="$CUDAQ_INSTALL_PREFIX"
export PYTHONPATH="\$CQ:\${PYTHONPATH:-}"
export LD_LIBRARY_PATH="\$CQ/lib:$NCCL_ROOT/lib:${CONDA_PREFIX:-/usr/lib}/lib:\${LD_LIBRARY_PATH:-}"
PYBIN="\$(command -v python || command -v python3)"
cd "$PROJECT_DIR/VQE_Docking"
"\$PYBIN" "\${1:-Main.py}" "\${@:2}"
RUNEOF
chmod +x "$PROJECT_DIR/run.sh"

cat <<EOF

============================================================================
Done. Installed under: $LOCAL
  CUDA-Q:   $CUDAQ_INSTALL_PREFIX
  backend:  $CUDAQ_INSTALL_PREFIX/lib/libnvqir-tensornet-mps-mgpu.so
  target:   $CUDAQ_INSTALL_PREFIX/targets/tensornet-mps-mgpu.yml

Run the VQE on GPUs $GPUS:
  ./run.sh        (ensure VQE_Docking/Main.py uses cudaq.set_target('tensornet-mps-mgpu'))
============================================================================
EOF
