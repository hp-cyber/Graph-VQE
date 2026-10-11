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

static inline double *heigDevice(GpuCtx& g, int l, cdouble* C) {
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
  CUDA_CHECK(cudaFree(work)); CUDA_CHECK(cudaFree(info));
  return w;
}

static inline void heig(GpuCtx& g, int l, cdouble* C, double* w_host) {
  double *w = heigDevice(g, l, C);
  CUDA_CHECK(cudaMemcpy(w_host, w, sizeof(double)*l, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(w));
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

// Device-resident factors for the singleton-GPU MQPU path. `left` is the
// truncated U factor (m x k, column-major) and `right` is S*Vh (k x n,
// column-major). Keeping both factors on-device avoids the old per-entangler
// GPU -> CPU -> GPU round trip through SvdResult's host vectors.
struct DeviceSvdFactors {
  cdouble *left;
  cdouble *right;
  int k;
};

__global__ static void selectDescendingEigenvectors(
    const cdouble *ascending, cdouble *selected, int l, int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= l * k) return;
  int row = idx % l;
  int col = idx / l;
  selected[row + (size_t)col * l] =
      ascending[row + (size_t)(l - 1 - col) * l];
}

__device__ static unsigned long long svdRandomHash(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Generate the randomized-SVD test matrix directly on the GPU. The old path
// generated 2*n*l normal variates on the CPU and uploaded the full matrix for
// every entangling gate.
__global__ static void fillGaussianOmega(cdouble *omega, size_t count,
                                         unsigned seed) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  unsigned long long a = svdRandomHash(
      ((unsigned long long)seed << 32) ^ idx);
  unsigned long long b = svdRandomHash(a ^ 0xd1b54a32d192ed03ULL);
  double u1 = ((double)(a >> 11) + 1.0) * 0x1.0p-53;
  double u2 = (double)(b >> 11) * 0x1.0p-53;
  double radius = sqrt(-2.0 * log(u1));
  double sine, cosine;
  sincos(6.283185307179586476925286766559 * u2, &sine, &cosine);
  omega[idx] = make_cuDoubleComplex(radius * cosine, radius * sine);
}

__global__ static void selectTruncationRank(const double *ascending,
                                            int *rank, int l, int maxBond,
                                            double absCut, double relCut) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double maximum = ascending[l - 1] > 0.0 ? ascending[l - 1] : 0.0;
  double thresholdSquared = fmax(absCut * absCut,
                                  relCut * relCut * maximum);
  int selected = 0;
  for (int i = l - 1; i >= 0; --i)
    if (ascending[i] > thresholdSquared) ++selected;
  if (selected < 1) selected = 1;
  if (selected > maxBond) selected = maxBond;
  if (selected > l) selected = l;
  *rank = selected;
}

__global__ static void normalizeColumnsFromEigenvalues(
    cdouble *matrix, int rows, int columns, const double *ascending,
    int eigenvalueCount) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * columns) return;
  int column = index / rows;
  double eigenvalue = ascending[eigenvalueCount - 1 - column];
  double inverse = eigenvalue > 0.0 ? rsqrt(eigenvalue) : 1.0;
  matrix[index].x *= inverse;
  matrix[index].y *= inverse;
}

// Randomized truncated SVD specialized for one GPU. This performs the same
// projection/eigendecomposition as distSvd(), but returns U and S*Vh directly
// on the GPU. Only the l singular values (a few hundred doubles) cross to the
// host to choose the truncation rank.
static inline DeviceSvdFactors singleGpuSvdFactors(
    GpuCtx &g, cdouble *matrix, int m, int n, int maxBond,
    double absCut, double relCut, int oversample, unsigned seed) {
  CUDA_CHECK(cudaSetDevice(g.dev));
  (void)oversample;
  (void)seed;

  // For the two-site matrices produced by MPS gates, an exact covariance SVD
  // is substantially cheaper than the previous randomized QR projection. The
  // old routine performed several allocations, QR, five GEMMs, and an
  // eigensolve for every entangler. Here the Gram matrix has dimension
  // min(m,n), and one eigensolve yields the exact truncated factors.
  const bool useLeftGram = m <= n;
  const int gramDimension = useLeftGram ? m : n;
  cdouble *covariance = nullptr;
  CUDA_CHECK(cudaMalloc(
      &covariance,
      sizeof(cdouble) * (size_t)gramDimension * gramDimension));
  if (useLeftGram) {
    CUBLAS_CHECK(cublasZgemm(
        g.blas, CUBLAS_OP_N, CUBLAS_OP_C, m, m, n, &C_ONE, matrix, m,
        matrix, m, &C_ZERO, covariance, m));
  } else {
    CUBLAS_CHECK(cublasZgemm(
        g.blas, CUBLAS_OP_C, CUBLAS_OP_N, n, n, m, &C_ONE, matrix, m,
        matrix, m, &C_ZERO, covariance, n));
  }

  double *eigenvalues = heigDevice(g, gramDimension, covariance);
  int *deviceRank = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceRank, sizeof(int)));
  selectTruncationRank<<<1, 1, 0, g.stream>>>(
      eigenvalues, deviceRank, gramDimension, maxBond, absCut, relCut);
  int k = 1;
  CUDA_CHECK(cudaMemcpyAsync(&k, deviceRank, sizeof(int),
                             cudaMemcpyDeviceToHost, g.stream));
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  CUDA_CHECK(cudaFree(deviceRank));

  cdouble *basis = nullptr, *left = nullptr, *right = nullptr;
  CUDA_CHECK(cudaMalloc(
      &basis, sizeof(cdouble) * (size_t)gramDimension * k));
  int threads = 256;
  selectDescendingEigenvectors<<<
      (gramDimension * k + threads - 1) / threads, threads, 0, g.stream>>>(
      covariance, basis, gramDimension, k);

  if (useLeftGram) {
    // The selected covariance eigenvectors are U directly.
    left = basis;
    basis = nullptr;
  } else {
    // V comes from M^H M. Form U = M V S^-1.
    CUDA_CHECK(cudaMalloc(&left, sizeof(cdouble) * (size_t)m * k));
    CUBLAS_CHECK(cublasZgemm(
        g.blas, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &C_ONE, matrix, m,
        basis, n, &C_ZERO, left, m));
    normalizeColumnsFromEigenvalues<<<
        (m * k + threads - 1) / threads, threads, 0, g.stream>>>(
        left, m, k, eigenvalues, gramDimension);
  }

  // U^H M is S*Vh directly.
  CUDA_CHECK(cudaMalloc(&right, sizeof(cdouble) * (size_t)k * n));
  CUBLAS_CHECK(cublasZgemm(
      g.blas, CUBLAS_OP_C, CUBLAS_OP_N, k, n, m, &C_ONE, left, m,
      matrix, m, &C_ZERO, right, k));
  CUDA_CHECK(cudaStreamSynchronize(g.stream));

  if (basis) CUDA_CHECK(cudaFree(basis));
  CUDA_CHECK(cudaFree(covariance));
  CUDA_CHECK(cudaFree(eigenvalues));
  return {left, right, k};
}

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
