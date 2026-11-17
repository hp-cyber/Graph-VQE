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
    if (G > 1) {
      // Multi-GPU: init NCCL communicators for cross-GPU allreduce / peer copies.
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
    } else {
      // Single GPU: no NCCL needed. allreduceSum and peer-copies in dist_svd.cuh
      // and expectPauli/expectPauliBatch are already guarded by G==1 checks.
      pool[0].dev = gpu_ids[0]; CUDA_CHECK(cudaSetDevice(pool[0].dev));
      CUDA_CHECK(cudaStreamCreate(&pool[0].stream));
      CUBLAS_CHECK(cublasCreate(&pool[0].blas)); CUBLAS_CHECK(cublasSetStream(pool[0].blas, pool[0].stream));
      CUSOLVER_CHECK(cusolverDnCreate(&pool[0].solver)); CUSOLVER_CHECK(cusolverDnSetStream(pool[0].solver, pool[0].stream));
      pool[0].comm = nullptr;
    }
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
