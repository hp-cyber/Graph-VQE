// Exact Pauli-prefix DAG contractions. One device, bounded chunks; no SVD changes.
// Prefix maps depend only on the Hamiltonian. Numerical values are recomputed
// every call. Workspaces remain allocated for the lifetime of this MPS.
__global__ static void analyticLeft(const cdouble *E, const cdouble *A,
    cdouble *S, const int *parents, const char *paulis,
    int dl, int dr, int nodes, size_t stride) {
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  size_t width=(size_t)dl*2*dr;
  if(idx>=nodes*width)return;
  int node=idx/width, b=idx%dr, i=(idx/dr)%2, a=(idx/dr/2)%dl;
  char p=paulis[node]; int ip=(p=='X'||p=='Y')?1-i:i;
  cdouble phase=make_cuDoubleComplex(1,0);
  if(p=='Z'&&i)phase.x=-1;
  if(p=='Y')phase=make_cuDoubleComplex(0,i?1:-1);
  cdouble sum=make_cuDoubleComplex(0,0);
  const cdouble *env=E+(size_t)parents[node]*stride;
  for(int ap=0;ap<dl;++ap)
    sum=cuCadd(sum,cuCmul(env[a*dl+ap],A[(ap*2+ip)*dr+b]));
  S[(size_t)node*2*stride+(a*2+i)*dr+b]=cuCmul(phase,sum);
}
__global__ static void analyticRight(const cdouble *A,const cdouble *S,
    cdouble *out,int dl,int dr,int nodes,size_t stride) {
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  size_t width=(size_t)dr*dr;
  if(idx>=nodes*width)return;
  int node=idx/width,b=(idx/ dr)%dr,bp=idx%dr;
  cdouble sum=make_cuDoubleComplex(0,0);
  for(int t=0;t<dl*2;++t)
    sum=cuCadd(sum,cuCmul(cuConj(A[t*dr+b]),S[(size_t)node*2*stride+t*dr+bp]));
  out[(size_t)node*stride+b*dr+bp]=sum;
}
__global__ static void analyticGather(const cdouble *E,cdouble *out,
    const int *leaves,int count,size_t stride) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)out[i]=E[(size_t)leaves[i]*stride];
}
struct AnalyticWorkspace {
  cdouble *e0=nullptr,*e1=nullptr,*scratch=nullptr,*output=nullptr;
  int *parents=nullptr,*leaves=nullptr;
  char *paulis=nullptr;
  size_t capacity=0;
  int device=-1,chunkSize=0,sites=0,allocatedChunk=0,allocatedSites=0;
  std::vector<std::string> key;
  struct Chunk { int offset,count; std::vector<int> counts,parents,leaves; std::vector<char> paulis; };
  std::vector<Chunk> chunks;
  ~AnalyticWorkspace(){ release(); }
  void release(){
    if(device<0)return;
    cudaSetDevice(device);
    cudaFree(e0);cudaFree(e1);cudaFree(scratch);cudaFree(output);
    cudaFree(parents);cudaFree(leaves);cudaFree(paulis);
    e0=e1=scratch=output=nullptr;parents=leaves=nullptr;paulis=nullptr;capacity=0;
  }
};
