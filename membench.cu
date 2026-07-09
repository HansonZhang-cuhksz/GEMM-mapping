// Device memory microbenchmarks: streaming BW, pointer-chase latency, real SM clock.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#define CK(x) do{cudaError_t err_=(x); if(err_){printf("CUDA %s @%d\n",cudaGetErrorString(err_),__LINE__);exit(1);} }while(0)

// ---- streaming read (sum reduction), vectorized uint4 ----
__global__ void read_kernel(const uint4* __restrict__ p, size_t n, uint4* sink){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  uint4 acc = make_uint4(0,0,0,0);
  for(; i<n; i+=stride){ uint4 v=p[i]; acc.x^=v.x; acc.y^=v.y; acc.z^=v.z; acc.w^=v.w; }
  if(acc.x==0xdeadbeef) sink[threadIdx.x]=acc;   // prevent DCE
}
// ---- streaming copy (read+write) ----
__global__ void copy_kernel(const uint4* __restrict__ s, uint4* __restrict__ d, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=stride) d[i]=s[i];
}
// ---- pointer chase (dependent loads) ----
__global__ void chase_kernel(const int* __restrict__ a, int start, long iters, int* sink){
  int p=start;
  for(long i=0;i<iters;++i) p=a[p];
  *sink=p;
}
// ---- real SM clock: cycles vs wall time ----
__global__ void clock_kernel(long iters, float* out, long long* cyc){
  long long t0=clock64();
  float x=threadIdx.x*1e-3f, y=1.0001f;
  for(long i=0;i<iters;++i){ x=x*y+y; y=y*1.0000001f+1e-6f; }
  long long t1=clock64();
  if(threadIdx.x==0 && blockIdx.x==0){ *out=x+y; *cyc=t1-t0; }
}

static float time_kernel(void(*launch)(), int reps){
  cudaEvent_t b,e; CK(cudaEventCreate(&b)); CK(cudaEventCreate(&e));
  launch(); CK(cudaDeviceSynchronize());               // warmup
  float best=1e30f;
  for(int r=0;r<reps;++r){
    CK(cudaEventRecord(b)); launch(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms=0; CK(cudaEventElapsedTime(&ms,b,e)); if(ms<best) best=ms;
  }
  CK(cudaEventDestroy(b)); CK(cudaEventDestroy(e)); return best;   // best = peak (least throttled)
}

static const uint4* g_p; static uint4* g_d; static uint4* g_sink; static size_t g_n;
static const int* g_a; static int* g_isink;
static void L_read(){ read_kernel<<<2048,256>>>(g_p,g_n,g_sink); }
static void L_copy(){ copy_kernel<<<2048,256>>>(g_p,g_d,g_n); }

int main(){
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  printf("GPU: %s   spec memClk=%.0f MHz busWidth=%d-bit  => spec BW=%.1f GB/s\n",
         prop.name, prop.memoryClockRate/1e3, prop.memoryBusWidth,
         2.0*prop.memoryClockRate*1e3*(prop.memoryBusWidth/8)/1e9);

  // ---------- streaming bandwidth ----------
  const size_t BYTES = 512ull*1024*1024;      // 512 MiB >> 32 MiB L2
  g_n = BYTES/sizeof(uint4);
  CK(cudaMalloc((void**)&g_p, BYTES)); CK(cudaMalloc((void**)&g_d, BYTES));
  CK(cudaMalloc((void**)&g_sink, 256*sizeof(uint4)));
  CK(cudaMemset((void*)g_p,1,BYTES));
  int reps=50;
  float rd_ms=time_kernel(L_read,reps), cp_ms=time_kernel(L_copy,reps);
  double rd_bw=BYTES/(rd_ms*1e-3)/1e9;         // read-only
  double cp_bw=2.0*BYTES/(cp_ms*1e-3)/1e9;      // read+write
  printf("streaming BW: read-only = %.1f GB/s   copy(r+w) = %.1f GB/s\n", rd_bw, cp_bw);

  // ---------- real SM clock ----------
  float* dclkout; long long* dcyc; CK(cudaMalloc(&dclkout,4)); CK(cudaMalloc(&dcyc,8));
  long citer=20000000;
  cudaEvent_t b,e; CK(cudaEventCreate(&b)); CK(cudaEventCreate(&e));
  clock_kernel<<<prop.multiProcessorCount*4,256>>>(citer,dclkout,dcyc); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(b)); clock_kernel<<<prop.multiProcessorCount*4,256>>>(citer,dclkout,dcyc);
  CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
  float clk_ms=0; CK(cudaEventElapsedTime(&clk_ms,b,e));
  long long cyc; CK(cudaMemcpy(&cyc,dcyc,8,cudaMemcpyDeviceToHost));
  double sm_ghz = cyc/(clk_ms*1e6);
  printf("sustained SM clock (clock64/wall) = %.3f GHz  (%lld cycles in %.3f ms)\n", sm_ghz, cyc, clk_ms);

  // ---------- pointer-chase latency ----------
  const size_t LN = 64ull*1024*1024;           // 64M ints = 256 MiB >> L2
  std::vector<int> perm(LN); std::iota(perm.begin(),perm.end(),0);
  std::mt19937_64 rng(12345); std::shuffle(perm.begin(),perm.end(),rng);
  std::vector<int> nxt(LN);
  for(size_t i=0;i<LN;++i) nxt[perm[i]]=perm[(i+1)%LN];   // single Hamiltonian cycle
  int* da; CK(cudaMalloc(&da,LN*sizeof(int))); CK(cudaMemcpy(da,nxt.data(),LN*sizeof(int),cudaMemcpyHostToDevice));
  int* dsink; CK(cudaMalloc(&dsink,4));
  long chase=8000000;
  chase_kernel<<<1,1>>>(da,0,100000,dsink); CK(cudaDeviceSynchronize());   // warmup
  CK(cudaEventRecord(b)); chase_kernel<<<1,1>>>(da,0,chase,dsink);
  CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
  float ch_ms=0; CK(cudaEventElapsedTime(&ch_ms,b,e));
  double lat_ns = ch_ms*1e6/chase;
  printf("pointer-chase latency = %.1f ns/access  = %.0f SM cycles @ %.3f GHz\n",
         lat_ns, lat_ns*sm_ghz, sm_ghz);
  return 0;
}
