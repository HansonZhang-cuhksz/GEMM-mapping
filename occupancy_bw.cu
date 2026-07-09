// Measure effective DRAM bandwidth vs occupancy, to test the estimator's
// assumption bw_eff = bw_physical * sm_util.
//
// Two independent occupancy axes, each isolated by forcing exactly ONE block per
// SM (via a large dynamic-SMEM reservation that makes a 2nd block not fit):
//   (A) SM-count sweep : gridDim = k blocks (k = 1..2*num_sm), blockDim fixed large.
//                        occupancy_x = k / num_sm  (== the model's sm_util).
//   (B) warps/SM sweep : gridDim = num_sm, blockDim = 32..1024 threads.
//                        occupancy_x = threads / max_threads_per_sm.
// Each config streams the same 512 MiB (grid-stride read); bw_eff = bytes / min_time.
//
// Output: CSV to stdout (sweep,x_occupancy,active_units,threads,bw_GBs).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define CK(x) do{cudaError_t e_=(x); if(e_){printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);} }while(0)

// Read-only streaming (XOR-reduce so it isn't dead-code-eliminated). Reserves
// dynamic shared memory only to cap occupancy at 1 block/SM; it isn't otherwise used.
__global__ void stream_read(const uint4* __restrict__ p, size_t n, uint4* sink){
  extern __shared__ char smem[];
  if (threadIdx.x == 12345) smem[0] = 1;                 // keep the reservation live
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  uint4 acc = make_uint4(0,0,0,0);
  for(; i<n; i+=stride){ uint4 v=p[i]; acc.x^=v.x; acc.y^=v.y; acc.z^=v.z; acc.w^=v.w; }
  if(acc.x==0xdeadbeef) sink[threadIdx.x]=acc;
}

static const uint4* g_p; static uint4* g_sink; static size_t g_n; static int g_smem;

static float best_ms(int grid, int block, int reps){
  cudaEvent_t b,e; CK(cudaEventCreate(&b)); CK(cudaEventCreate(&e));
  stream_read<<<grid,block,g_smem>>>(g_p,g_n,g_sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float best=1e30f;
  for(int r=0;r<reps;++r){
    CK(cudaEventRecord(b)); stream_read<<<grid,block,g_smem>>>(g_p,g_n,g_sink);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms=0; CK(cudaEventElapsedTime(&ms,b,e)); if(ms<best) best=ms;
  }
  CK(cudaEventDestroy(b)); CK(cudaEventDestroy(e)); return best;
}

int main(){
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  int num_sm = prop.multiProcessorCount;
  int max_thr_sm = prop.maxThreadsPerMultiProcessor;
  // Reserve enough dynamic SMEM that only one block fits per SM (2*g_smem > SMEM/SM).
  g_smem = (int)(prop.sharedMemPerMultiprocessor * 0.6);
  CK(cudaFuncSetAttribute(stream_read, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem));

  const size_t BYTES = 512ull*1024*1024;   // 512 MiB >> 32 MiB L2
  g_n = BYTES/sizeof(uint4);
  CK(cudaMalloc((void**)&g_p, BYTES)); CK(cudaMalloc((void**)&g_sink, 1024*sizeof(uint4)));
  CK(cudaMemset((void*)g_p, 1, BYTES));
  const int reps = 30;
  const double GB = 1e9;

  fprintf(stderr,"# num_sm=%d max_threads/SM=%d dyn_smem/block=%d B (forces 1 block/SM)\n",
          num_sm, max_thr_sm, g_smem);
  printf("sweep,x_occupancy,active_units,threads,bw_GBs\n");

  // (A) SM-count sweep: k blocks of 1024 threads, k = 1..2*num_sm.
  for(int k=1;k<=2*num_sm;++k){
    float ms = best_ms(k, 1024, reps);
    double bw = BYTES/(ms*1e-3)/GB;
    printf("smcount,%.4f,%d,%d,%.1f\n", (double)k/num_sm, k, 1024, bw);
  }
  // (B) warps/SM sweep: all num_sm SMs active, blockDim = 32..1024.
  for(int t=32; t<=1024; t+=32){
    float ms = best_ms(num_sm, t, reps);
    double bw = BYTES/(ms*1e-3)/GB;
    printf("warps,%.4f,%d,%d,%.1f\n", (double)t/max_thr_sm, num_sm, t, bw);
  }
  return 0;
}
