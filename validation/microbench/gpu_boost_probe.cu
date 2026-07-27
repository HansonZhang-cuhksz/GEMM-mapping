// Warm the GPU with a CONTINUOUS load (no host-sync gaps so it can boost out of
// its idle underclock), then measure real sustained SM clock + compute throughput.
//   * SM clock          : clock64() cycles / wall time of a warmed kernel.
//   * FP32 FMA throughput: known FLOP count / time.
//   * FP16 tensor peak   : back-to-back cuBLAS 4096^3 GEMMs (one sync), warmed.
// Run alongside:  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <chrono>

#define CK(x)  do{cudaError_t e_=(x); if(e_){printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);return 1;}}while(0)
#define CB(x)  do{cublasStatus_t s_=(x); if(s_){printf("cuBLAS %d @%d\n",(int)s_,__LINE__);return 1;}}while(0)
using clk = std::chrono::steady_clock;
static double now_s(){ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

// 4 dependent FMAs / iter = 8 FLOP/iter, register-resident (no memory traffic).
__global__ void fma_kernel(long iters, float* out, long long* cyc){
  float a=threadIdx.x*1e-3f+0.1f, b=1.0001f, c=0.9999f, d=1.00002f;
  long long t0=clock64();
  for(long i=0;i<iters;++i){ a=a*b+c; b=b*c+d; c=c*d+a; d=d*a+b; }
  long long t1=clock64();
  out[blockIdx.x*blockDim.x+threadIdx.x]=a+b+c+d;
  if(threadIdx.x==0 && blockIdx.x==0) *cyc=t1-t0;
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int nsm=p.multiProcessorCount;
  printf("GPU: %s  SMs=%d  spec-maxSMclk=%.0f MHz\n", p.name, nsm, p.clockRate/1e3);
  int grid=nsm*16, block=256; long threads=(long)grid*block;
  float* out; long long* dcyc; CK(cudaMalloc(&out,threads*sizeof(float))); CK(cudaMalloc(&dcyc,8));
  cudaEvent_t b,e; CK(cudaEventCreate(&b)); CK(cudaEventCreate(&e));

  // ---- calibrate iters -> time so we can size ~0.4 s kernels (bounded backlog) ----
  float ms=0;
  CK(cudaEventRecord(b)); fma_kernel<<<grid,block>>>(2'000'000,out,dcyc); CK(cudaEventRecord(e));
  CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms,b,e));
  long iters_040 = (long)(2'000'000 * (400.0/ms));      // iters for ~0.4 s at current clock

  // ---- WARM-UP: ~6 s of continuous load, one sync per 0.4 s kernel (>99% duty) ----
  printf("warming up (continuous FMA load, ~6 s)...\n"); fflush(stdout);
  double t0=now_s(); int launches=0;
  while(now_s()-t0 < 6.0){ fma_kernel<<<grid,block>>>(iters_040,out,dcyc); CK(cudaDeviceSynchronize()); ++launches; }
  printf("  (%d warm kernels run)\n", launches);

  // ---- MEASURE SM clock + FP32 throughput on a warmed kernel (~0.8 s) ----
  long M_ITERS = iters_040*2;
  CK(cudaEventRecord(b)); fma_kernel<<<grid,block>>>(M_ITERS,out,dcyc); CK(cudaEventRecord(e));
  CK(cudaEventSynchronize(e));
  CK(cudaEventElapsedTime(&ms,b,e));
  long long cyc; CK(cudaMemcpy(&cyc,dcyc,8,cudaMemcpyDeviceToHost));
  double ghz = cyc/(ms*1e6);
  double fp32_flops = (double)threads*M_ITERS*8.0/(ms*1e-3);
  printf("SM clock (clock64/wall)   : %.3f GHz\n", ghz);
  printf("FP32 FMA throughput       : %.1f GFLOP/s  (%.2f TFLOP/s)\n", fp32_flops/1e9, fp32_flops/1e12);

  // ---- FP16 tensor peak via back-to-back cuBLAS 4096^3 (one sync) ----
  int N=4096; size_t szh=(size_t)N*N*sizeof(__half), szf=(size_t)N*N*sizeof(float);
  __half *dA,*dB; float *dC; CK(cudaMalloc(&dA,szh)); CK(cudaMalloc(&dB,szh)); CK(cudaMalloc(&dC,szf));
  CK(cudaMemset(dA,1,szh)); CK(cudaMemset(dB,1,szh));
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
  float al=1,be=0;
  auto gemm=[&](){ cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&al,dA,CUDA_R_16F,N,dB,CUDA_R_16F,N,
                                &be,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
  for(int i=0;i<10;++i) gemm(); CK(cudaDeviceSynchronize());          // warm
  int iters=60;
  CK(cudaEventRecord(b)); for(int i=0;i<iters;++i) gemm(); CK(cudaEventRecord(e));
  CK(cudaEventSynchronize(e)); CK(cudaEventElapsedTime(&ms,b,e));
  double tflops = 2.0*N*N*N*iters/(ms*1e-3)/1e12;
  printf("FP16 tensor peak (cuBLAS) : %.1f TFLOP/s  (%.3f ms/gemm, 4096^3, back-to-back)\n",
         tflops, ms/iters);
  return 0;
}
