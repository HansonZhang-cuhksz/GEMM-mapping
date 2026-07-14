// =============================================================================
//  gemm.cu — CUTLASS GEMM mapping playground with a cuBLAS baseline
// -----------------------------------------------------------------------------
//  What it does
//    * Runs a mixed-precision (fp16 OR bf16 in / fp32 accumulate / fp32 out) GEMM
//        C[M,N] = A[M,K] * B[K,N]   (all row-major)
//      through several CUTLASS "mappings" and through cuBLAS.
//    * TEST-DRIVEN: before any timing, it verifies every kernel against a CPU
//      reference (via cuBLAS, itself checked against a CPU oracle at startup).
//      If a kernel is wrong, its time is never reported.
//    * Prints run time (ms), TFLOP/s, and speedup vs. cuBLAS for each mapping,
//      then AUTO-TUNES: picks the fastest correct mapping for the given size and
//      prints its full schedule (tile / warp / MMA / stages / swizzle).
//
//  The "GEMM mapping" knobs (this is what you experiment with)
//    CUTLASS exposes the schedule as COMPILE-TIME template parameters:
//       - ThreadblockShape  (CTA tile)   <- tiling dimensions
//       - WarpShape         (warp tile)  <- tiling dimensions
//       - InstructionShape  (MMA tile)   <- tensor-core instruction (16x8x16)
//       - Stages            (pipeline)   <- software-pipelining depth == Triton num_stages
//       - ThreadblockSwizzle(raster)     <- loop order over output tiles (L2 locality)
//    Because they are compile-time, you experiment two ways:
//       1. Edit the PRIMARY MAPPING block below and recompile (the "custom" config).
//       2. Pick from the pre-built REGISTRY at runtime (no recompile) with --config.
//
//  On "let CUTLASS pick the mapping": there is no runtime CUTLASS oracle that
//  returns the optimal tile for a size. CUTLASS's native auto-selection is either
//  OFFLINE (the `cutlass_profiler` sweeps kernels) or COMPILE-TIME (the 3.x
//  CollectiveBuilder with KernelScheduleAuto picks a schedule for the arch). The
//  practical runtime answer is empirical: sweep candidates and keep the winner.
//  That is what the [auto-tune] step below does, and it prints the chosen mapping.
//  (The library that DOES expose a runtime heuristic returning a readable tile is
//   cuBLASLt via cublasLtMatmulAlgoGetHeuristic — a different library.)
//
//  On "loop order": CUTLASS does NOT let you reorder the inner K mainloop. What it
//  exposes is the order threadblocks walk the output-tile grid (ThreadblockSwizzle);
//  that rasterization is the practical "loop order" knob — it changes L2 reuse.
//
//  Build:  ./build.sh          Run:  ./run.sh   (or ./gemm --help)
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>          // cutlass::half_t, cutlass::bfloat16_t
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

// =============================================================================
//  GLOBAL VARIABLES — problem size & benchmark controls (also overridable on CLI)
// =============================================================================
int  G_M       = 128;    // rows of A / C
int  G_N       = 4096;   // cols of B / C
int  G_K       = 4096;   // contraction dim
int  G_WARMUP  = 10;     // warmup iterations (not timed)
int  G_ITERS   = 50;     // timed iterations (averaged)
bool G_RUN_ALL = true;   // run every registry mapping; false => only --config
std::string G_ONLY  = ""; // name filter for --config (comma-separated substrings, OR)
std::string G_DTYPE = "both"; // input precision: fp16 | bf16 | both
int  G_ROUNDS = 9;       // rounds whose median is taken (interleaved A/B vs cuBLAS)
bool G_NO_SPLITK = false; // exclude split-K mappings from the sweep & auto-tune

// =============================================================================
//  PRIMARY MAPPING  — edit these and recompile to explore the "custom" schedule.
//  (These feed the registry entry named "custom".)
// =============================================================================
#define CFG_TB_M     128   // threadblock (CTA) tile M   \_ tiling dimensions
#define CFG_TB_N     128   // threadblock (CTA) tile N   /
#define CFG_TB_K      32   // threadblock (CTA) tile K   (mainloop step)
#define CFG_WARP_M    64   // warp tile M                \_ tiling dimensions
#define CFG_WARP_N    64   // warp tile N                /
#define CFG_WARP_K    32   // warp tile K
#define CFG_INST_M    16   // MMA instruction M   (tensor-core shape; fixed 16x8x16)
#define CFG_INST_N     8   // MMA instruction N
#define CFG_INST_K    16   // MMA instruction K
#define CFG_STAGES     3   // software-pipeline stages    == Triton num_stages
#define CFG_SWIZZLE    1   // rasterization group log (loop order / L2 locality)

// =============================================================================
//  Output/accumulate types & layouts (shared by every mapping and both dtypes).
//  The INPUT element (fp16 or bf16) is a template parameter, see GemmMappingT.
// =============================================================================
using ElementC   = float;
using ElementAcc = float;
using LayoutA    = cutlass::layout::RowMajor;
using LayoutB    = cutlass::layout::RowMajor;
using LayoutC    = cutlass::layout::RowMajor;

// 128-bit vectorized epilogue store for fp32 output (4 floats).
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc, ElementAcc>;

// Alias template: a CUTLASS tensor-op GEMM parameterized by the INPUT element type
// (fp16 or bf16) and the mapping knobs.  ArchTag Sm80 selects the Ampere-style
// multistage mainloop, which runs on the RTX 4060 (Ada, sm_89); compile -arch=sm_89.
template <typename Element,
          int TBM, int TBN, int TBK,
          int WM,  int WN,  int WK,
          int IM,  int IN,  int IK,
          int Stages, int SwizzleLog>
using GemmMappingT = cutlass::gemm::device::Gemm<
    Element, LayoutA,
    Element, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TBM, TBN, TBK>,   // ThreadblockShape  (tiling)
    cutlass::gemm::GemmShape<WM,  WN,  WK >,   // WarpShape         (tiling)
    cutlass::gemm::GemmShape<IM,  IN,  IK >,   // InstructionShape  (MMA)
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleLog>, // loop order
    Stages,                                    // software pipeline depth
    8, 8>;                                     // alignment of A, B (128-bit loads)

// Same, but with SERIAL SPLIT-K (in-place semaphore reduction) enabled — the
// direct analog of cuBLAS's numSplitsK + REDUCTION_SCHEME_INPLACE. The number of
// K-slices is a RUNTIME argument (split_k_slices), passed via Gemm::Arguments.
template <typename Element,
          int TBM, int TBN, int TBK,
          int WM,  int WN,  int WK,
          int IM,  int IN,  int IK,
          int Stages, int SwizzleLog>
using GemmMappingSplitKT = cutlass::gemm::device::Gemm<
    Element, LayoutA,
    Element, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TBM, TBN, TBK>,
    cutlass::gemm::GemmShape<WM,  WN,  WK >,
    cutlass::gemm::GemmShape<IM,  IN,  IK >,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleLog>,
    Stages,
    8, 8,
    true>;                                     // SplitKSerial = true

// Map a CUTLASS input element type -> the cuBLAS cudaDataType and a label.
template <typename E> struct DtypeTraits;
template <> struct DtypeTraits<cutlass::half_t> {
  static constexpr cudaDataType_t cublas = CUDA_R_16F;  static constexpr const char* name = "fp16";
};
template <> struct DtypeTraits<cutlass::bfloat16_t> {
  static constexpr cudaDataType_t cublas = CUDA_R_16BF; static constexpr const char* name = "bf16";
};

// =============================================================================
//  Small helpers
// =============================================================================
#define CUDA_CHECK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) { \
  std::fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); \
  std::exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr,"cuBLAS error %d at %s:%d\n",(int)s_,__FILE__,__LINE__); \
  std::exit(1);} } while(0)

// The mapping (schedule) actually used by a kernel, read off its CUTLASS type.
struct MappingInfo {
  bool valid = false;
  int tbM=0,tbN=0,tbK=0, wM=0,wN=0,wK=0, iM=0,iN=0,iK=0, stages=0, swizzle=0, splitk=1;
};

struct BenchResult {
  std::string name;
  bool  implementable = false;
  bool  correct       = false;
  double max_rel      = 0.0;   // max relative error vs reference
  double ms           = 0.0;   // average run time (ms)
  double tflops       = 0.0;
  double fair_speedup = 0.0;   // median (cuBLAS_ms / this_ms) from interleaved A/B
  double fair_cublas_ms = 0.0; // median cuBLAS ms measured adjacent to this kernel
  MappingInfo map;
};

// Device tensors reused by every kernel within one dtype pass.
static void  *g_dA = nullptr, *g_dB = nullptr;  // inputs (fp16 or bf16, cast per kernel)
static float *g_dC = nullptr;                   // per-kernel output
static std::vector<float> g_ref_host;           // cuBLAS reference (host)
static int g_M, g_N, g_K;
static cublasHandle_t g_handle = nullptr;       // for interleaved fair comparison

// CPU reference GEMM in fp32 (used only for the startup oracle check).
static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[i*K+k] * B[k*N+j];
      C[i*N+j] = acc;
    }
}

// Row-major C = A*B via column-major cuBLAS (swap operands). fp16/bf16 in, fp32 out.
template <typename Element>
static void cublas_gemm(cublasHandle_t h, const void* dA, const void* dB, float* dC,
                        int M, int N, int K) {
  float alpha = 1.f, beta = 0.f;
  cudaDataType_t dt = DtypeTraits<Element>::cublas;
  CUBLAS_CHECK(cublasGemmEx(
      h, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K, &alpha,
      dB, dt, N,
      dA, dt, K,
      &beta, dC, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Max relative error of a device result buffer vs the host reference.
static double max_rel_error(const float* dC, const std::vector<float>& ref) {
  std::vector<float> host(ref.size());
  CUDA_CHECK(cudaMemcpy(host.data(), dC, ref.size()*sizeof(float), cudaMemcpyDeviceToHost));
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    num = std::max(num, (double)std::fabs(host[i] - ref[i]));
    den = std::max(den, (double)std::fabs(ref[i]));
  }
  return den > 0 ? num / den : num;
}

// Average ms per launch over `iters` back-to-back launches (one event pair).
template <typename F>
static double time_launches(F&& launch, int iters) {
  cudaEvent_t b, e; CUDA_CHECK(cudaEventCreate(&b)); CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(b));
  for (int i = 0; i < iters; ++i) launch();
  CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, b, e));
  CUDA_CHECK(cudaEventDestroy(b)); CUDA_CHECK(cudaEventDestroy(e));
  return ms / iters;
}
static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size()/2];
}

// Read the schedule off a CUTLASS Gemm type (swizzle log passed in — not on type).
template <typename Gemm>
static MappingInfo mapping_of(int swizzle_log, int split_k) {
  MappingInfo m; m.valid = true;
  m.tbM = Gemm::ThreadblockShape::kM; m.tbN = Gemm::ThreadblockShape::kN; m.tbK = Gemm::ThreadblockShape::kK;
  m.wM  = Gemm::WarpShape::kM;        m.wN  = Gemm::WarpShape::kN;        m.wK  = Gemm::WarpShape::kK;
  m.iM  = Gemm::InstructionShape::kM; m.iN  = Gemm::InstructionShape::kN; m.iK  = Gemm::InstructionShape::kK;
  m.stages = Gemm::kStages;           m.swizzle = swizzle_log;            m.splitk = split_k;
  return m;
}

// =============================================================================
//  Run one CUTLASS mapping: correctness check first, then time it.
// =============================================================================
template <typename Gemm>
static BenchResult run_cutlass(const std::string& name, int swizzle_log, int split_k = 1) {
  BenchResult r; r.name = name; r.map = mapping_of<Gemm>(swizzle_log, split_k);
  using EA = typename Gemm::ElementA;
  using EB = typename Gemm::ElementB;
  const EA* pA = reinterpret_cast<const EA*>(g_dA);
  const EB* pB = reinterpret_cast<const EB*>(g_dB);

  Gemm gemm_op;
  cutlass::gemm::GemmCoord problem(g_M, g_N, g_K);
  typename Gemm::Arguments args(
      problem,
      {pA, g_K}, {pB, g_N}, {g_dC, g_N}, {g_dC, g_N},
      {ElementAcc(1.f), ElementAcc(0.f)},
      split_k);   // split_k_slices (ignored unless the type has SplitKSerial=true)

  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
    r.implementable = false;   // e.g. tile/warp shape invalid for this problem/arch
    return r;
  }
  r.implementable = true;

  size_t ws = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (ws) CUDA_CHECK(cudaMalloc(&workspace, ws));
  if (gemm_op.initialize(args, workspace) != cutlass::Status::kSuccess) {
    r.implementable = false;
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    return r;
  }

  // ---- correctness (TDD gate): must match reference before we trust its time ----
  CUDA_CHECK(cudaMemset(g_dC, 0, (size_t)g_M*g_N*sizeof(float)));
  if (gemm_op() != cutlass::Status::kSuccess) {
    r.implementable = false;
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    return r;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  r.max_rel = max_rel_error(g_dC, g_ref_host);
  r.correct = r.max_rel < 1e-2;
  if (!r.correct) {                 // do not benchmark an incorrect kernel
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    return r;
  }

  // ---- benchmark: interleave candidate & cuBLAS so both share thermal/power
  //      state; take the median (cuBLAS_ms / cand_ms) over rounds. The RATIO
  //      cancels the slow power-cap drift that would bias a candidate-then-cuBLAS
  //      comparison (cuBLAS timed last on a hot, throttled GPU looks slow).
  auto cand_launch   = [&]{ (void)gemm_op(); };
  auto cublas_launch = [&]{ cublas_gemm<EA>(g_handle, g_dA, g_dB, g_dC, g_M, g_N, g_K); };
  for (int i = 0; i < G_WARMUP; ++i) { cand_launch(); cublas_launch(); }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> cand, cub, ratio;
  for (int rnd = 0; rnd < G_ROUNDS; ++rnd) {
    double c = time_launches(cand_launch, G_ITERS);
    double b = time_launches(cublas_launch, G_ITERS);
    cand.push_back(c); cub.push_back(b); ratio.push_back(b / c);
  }
  r.ms = median(cand);
  r.fair_cublas_ms = median(cub);
  r.fair_speedup = median(ratio);
  r.tflops = (2.0*g_M*g_N*g_K) / (r.ms*1e-3) / 1e12;

  if (workspace) CUDA_CHECK(cudaFree(workspace));
  return r;
}

// Benchmark cuBLAS itself (already validated to be the reference).
template <typename Element>
static BenchResult run_cublas(cublasHandle_t h) {
  BenchResult r; r.name = "cuBLAS"; r.implementable = true; r.correct = true; r.max_rel = 0.0;
  for (int i = 0; i < G_WARMUP; ++i) cublas_gemm<Element>(h, g_dA, g_dB, g_dC, g_M, g_N, g_K);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t beg, end; CUDA_CHECK(cudaEventCreate(&beg)); CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < G_ITERS; ++i) cublas_gemm<Element>(h, g_dA, g_dB, g_dC, g_M, g_N, g_K);
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float total_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&total_ms, beg, end));
  r.ms = total_ms / G_ITERS;
  r.tflops = (2.0*g_M*g_N*g_K) / (r.ms*1e-3) / 1e12;
  CUDA_CHECK(cudaEventDestroy(beg)); CUDA_CHECK(cudaEventDestroy(end));
  return r;
}

// =============================================================================
//  REGISTRY of pre-built mappings (selectable at runtime, no recompile).
//  Each entry pairs a human name with a distinct compile-time CUTLASS type.
//  NOTE: every entry is a full kernel instantiation, and it is instantiated once
//  per dtype, so more entries == slower compile. Trim if build time bothers you.
// =============================================================================
struct Registry {
  std::vector<BenchResult>* out;
  bool wanted(const std::string& n) const {
    if (G_RUN_ALL) return true;
    // G_ONLY is a comma-separated list of substrings; match if ANY is contained.
    size_t start = 0;
    while (start <= G_ONLY.size()) {
      size_t comma = G_ONLY.find(',', start);
      std::string tok = G_ONLY.substr(start, comma==std::string::npos ? std::string::npos : comma-start);
      if (!tok.empty() && n.find(tok) != std::string::npos) return true;
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    return false;
  }
  template <typename Gemm> void add(const std::string& n, int swizzle_log, int split_k = 1) {
    if (!wanted(n)) return;
    if (G_NO_SPLITK && split_k > 1) return;   // apples-to-apples vs a non-split-K estimator
    std::printf("  [%-28s] testing... ", n.c_str()); std::fflush(stdout);
    BenchResult r = run_cutlass<Gemm>(n, swizzle_log, split_k);
    if (!r.implementable)      std::printf("SKIP (not implementable)\n");
    else if (!r.correct)       std::printf("FAIL correctness (max_rel=%.3g)\n", r.max_rel);
    else                       std::printf("ok  %.3f ms  %.1f TFLOP/s  %.2fx cuBLAS%s\n",
                                   r.ms, r.tflops, r.fair_speedup, r.fair_speedup>1.0?"  <== faster":"");
    out->push_back(r);
  }
};

template <typename E>
static void run_all_mappings(std::vector<BenchResult>& results) {
  Registry reg{&results};
  // ---- the user-editable "custom" mapping (from the macros at the top) ----
  reg.add<GemmMappingT<E, CFG_TB_M,CFG_TB_N,CFG_TB_K, CFG_WARP_M,CFG_WARP_N,CFG_WARP_K,
                       CFG_INST_M,CFG_INST_N,CFG_INST_K, CFG_STAGES, CFG_SWIZZLE>>("custom (macros)", CFG_SWIZZLE);

  // ---- vary the TILING dimensions (threadblock / warp tile) ----
  reg.add<GemmMappingT<E, 128,128,32, 64,64,32, 16,8,16, 3, 1>>("tb128x128x32_w64x64_s3", 1);
  reg.add<GemmMappingT<E,  64, 64,32, 32,32,32, 16,8,16, 4, 1>>("tb64x64x32_w32x32_s4", 1);
  reg.add<GemmMappingT<E, 128,256,32, 64,64,32, 16,8,16, 3, 1>>("tb128x256x32_w64x64_s3", 1);
  reg.add<GemmMappingT<E, 256,128,32, 64,64,32, 16,8,16, 3, 1>>("tb256x128x32_w64x64_s3", 1);
  reg.add<GemmMappingT<E, 128,128,64, 64,64,64, 16,8,16, 3, 1>>("tb128x128x64_w64x64_s3", 1);

  // ---- vary the SOFTWARE-PIPELINE stages (== Triton num_stages) ----
  reg.add<GemmMappingT<E, 128,128,32, 64,64,32, 16,8,16, 2, 1>>("tb128x128x32_stages2", 1);
  reg.add<GemmMappingT<E, 128,128,32, 64,64,32, 16,8,16, 4, 1>>("tb128x128x32_stages4", 1);
  reg.add<GemmMappingT<E, 128,128,32, 64,64,32, 16,8,16, 5, 1>>("tb128x128x32_stages5", 1);

  // ---- vary the LOOP ORDER (threadblock rasterization group; log2 of group) ----
  reg.add<GemmMappingT<E, 128,128,32, 64,64,32, 16,8,16, 3, 0>>("tb128x128x32_swizzle_grp1", 0);
  reg.add<GemmMappingT<E, 128,128,32, 64,64,32, 16,8,16, 3, 4>>("tb128x128x32_swizzle_grp16", 4);

  // ---- SPLIT-K (serial, in-place) — the analog of cuBLAS numSplitsK. The last
  //      arg is split_k_slices, a runtime value. "cublas_match" mirrors cuBLAS's
  //      inspected mapping for the skinny shape: tile 128x128x32 + split-K=3.
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 2, 1>>("cublas_match_s2_splitK3", 1, 3);
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 3, 1>>("tb128x128x32_splitK2", 1, 2);
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 3, 1>>("tb128x128x32_splitK3", 1, 3);
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 3, 1>>("tb128x128x32_splitK4", 1, 4);
  reg.add<GemmMappingSplitKT<E,  64, 64,32, 32,32,32, 16,8,16, 4, 1>>("tb64x64x32_splitK3", 1, 3);

  // ---- NEIGHBORHOOD around cuBLAS's optimum (tile 128x128x32, s2, split-K~3):
  //      denser split-K sweep at cuBLAS's stage count, plus nearby tile/warp shapes.
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 2, 1>>("nbr_s2_splitK2", 1, 2);
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 2, 1>>("nbr_s2_splitK4", 1, 4);
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 2, 1>>("nbr_s2_splitK6", 1, 6);
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 2, 1>>("nbr_s2_splitK8", 1, 8);
  reg.add<GemmMappingSplitKT<E, 128, 64,32, 64,32,32, 16,8,16, 2, 1>>("nbr_tb128x64_splitK3", 1, 3);
  reg.add<GemmMappingSplitKT<E,  64,128,32, 32,64,32, 16,8,16, 2, 1>>("nbr_tb64x128_splitK3", 1, 3);
  reg.add<GemmMappingSplitKT<E, 128,128,32, 64,64,32, 16,8,16, 2, 2>>("nbr_s2_splitK3_swz2", 2, 3);

  // ---- 64x64x32 num_stages sweep (num_stages experiment; CUTLASS multistage
  //      floor is 2 stages, so stg1 may be reported not-implementable) ----
  reg.add<GemmMappingT<E, 64,64,32, 32,32,32, 16,8,16, 1, 1>>("s64stg1", 1);
  reg.add<GemmMappingT<E, 64,64,32, 32,32,32, 16,8,16, 2, 1>>("s64stg2", 1);
  reg.add<GemmMappingT<E, 64,64,32, 32,32,32, 16,8,16, 3, 1>>("s64stg3", 1);
  reg.add<GemmMappingT<E, 64,64,32, 32,32,32, 16,8,16, 4, 1>>("s64stg4", 1);
  reg.add<GemmMappingT<E, 64,64,32, 32,32,32, 16,8,16, 5, 1>>("s64stg5", 1);

  // ---- SNOWCAT's estimated-optimal mapping (min-traffic tile from the roofline
  //      estimator, gemm_time_estimator.py --optimal): tile 128x256xBK, no split-K,
  //      C=1. Snowcat picks BK=16, but CUTLASS 2.x needs WarpK>=32 (kWarpGemm-
  //      Iterations must be even), so BK is rounded up to 32 here. Stages=2 is the
  //      CUTLASS floor (snowcat's C=1). Snowcat minimizes traffic, so its optimum
  //      never uses split-K, even though its own latency model rates split-K faster.
  reg.add<GemmMappingT<E, 128,256,32, 64,64,32, 16,8,16, 2, 1>>("snowcat_opt_tb128x256", 1);
}

// =============================================================================
//  Startup oracle: prove cuBLAS (our reference) matches a CPU GEMM on a small
//  problem, so that every later "vs reference" check is trustworthy.
// =============================================================================
template <typename Element>
static void oracle_selftest(cublasHandle_t h) {
  const int M = 96, N = 80, K = 128;
  std::vector<float> A(M*K), B(K*N), Cref(M*N);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> d(-4, 4);
  // Round inputs through the target element type so the CPU ref uses the same bits.
  for (auto& x : A) x = float(Element(0.5f*d(rng)));
  for (auto& x : B) x = float(Element(0.25f*d(rng)));
  cpu_gemm(A, B, Cref, M, N, K);

  std::vector<Element> Ah(M*K), Bh(K*N);
  for (int i=0;i<M*K;++i) Ah[i]=Element(A[i]);
  for (int i=0;i<K*N;++i) Bh[i]=Element(B[i]);
  Element *dA,*dB; float* dC;
  CUDA_CHECK(cudaMalloc(&dA, M*K*sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, Ah.data(), M*K*sizeof(Element), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, Bh.data(), K*N*sizeof(Element), cudaMemcpyHostToDevice));
  cublas_gemm<Element>(h, dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  double rel = max_rel_error(dC, Cref);
  std::printf("[self-test] cuBLAS %s vs CPU reference: max_rel=%.3g  ->  %s\n",
              DtypeTraits<Element>::name, rel, rel < 1e-2 ? "PASS" : "FAIL");
  if (!(rel < 1e-2)) { std::fprintf(stderr, "reference oracle FAILED; aborting.\n"); std::exit(2); }
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
}

static void print_mapping(const MappingInfo& m, const char* indent) {
  std::printf("%sThreadblockShape = %d x %d x %d\n", indent, m.tbM, m.tbN, m.tbK);
  std::printf("%sWarpShape        = %d x %d x %d\n", indent, m.wM,  m.wN,  m.wK);
  std::printf("%sInstructionShape = %d x %d x %d\n", indent, m.iM,  m.iN,  m.iK);
  std::printf("%sStages           = %d   (software-pipeline depth)\n", indent, m.stages);
  std::printf("%sSwizzle log      = %d   (threadblock raster group = 1<<%d)\n", indent, m.swizzle, m.swizzle);
  std::printf("%sSplit-K slices   = %d%s\n", indent, m.splitk, m.splitk>1 ? "   (serial in-place reduction)" : "");
}

// =============================================================================
//  Full benchmark pass for one input dtype (fp16 or bf16).
// =============================================================================
template <typename Element>
static void run_dtype(cublasHandle_t handle) {
  const char* label = DtypeTraits<Element>::name;
  std::printf("\n################################  dtype = %s  ################################\n", label);

  // 1) Prove the reference oracle for this dtype.
  oracle_selftest<Element>(handle);

  // 2) Allocate the problem and build the cuBLAS reference (also the perf baseline).
  std::printf("Allocating & initializing %dx%dx%d (%s) ...\n", g_M, g_N, g_K, label);
  std::vector<Element> hA((size_t)g_M*g_K), hB((size_t)g_K*g_N);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : hA) x = Element(dist(rng));
  for (auto& x : hB) x = Element(dist(rng));
  CUDA_CHECK(cudaMalloc(&g_dA, (size_t)g_M*g_K*sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&g_dB, (size_t)g_K*g_N*sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&g_dC, (size_t)g_M*g_N*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(g_dA, hA.data(), (size_t)g_M*g_K*sizeof(Element), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(g_dB, hB.data(), (size_t)g_K*g_N*sizeof(Element), cudaMemcpyHostToDevice));

  cublas_gemm<Element>(handle, g_dA, g_dB, g_dC, g_M, g_N, g_K);
  CUDA_CHECK(cudaDeviceSynchronize());
  g_ref_host.assign((size_t)g_M*g_N, 0.f);
  CUDA_CHECK(cudaMemcpy(g_ref_host.data(), g_dC, g_ref_host.size()*sizeof(float), cudaMemcpyDeviceToHost));

  // 3) Run + validate + time every mapping.
  std::printf("Running CUTLASS mappings (correctness-gated):\n");
  std::vector<BenchResult> results;
  run_all_mappings<Element>(results);

  // 4) cuBLAS baseline timing.
  std::printf("Running cuBLAS baseline...\n");
  BenchResult cub = run_cublas<Element>(handle);
  std::printf("  [%-28s] %.3f ms  %.1f TFLOP/s\n", cub.name.c_str(), cub.ms, cub.tflops);

  // 5) Summary table. The "vs cuBLAS" column is the median interleaved speedup
  //    (cuBLAS_ms / this_ms measured back-to-back), robust to power-cap drift.
  cub.fair_speedup = 1.0;   // baseline vs itself
  auto speedup = [&](const BenchResult& r){ return r.fair_speedup; };
  std::vector<BenchResult> table;
  for (auto& r : results) if (r.implementable && r.correct) table.push_back(r);
  table.push_back(cub);
  std::sort(table.begin(), table.end(),
            [&](const BenchResult& a, const BenchResult& b){ return speedup(a) > speedup(b); });

  std::printf("\n=========== SUMMARY (%s, M=%d N=%d K=%d, fair interleaved) ===========\n",
              label, g_M, g_N, g_K);
  std::printf("%-30s %10s %10s %9s %8s\n", "mapping", "time(ms)", "TFLOP/s", "vs cuBLAS", "max_rel");
  std::printf("----------------------------------------------------------------\n");
  for (auto& r : table) {
    const char* tag = r.name=="cuBLAS" ? "  <- baseline" : (speedup(r) > 1.0 ? "  <== faster" : "");
    std::printf("%-30s %10.3f %10.1f %8.2fx %9.1e%s\n",
                r.name.c_str(), r.ms, r.tflops, speedup(r), r.max_rel, tag);
  }
  std::printf("================================================================\n");

  // 6) AUTO-TUNE: pick the best correct CUTLASS mapping for this size (by fair
  //    speedup) and print its full schedule.
  const BenchResult* best = nullptr;
  for (auto& r : results)
    if (r.implementable && r.correct && (!best || speedup(r) > speedup(*best))) best = &r;
  if (best) {
    std::printf("\n[auto-tune] best CUTLASS mapping for %dx%dx%d (%s): \"%s\"\n",
                g_M, g_N, g_K, label, best->name.c_str());
    print_mapping(best->map, "            ");
    std::printf("            -> %.3f ms, %.1f TFLOP/s, %.2fx cuBLAS%s\n",
                best->ms, best->tflops, speedup(*best),
                speedup(*best) > 1.0 ? "  (beats cuBLAS)" : "");
  }

  CUDA_CHECK(cudaFree(g_dA)); CUDA_CHECK(cudaFree(g_dB)); CUDA_CHECK(cudaFree(g_dC));
  g_dA = g_dB = nullptr; g_dC = nullptr;
}

// =============================================================================
//  CLI
// =============================================================================
static void usage(const char* prog) {
  std::printf(
    "Usage: %s [options]\n"
    "  --m N --n N --k N     problem size            (default %d %d %d)\n"
    "  --dtype fp16|bf16|both  input precision       (default %s)\n"
    "  --iters N             timed iterations        (default %d)\n"
    "  --warmup N            warmup iterations       (default %d)\n"
    "  --config LIST         run only mappings whose name contains any comma-separated\n"
    "                        substring in LIST (plus cuBLAS)\n"
    "  --rounds N            interleaved A/B rounds for the median  (default %d)\n"
    "  --no-splitk           exclude split-K mappings (auto-tune over non-split-K only;\n"
    "                        apples-to-apples vs a non-split-K estimator)\n"
    "  --help                this message\n"
    "\nTiming is always fair: each candidate is interleaved with cuBLAS and the\n"
    "reported speedup is the median cuBLAS_ms/candidate_ms over --rounds rounds.\n",
    prog, G_M, G_N, G_K, G_DTYPE.c_str(), G_ITERS, G_WARMUP, G_ROUNDS);
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int& v){ if (i+1<argc) v = std::atoi(argv[++i]); };
    if      (a=="--m")      next(G_M);
    else if (a=="--n")      next(G_N);
    else if (a=="--k")      next(G_K);
    else if (a=="--iters")  next(G_ITERS);
    else if (a=="--warmup") next(G_WARMUP);
    else if (a=="--dtype") { if(i+1<argc) G_DTYPE = argv[++i]; }
    else if (a=="--config"){ if(i+1<argc){ G_ONLY=argv[++i]; G_RUN_ALL=false; } }
    else if (a=="--rounds") next(G_ROUNDS);
    else if (a=="--no-splitk") { G_NO_SPLITK = true; }
    else if (a=="--help"){ usage(argv[0]); return 0; }
    else { std::fprintf(stderr,"unknown arg: %s\n", a.c_str()); usage(argv[0]); return 1; }
  }
  g_M = G_M; g_N = G_N; g_K = G_K;

  if (G_DTYPE!="fp16" && G_DTYPE!="bf16" && G_DTYPE!="both") {
    std::fprintf(stderr, "bad --dtype '%s' (want fp16|bf16|both)\n", G_DTYPE.c_str()); return 1;
  }

  int dev; CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("GPU: %s (sm_%d%d)   problem: M=%d N=%d K=%d   dtype=%s   iters=%d warmup=%d\n",
              prop.name, prop.major, prop.minor, g_M, g_N, g_K, G_DTYPE.c_str(), G_ITERS, G_WARMUP);

  cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  g_handle = handle;   // used by the fair interleaved comparison inside run_cutlass
  std::printf("(fair timing: interleaved A/B vs cuBLAS, median of %d rounds)%s\n",
              G_ROUNDS, G_NO_SPLITK ? "  [split-K mappings excluded]" : "");

  if (G_DTYPE=="fp16" || G_DTYPE=="both") run_dtype<cutlass::half_t>(handle);
  if (G_DTYPE=="bf16" || G_DTYPE=="both") run_dtype<cutlass::bfloat16_t>(handle);

  cublasDestroy(handle);
  return 0;
}
