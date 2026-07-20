# Physical MetaX C500 measurement campaign — plan & log

Goal: run the fusion analyses (so far all analytical snowcat-roofline estimates) on a PHYSICAL
MetaX C500, same parameters as the estimations, and compare measured vs estimated.

## Hardware / stack discovered
 - 4x MetaX C500, 64 GB each, 350W, MACA 3.7.0 stack (/opt/maca*), driver 3.8.23. mx-smi = smi.
 - conda env 'fusion': torch 2.8.0+metax3.7.1.3, torch.cuda maps to MACA, 4 devices visible.
 - Device props (torch.cuda.get_device_properties): 104 SMs, warp=64, 2048 threads/SM,
   L2 = 8 MiB, SMEM/block = 64 KiB (optin 64 KiB), SMEM/SM = 64 KiB, mem = 63.6 GiB.
   -> vs H100 profile: far smaller L2 (8 vs 50 MiB) and SMEM/block (64 vs 227 KiB). 104 vs 132 SM.
 - STILL TO MEASURE: peak BF16 tensor TFLOP/s, HBM bandwidth, (latency for the roofline).

## Feasibility of "the corresponding analysis" on real silicon
 - Single GEMM: directly runnable (torch.matmul bf16) -> measure, validate the estimator (analog of
   the RTX4060 72-96% validation). FOUNDATION.
 - Unfused chain (2-GEMM, multi-GEMM): natural torch execution materialises intermediates in HBM ->
   directly measurable = the unfused path.
 - TRUE fused chain kernel (intermediate kept on-chip across GEMMs): needs a custom MACA kernel, NOT
   feasible to author here. So the fused side stays estimator-predicted; we compare the estimator's
   fused prediction against the MEASURED unfused to test whether the fuse/unfuse verdict survives on
   real hardware, and validate the single-GEMM estimator that underpins all fused predictions.

## Steps
1. [x] Discover HW + env (mx-smi, torch device props).
2. [ ] Measure peak BF16 TFLOP/s + HBM BW + sanity GEMM timings (build C500 GpuModel).
3. [ ] Measure single-GEMM times across the study dimension sets; compare to estimator -> validation.
4. [ ] Measure unfused chain times (2-GEMM + multi-GEMM); compare to estimator's unfused.
5. [ ] Report: does the estimator hold on C500, and do the fuse/unfuse verdicts survive?

## Steps 2-4 results [done]
C500 model built (metax_c500_model.py): peak 226 TF/s, BW 1.43 TB/s, 104 SM, L2 8 MiB, SMEM 64 KiB,
1125 MHz, latency 400 ns (est), ridge OI 158. Measured 32 GEMMs + 9 chains (metax_measure.py ->
metax_measured.json). Compared to estimator (metax_compare.py):

Single-GEMM est/meas: geomean 1.074, median 1.160, 55% within 1.5x, 79% within 2x. TWO systematic
biases:
 - OVER-predicts large-K / compute GEMMs 2.5-2.8x (4096x4096x16384=2.51, mla_o=2.58,
   8192x4096x16384=2.77, 16384^3=1.77). Real C500 hits ~207-228 TF/s (compute-bound near peak) but
   the estimator predicts memory-bound — its 8 MiB-L2 traffic model is too pessimistic (charges
   cross-tile re-reads to DRAM that the real cache/prefetch hides). => l2 model / alpha miscalibrated.
 - UNDER-predicts small/narrow GEMMs (0.42-0.65x): square 1024=0.42, glm down=0.43, up_gate=0.53,
   multi_stage w128=0.52. No launch-overhead / fixed-cost term -> tiny GEMMs predicted too fast.
Chains echo it: narrow multi_w128/256 chains under-predicted (0.56-0.60x); 2-GEMM chain 0.98,
square_n2048 0.90 (close).

Takeaway: order-of-magnitude right, but uncalibrated on C500. RELATIVE fused-vs-unfused verdicts are
more robust than absolute times (shared bias). C500's 64 KiB SMEM (< H100 227) makes fusion even more
SMEM-feasibility-constrained. Next: rerun fusion studies w/ C500 model; verify; synthesize.

## Step 5 — rerun fusion studies on C500 + verify [done, workflow w4xhnt0d0]
Verified (audit: methodology + calibration sound; latency/bw_sat guesses provably immaterial).
 - Fusion verdicts C500 vs H100: 27-shape 3/19/5 -> 1/5/21; focused 12/12/0 -> 6/18/0; multi L=6
   speedup w128 4.59->2.46x, w256 2.30->1.23x, w512 1.16->1.00x, w1024 INFEASIBLE; N* 12/6 -> 2.
   Cause: 64 vs 227 KiB SMEM (held-slice cap 4.4x tighter) + 8 vs 50 MiB L2 (weight-resident band 6x
   smaller) + lower 226 TF/s peak (compresses speedups to compute floor).
 - CORRECTION (audit, headline): over-prediction is NOT the 8 MiB L2 model -- it's the SMEM-coupled
   tile cap (64 KiB + C>=2 forces 128x64x32 -> OI below 158 ridge -> compute GEMMs mislabeled
   memory-bound; hits 8192^3, 16384^3 too). Real fix: decouple register output tile from SMEM stage.
 - Calibration (measured set): eff-L2 32 MiB + 20 us launch overhead -> geomean 0.99, 100% within
   1.5x. Kept out of the model file (8 MiB L2 is real; the bump is a numerical compensation).
 - Trust: FUSE narrow multi_w128 (2.46x) TRUSTED; square chains INFEASIBLE (structural, most robust);
   chain2 ~1.55x LEAN; multi_w256 1.23x within error; multi_w512 tie NOT trustworthy.
 - Gap: fused side never measured (no custom MACA fused kernel); unfused measured + validated.
Results -> metax_c500_results.md.

## Step 6 — GLM-5.2 decode 6-fusion on C500 (measured) [done]
Audit forced closing the gap: measured F3 + F5 too (not just F1/F4). F3_unfused(eager) 11.82 ->
compiled 9.82 (but that only rescues the atrocious eager rmsnorm; still 0.5ms > bare up_gate 9.31 ->
no true prologue fusion). F5_unfused 4.874 -> compiled 4.837 (0.8%, negligible; ~5% ceiling of
skipping the activated round-trip unrealized). F1 addmm 2.008 even with preallocated out (not just the
D2D copy -> beta path itself ~70us > mm). F6 softened: not a single-block SMEM fusion (activated
64x2048=64KiB = whole budget) AND weight floor 12.9ms ~ no benefit. => on real C500 NO fusion
(F1/F3/F4/F5) delivers a win through cuBLAS/torch.compile; FFN is 84% weight-bound (~4% ceiling); the
vendor stack can't fuse elementwise into grouped GEMM -> needs custom CUTLASS. Results ->
metax_glm_results.md.
(old:)
Estimator on C500 model: F1 0.824x SKIP, F2 0.827x SKIP, F3 1.002 FUSE, F4 1.020 FUSE, F5 1.020 FUSE,
F6 INFEASIBLE. (vs H100: F1 1.029 FUSE, F2 1.044 FUSE, F3-5 same, F6 0.515 skip.) F1/F2 flip to skip
is SUSPECT — driven by the mla_o (2048x6144x16384) over-prediction bias (2.58x, tile-cap).

Measured GLM kernels (metax_glm_measure.py -> metax_glm.json), C500 ms:
 mla_o 1.935, router 0.070, up_gate_grouped(256) 9.308, down_grouped(256) 4.628,
 residual 0.063, rmsnorm 0.373, swiglu 0.256. Unfused decode layer ~16.6 ms; FFN GEMMs
 (up_gate+down=13.9ms) = 84% -- weight-bandwidth-bound (up_gate W=12.8GB/1.43TB/s=8.95ms ~ measured).
Real fusions via the cuBLAS/compiler stack:
 - F1 fused via cuBLAS addmm (residual + mla_o in one beta-accumulate GEMM): 2.001 ms vs unfused
   (mm+add) 1.982 ms -> ~1% SLOWER. Fusion does NOT help (mla_o compute-bound, the separate add is
   already ~free, and the addmm beta path is slightly less efficient than plain mm 1.935).
 - F4 via torch.compile (bmm+swiglu): 9.607 vs unfused 9.549 -> no benefit (compiler didn't fuse).
=> Real C500: the small analytic F1-F5 wins (0.2-4.4%) do NOT materialize through addmm/torch.compile;
   F6 infeasible (physical, 64KiB SMEM). To VERIFY the surprises (addmm really fused? compile fused?
   grouped GEMM weight-bound?) + build the full C500-vs-H100 table -> workflow.

## Timing methodology
bf16 inputs; warmup 10-30 iters; torch.cuda.synchronize(); median of N timed iters via
torch.cuda.Event(enable_timing=True) or perf_counter+sync; fixed device 0; report GFLOP/s + ms.
