# TODO — GEMM time estimator

## [ ] Add a tile-size compute-efficiency term to the roofline

**Problem.** `peak_tensor_flops` is treated as tile-size-independent, so the compute
roof is a flat ceiling every tile can reach. Combined with the compute
wave-quantization derate (`compute_time / sm_util`, which rewards *more, smaller*
tiles), the free tile search collapses to the smallest MMA-legal tile (16×16) or a
thin 128×16 — whereas real tensor-core efficiency *rises* with tile size (bigger
tiles amortize the MMA pipeline and reuse operands in registers/SMEM). This is why
the estimator's unconstrained optimum ≠ CUTLASS's balanced 64×64.

**Fix (measurement-first, do NOT curve-fit).** Model achieved tensor throughput as a
function of tile shape: `peak_tflops_eff(BM, BN, BK) = peak_tflops * eff(tile)`, where
`eff ∈ (0,1]` rises toward 1 for large tiles and drops for thin/tiny ones.
- Measure `eff` directly: run compute-bound GEMMs (large, high-occupancy) across a
  sweep of tile sizes and record achieved TFLOP/s (the CUTLASS harness already
  reports per-tile TFLOP/s; use a big square GEMM so occupancy≈1 isolates efficiency).
  Fit/tabulate `eff(BM,BN,BK)` from that data, then validate on **held-out** tiles.
- Plausible drivers to encode: warps/CTA and MMA-tiles/CTA (arithmetic-intensity of
  the register tile), K-loop length amortizing prologue/epilogue, min BM,BN≥64 for
  full MMA-N/-M utilization.
- Acceptance: with the term in, the free search (`snowcat_time_search.py`, non-split-K)
  should select ~64×64 (not 16×16 / 128×16), matching CUTLASS `--no-splitk`.

## [x] Loop-order auto-selection (`--order auto`)  — DONE
Estimate under M-N-K and N-M-K, keep the faster (models CUTLASS's N-major
rasterization / L2 B-column reuse). Fixed 64×64 from 0.402 (1.31× over) to 0.262 ms
(0.86×) and corrected the 64×64-vs-128×256 ranking. See `estimate_best_order`.

## [ ] num_stages has no effect on the estimate (stage benefit unmodeled)
[done] num_stages=1 is now rejected and auto-pick floors at MIN_NUM_STAGES=2 (CUTLASS
multistage floor; stage-1 measured -> NaN). The *efficiency-vs-stages* term below is
still open.

The estimator's time is flat across C=2..6 (64x64x32 all give the same ms), because the
saturating-BW model no longer uses `inflight = active_sm*C*W`. So the estimator can't
pick a pipeline depth — it defaults to the smallest feasible C=1, while CUTLASS's
auto-tune uses stages=4 (deeper pipelining overlaps cp.async loads / feeds the tensor
cores better even when latency is nominally hidden). Needs a term that rewards deeper
pipelines up to a point (then penalizes SMEM pressure / occupancy loss). Tiles already
AGREE (both 64x64x32); this is the remaining mapping-parameter disagreement.

MEASURED (64x64x32, fp16, locked clocks, fair vs cuBLAS, stable over 3 runs):
  stages: 1->FAIL(NaN; CUTLASS floor is 2)  2->0.88x  3->0.93x  4->0.935x  5->0.925x
So achieved tensor throughput rises ~6% from stg2 to stg3/4 then dips at 5. Mechanism:
deeper pipeline hides the on-chip global->shared->register load latency so the tensor
cores never stall between MMAs (a COMPUTE-side overlap effect, NOT DRAM BW/latency --
that saturates regardless of stages). The roofline's max(compute,memory) assumes
perfect overlap, so it's blind to this. Fix: compute_efficiency(stages) factor with
this shape (plateau ~3-4, floor at 2, invalid at 1).

## [ ] Count the C write at its real width (fp32), not bytes_per_element
Both estimators charge all three operands at bpe=2, but the kernel writes fp32 C
(4 B/elem). Validation shows this is the largest single error on C-heavy shapes:
short-K256 (4096x4096x256) est/meas = 0.68x, the worst of the 10-shape suite.
Fix: OUT traffic = M*N*output_bytes (4), keep A/B at bpe. Cheap and principled.

## [ ] Wide/tall asymmetry in the snowcat L2 traffic model
Measured wide (1024x8192x2048) and tall (8192x1024x2048) are symmetric (2.510 vs
2.512 ms) but snowcat predicts 1.879 vs 2.427: the M/BM-driven B re-read survives
its L2 model while the mirrored N/BN-driven A re-read is absorbed. One of the two
is wrong (probably both slightly). Check _reuse_distance_bytes/_l2_concurrency
operand asymmetry against ncu dram__bytes.

## [ ] Traffic / L2 model vs real rasterization (related, lower priority)
`--order auto` covers the M-N-K/N-M-K choice, but the L2 reuse-distance model's
absolute DRAM traffic is still only validated indirectly. Measure CUTLASS's actual
DRAM bytes per tile with `ncu` (dram__bytes) and check the snowcat+L2 traffic against
it, rather than trusting the analytic count.

## [ ] Power-cap DVFS note (resolved for now by locking clocks)
On the 35 W cap the achievable SM clock — hence compute roof — is mapping-dependent
(efficient 4096³ → ~2 GHz; skinny few-tile GEMM → lower). Currently sidestepped by
locking the core clock to 1500 MHz (`nvidia-smi -lgc 1500`, `-lmc 5501`). If modeling
the unlocked GPU, `peak_tensor_flops` would need to depend on the mapping's power draw.
