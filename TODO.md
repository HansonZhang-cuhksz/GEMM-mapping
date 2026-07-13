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
The estimator's time is flat across C=1..6 (64x64x32 all give 0.262 ms), because the
saturating-BW model no longer uses `inflight = active_sm*C*W`. So the estimator can't
pick a pipeline depth — it defaults to the smallest feasible C=1, while CUTLASS's
auto-tune uses stages=4 (deeper pipelining overlaps cp.async loads / feeds the tensor
cores better even when latency is nominally hidden). Needs a term that rewards deeper
pipelines up to a point (then penalizes SMEM pressure / occupancy loss). Tiles already
AGREE (both 64x64x32); this is the remaining mapping-parameter disagreement.

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
