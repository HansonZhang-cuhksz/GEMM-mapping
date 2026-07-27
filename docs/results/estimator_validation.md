# Estimator validation: snowcat-roofline vs optimistic (min-traffic) vs measured

Setup: locked clocks (core 1500 MHz, VRAM 5501 MHz), fp16, `rtx4060-measured` profile.
Measured = best correct **non-split-K** CUTLASS mapping from the fair interleaved sweep
(`./gemm --no-splitk`, median of 5 rounds); estimators searched the same space
(BM,BN ≥ 64, BK ≥ 32, split_k=1). Raw data: `estimator_validation.csv`;
driver: `validate_estimators.py`.

## Accuracy (estimate ÷ measured, on the measured-best tile)

| pattern | M×N×K | measured (tile, stages) | snowcat | min-traffic |
|---|---|---|:--:|:--:|
| square-small | 512³ | 0.021 ms (64×64×32 s3) | 0.78× | 0.78× |
| square-mid | 2048³ | 0.967 ms (64×64×32 s2) | **0.97×** | **0.97×** |
| square-large | 4096³ | 8.680 ms (128×128×32 s3) | 0.87× | 0.87× |
| skinny-M128 | 128×4096×4096 | 0.362 ms (64×64×32 s4) | 0.72× | 0.72× |
| skinny-M64 | 64×4096×4096 | 0.188 ms (64×64×32 s4) | 1.08× | 1.08× |
| skinny-N128 | 4096×128×4096 | 0.361 ms (64×64×32 s4) | 0.73× | 0.73× |
| short-K256 | 4096×4096×256 | 0.682 ms (64×64×32 s3) | 0.68× | 0.68× |
| deep-K8192 | 512×512×8192 | 0.365 ms (64×64×32 s4) | 0.72× | 0.72× |
| wide | 1024×8192×2048 | 2.510 ms (64×64×32 s3) | 0.75× | 0.75× |
| tall | 8192×1024×2048 | 2.512 ms (64×64×32 s3) | **0.97×** | 0.75× |

**Aggregate**: geomean est/meas — snowcat **0.82**, min-traffic **0.80**;
mean |log error| — snowcat ~24%, min-traffic ~28%.

## Which is more accurate?

**They are numerically identical on 9 of 10 shapes.** On the tiles that matter
(64×64/128×128 with the best loop order), snowcat's traffic + L2 reuse model already
collapses to the algorithmic minimum — or the compute roof binds and masks traffic —
so removing snowcat changes nothing there. The single divergence is **tall**
(8192×1024×2048): snowcat 2.427 ms (0.97×) vs min 1.879 ms (0.75×). Snowcat wins it,
BUT with a caveat: measured *wide* (2.510) and *tall* (2.512) are symmetric, while
snowcat predicts them asymmetric (1.879 / 2.427) — its extra traffic term fires for
`M/BM` re-reads but the mirrored `N/BN` case is absorbed by its L2 model, so the tall
"win" is half luck. Verdict: **snowcat ≥ min everywhere (never worse, once better);
the min-traffic model is a surprisingly tight, much simpler floor.**

## Systematic error pattern (both estimators)

- **Optimistic by ~25–30% on memory-heavy shapes** (skinny, short-K, deep-K, wide):
  the roofline assumes perfect compute/memory overlap + peak-achievable BW.
- **Worst case short-K256 (0.68×)**: both models count the C write at 2 B/elem, but
  the real kernel writes **fp32 C (4 B)**; on short-K shapes C dominates traffic.
  (Known simplification — worth fixing: count OUT at accum/output width.)
- **square-small (0.78×)**: per-launch overhead (~µs) unmodeled; irrelevant ≥0.1 ms.
- **skinny-M64 over-estimate (1.08×)**: the real GEMM achieved ~184 GB/s, above the
  170 GB/s read-stream microbench — streaming B once with tiny A/C beats the
  synthetic read kernel, so the BW constant is slightly conservative.
- **square-mid is the sweet spot (0.97×)**: big enough to hide launch overhead,
  compute-bound so traffic errors are masked, occupancy ≈ 1.

## Mapping-selection log

| pattern | CUTLASS best | snowcat --optimal | min --optimal | cuBLAS's own algo |
|---|---|---|---|---|
| square-small | 64×64×32 s3 | 64×64×32 | 64×64×32 | 128×64 stg32×6 |
| square-mid | 64×64×32 s2 | 64×64×32 | 64×64×32 | 128×128 stg32×1 |
| square-large | 128×128×32 s3 | 128×64×32 | 64×64×32 | 128×128 stg32×1 |
| skinny-M128 | 64×64×32 s4 | 64×64×32 | 64×64×32 | 128×128 **splitK3** |
| skinny-M64 | 64×64×32 s4 | 64×64×32 | 64×64×32 | 256×64 **splitK3** |
| skinny-N128 | 64×64×32 s4 | 64×64×32 | 64×64×32 | 128×128 **splitK3** |
| short-K256 | 64×64×32 s3 | 64×64×32 | 64×64×32 | 128×128 stg32×1 |
| deep-K8192 | 64×64×32 s4 | 64×64×32 | 64×64×32 | 128×128 **splitK10** |
| wide | 64×64×32 s3 | 64×64×32 | 64×64×32 | 128×128 stg32×1 |
| tall | 64×64×32 s3 | 64×128×32 | 64×64×32 | 128×128 stg32×1 |

- **Tile agreement with CUTLASS best: min 9/10, snowcat 8/10** (both misses are
  near-ties among tiles the model rates within ~1%).
- CUTLASS's measured best is **64×64×32 in 9/10 shapes** on this 24-SM GPU (occupancy
  rules); only the fully compute-bound 4096³ prefers 128×128.
- **stages**: CUTLASS best varies s2–s4; both estimators always report stages=2
  (their time is stage-flat — the known unmodeled ~5% stage effect, see TODO.md).
- **cuBLAS thinks differently**: always big 128×128-class tiles, compensating with
  split-K on low-occupancy shapes (3× on skinny, **10× on deep-K**) — consistent with
  everything we inspected earlier.

## Caveat

At 4096³ the sustained-tensor measurement dropped vs the earlier back-to-back probe
(cuBLAS 9.75 ms here vs 7.9 ms) — under a long sustained tensor load the 35 W cap can
pull the clock below the 1500 MHz lock. Affects only the largest compute-bound shape.
