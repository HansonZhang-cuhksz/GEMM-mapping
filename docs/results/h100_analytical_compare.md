# Analytical: do the two estimators still agree on an H100?

**No.** On the RTX 4060 they matched on 9/10 shapes; on an H100 SXM5 spec profile
(989 TFLOP/s dense FP16, 3.35 TB/s HBM3, 132 SMs, 50 MB L2 — `--gpu h100-sxm`,
purely analytical, nothing run on hardware) they diverge on **60% of (shape,tile)
points, by up to 4.65×**, and flip the bottleneck label (min: compute, snowcat:
memory) at 37 of 67 points. Script: `h100_analytical_compare.py`.

## Why the 4060 masked the difference, and the H100 doesn't

1. **Ridge OI**: 4060-locked ridge = 18.4 TF / 170 GB/s ≈ **108 FLOP/B**; H100 ridge
   = 989 TF / 3.35 TB/s ≈ **295 FLOP/B**. A 64×64 tile's intrinsic OI is
   BM·BN/(BM+BN) = 32 FLOP/B, 128×128 → 64: on H100 *every* practical tile sits deep
   on the memory side, so the mapping-dependent traffic term (the snowcat part) is
   the binding constraint. On the 4060 the low compute roof usually bound first.
2. **L2 concurrency scales with SMs**: the reuse working set is (panel × concurrent
   CTAs). At 24 SMs the 4060's L2 (19.2 MB effective) absorbed the re-reads snowcat
   counts; at 132 SMs the same panels × conc exceed the 30 MB effective L2, so the
   re-reads hit HBM and the two traffic models genuinely part ways.

## Key numbers

Fixed-tile divergence (snow/min), worst cases — all "min says compute-bound, snow
says memory-bound":

| shape | 64×64×32 | 128×128×32 | 256×128×32 |
|---|:--:|:--:|:--:|
| square-4k | **4.55×** | 2.32× | 1.23× |
| square-8k | **4.65×** | 2.30× | 1.18× |
| square-16k | 4.64× | 2.33× | 1.17× |
| llm-ffn-up (2048×28672×8192) | 4.59× | 2.28× | 1.16× |
| llm-ffn-down (2048×8192×28672) | 4.52× | **2.91×** | 2.32× |
| wide / tall | 2.71× / 2.99× | 1.52× / 1.00× | 1.04× / 1.00× |

At each model's **own optimum** they still agree on 12/15 shapes (snowcat's search
finds thin panel tiles — 512×64, 64×512 — whose traffic reaches the algorithmic
floor), but diverge where no BM,BN≥64 tile can reach the floor:
**deep-K8192 → 1.72×** and **llm-ffn-down (K=28672) → 1.80×** (large-K shapes: the
non-resident operand must be re-read and 132-SM concurrency blows out L2).

## Implications

- **On H100-class hardware the traffic model is essential**: the optimistic floor
  model would under-predict standard 128×128-tile GEMMs by 2–3× and can't rank
  tiles at all (it rates every tile of a compute-side shape identically), while
  snowcat predicts a 3–4× spread across tiles of the same GEMM.
- Equivalently: **mapping choice matters ~4× on H100 where it mattered ~1.2× on the
  4060** (per the snowcat model).
- The 4060 agreement was therefore a *property of the small GPU* (low ridge, few
  SMs, relatively big L2), not of the models — exactly the suspected mechanism.

## Caveats

Spec-sheet profile; `bw_saturation_sms=26` is an assumption (shared by both models,
so cross-model ratios are insensitive to it); the L2-concurrency model is
unvalidated at 132 SMs; snowcat's wide/tall asymmetry (seen on the 4060) persists
here (2.71× vs 2.99× on symmetric shapes); no H100 measurements were taken.
