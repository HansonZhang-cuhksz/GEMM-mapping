# Exhaustive fusion-config enumeration + throughput-vs-batch (GLM-5.2 MoE decode, H100)

All fusion configurations enumerated, throughput = tokens/s swept over batch ≤ 16384, best found.
Estimation (snowcat-roofline). Code: `fusion_configs.py`; chart: `build_chart.py` ->
`fusion_throughput.html` (artifact). Verified by an adversarial workflow (model sound, ranking
trustworthy; one grid-artifact caveat).

## Config space (finer): 40 = ATTN(5) × FFN(4) × RES2(2)

Decode path `mla_o -> +residual1 -> RMSNorm -> {router, up_gate} -> SwiGLU -> down -> +residual2`.
- **ATTN** (residual1 + RMSNorm placement): S1 both separate | S2 res1→mla_o | S3 res1+norm→mla_o |
  S4 norm→up_gate-prologue | S5 res1→mla_o, norm→up_gate-prologue.
- **FFN** (SwiGLU + GEMM structure): N0 all separate | N4 SwiGLU→up_gate epilogue (½-width write) |
  N5 SwiGLU→down prologue (2×-wide read) | N6 on-chip full-FFN (weights re-read).
- **RES2**: r2s standalone | r2f fused into the combine. Router always standalone.

Enumeration verified **complete and valid** — all 40 present/distinct, none pruned. **S3 ≡ S5**
(RMSNorm in mla_o-epilogue vs up_gate-prologue are both HBM-free → identical curves), so 40 configs
collapse to ~30 distinct-cost classes.

## Best config: **S3-N4-r2f** (≡ S5-N4-r2f) → **1.215 Mtok/s at batch 15,360**

Fold **all four vector ops into GEMM epilogues** while keeping **up_gate and down as separate,
weight-amortized grouped GEMMs**; SwiGLU via **N4** (write the half-width activated once, out_factor
0.5) which edges N5 by ~0.2%; residual2 fused (r2f) beats standalone by ~0.7%.
- **+6% over fully unfused** (S1-N0-r2s, 1.145 → 1.215).
- **~4.9× over the best on-chip F6** (capped at 0.248).

## Two families (the chart's headline)

| family | configs | throughput | why |
|---|---|---|---|
| **Amortized** (N0/N4/N5) | 30 | scales to ~1.10–1.22 Mtok/s | weights read once, amortized over the batch |
| **On-chip F6** (N6) | 10 | **capped flat ~0.17–0.25** | holding the FFN intermediate re-reads weights ∝ batch → forfeits amortization |

Within the amortized family the top cluster (S3/S5/S2 × N4/N5, all r2f) is within ~1% — **the exact
vector-op homes barely matter as long as the GEMMs stay amortized.** The decisive choice is
amortized-vs-on-chip, a ~5× gap.

## Caveats (from the audit — none change the winning config)

1. **The winning *batch* is grid/tile-quantization-sensitive** (medium): the throughput-vs-batch
   curve is *jagged*, not smooth — batches of form 3·2^k (1536/3072/6144/12288/15360) hit better
   divisor tiles than pure powers of 2. Coarse grid peaked at B=12288 (1.208); finer grid peaks at
   **B=15360 (1.215)**; at B=16384 mla_o's optimal mapping flips (m0 384→256), DRAM ~doubles, and it
   tips compute→memory-bound (a dip). The **config ranking is robust**; only the exact peak batch/value
   is fragile.
2. **Modeling scope** (low): `layer_time` spans mla_o → router/FFN → residual2 only — the MoE
   expert-combine reduction and the pre-mla_o work (input norm, QKV/MLA core, RoPE, KV reads) are not
   modeled, and are uniform across all 40 configs. So **absolute tokens/s is inflated but the ranking
   is intact**. r2f "folding into the combine" is notional (combine carries no modeled cost).
3. Estimation, not measured. Prior C500 work showed such small analytic wins often don't survive the
   real vendor stack (needs custom CUTLASS fused epilogues).

## Bottom line

Of all 40 configurations, the best is **fold every vector op into the GEMM epilogues, keep up_gate
and down as amortized grouped GEMMs** (S3-N4-r2f) → 1.215 Mtok/s at batch 15,360, +6% over unfused.
The one thing to never do is the aggressive on-chip full-FFN fusion (F6) — it caps throughput ~5×
below by forfeiting weight amortization.

## Reproduce
```
conda run -n area python fusion_configs.py --gpu h100-sxm   # 40 configs, batch<=16384, ranking
python3 build_chart.py                                       # -> fusion_throughput.html (the artifact)
```
