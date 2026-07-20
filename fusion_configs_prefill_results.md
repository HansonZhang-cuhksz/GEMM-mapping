# Exhaustive fusion-config enumeration — PREFILL (GLM-5.2 MoE, H100)

Same 40 configs (ATTN 5 × FFN 4 × RES2 2) as the decode study, but the layer now includes the
prefill attention core. Throughput = tokens/s = T / layer_time, swept over prefill token count
T ≤ 131,072. Estimation (snowcat-roofline). Code: `fusion_configs_prefill.py`; chart:
`build_chart_prefill.py` -> `prefill_throughput.html` (artifact). Verified by an adversarial
workflow (model sound, ranking trustworthy; two accuracy fixes applied).

## Prefill layer = fixed attention prefix + the decode config layer

The 40 configs vary the mla_o→FFN→residual2 part (same as decode). Prefill prepends a **fixed
attention prefix, uniform across all 40 configs** (so it shifts absolute throughput but never the
ranking):
- pre-attention RMSNorm.
- **MLA projections** (compressed, not one KV=16384 GEMM): q_proj `[T, N_HEADS·192, HIDDEN]` +
  kv down-proj `[T, latent=512, HIDDEN]` + kv up-proj `[T, N_HEADS·256, 512]`.
- **causal flash-attention core**: `ops = N_HEADS·T²·(qk 192 + v 256)`, compute-bound at large T
  (verified O(T²): ~16× per 4× token step). This is the O(T²) term that dominates at long context.

tokens/expert = T·top_k/experts = T/32 (large in prefill → FFN GEMMs **compute-bound**, unlike
decode where they were weight-bound).

## Best config: **S3-N4-r2f** (≡ S5-N4-r2f) → **0.748 Mtok/s at T=10,240** — same as decode

Fold **all four vector ops into GEMM epilogues, keep up_gate and down as weight-amortized grouped
GEMMs** (N4 half-width SwiGLU write; r2f). Identical ranking and two families to decode; worst is
S1-N6 (on-chip full-FFN, 0.233). **So the optimal fusion path is robust across decode AND prefill.**

## What's different in prefill

**Throughput is non-monotonic — rises, peaks near 10k tokens, then falls:**

| tokens | best Mtok/s | vs unfused | flash % of layer |
|---:|---:|---:|---:|
| 2,048 | 0.29 | +1.7% | 2% |
| 8,192 | 0.68 | +3.9% | 16% |
| **10,240** | **0.748** (peak) | **+4.6%** (peak) | 22% |
| 32,768 | 0.50 | +2.4% | 47% |
| 131,072 | 0.21 | +1.0% | 78% |

**Fusion is worth less in prefill — peak +4.6% (vs decode's +6.2%), shrinking to +1% at long
context** — and throughput itself collapses at long context. Both are the same cause: the **O(T²)
attention core grows from ~0% of the layer at 512 tokens to ~78% at 128k**, so (1) it dominates
total time (throughput → O(1/T) past the peak) and (2) it dilutes the FFN that fusion targets. The
audit's clean decomposition: benefit-per-unit-FFN-share is the same in both regimes (~0.085); the
FFN's share of the layer falls from 74% (decode) / 96% (T=512) to ~44% at the prefill peak, so
prefill's peak benefit ≈ 6.2% × (44/74) ≈ 3.7–4.6%. Secondary: at prefill scale the FFN GEMMs are
compute-bound (tokens/expert = T/32 large), so the eliminated vector kernels hide under the GEMM math.

## Verification verdict

**Sound; config ranking trustworthy.** The flash O(T²) formula is correct and compute-bound for
T≥2048; the attention prefix is bit-identical across all 40 configs (pure additive constant, no
ranking effect); the non-monotonicity and the shrinking fusion benefit are genuine (flash share
0.3→26→78%); S3-N4-r2f ≡ S5-N4-r2f exactly; reuse of the decode layer is clean (no prefix/body
desync). Two medium accuracy caveats were **fixed**: the peak grid missed the 3·2^k points (added
10240/11264/14336 → true peak 0.748 @ 10,240, vs the coarse-grid 0.718 @ 12,288), and the QKV was
modeled at the full KV=16384 width (switched to MLA-compressed latent → absolute throughput up
~8–15%). Both were ranking-neutral.

## Caveats

Estimation, not measured. Attention head-dim / MLA-latent values are reasonable estimates
(qk=192, v=256, latent=512). KV-cache write and expert-combine not separately costed (O(T),
negligible vs O(T²) flash; uniform across configs). Peak token count is tile-quantization-sensitive.

## Bottom line

Across **both decode and prefill**, the best of all 40 fusion configs is the same: **fold every
vector op into weight-amortized GEMM epilogues; never make the FFN intermediate-resident (F6)**.
But **fusion pays less in prefill** (+4.6% peak vs +6.2%) and least at long context — because in
prefill, time goes to the O(T²) attention, not the FFN the fusion targets.

## Reproduce
```
conda run -n area python fusion_configs_prefill.py --gpu h100-sxm   # 40 configs, T<=131072, ranking
python3 build_chart_prefill.py                                       # -> prefill_throughput.html (artifact)
```
