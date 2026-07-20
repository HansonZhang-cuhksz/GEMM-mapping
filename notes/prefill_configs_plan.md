# Exhaustive fusion-config enumeration — PREFILL (GLM-5.2 MoE, H100)

Same 40 configs (ATTN 5 x FFN 4 x RES2 2), throughput = tokens/s = T / layer_time, sweep prefill
token count T <= 131072. Difference vs decode: ADD the prefill attention core (O(T^2) flash).

## Prefill layer = attention prefix + the decode config layer
The 40 configs vary the mla_o->FFN->res2 part (same as decode). Prefill prepends a FIXED attention
prefix (uniform across all 40 configs -> doesn't change the RELATIVE ranking, but dominates at large T):
 - pre-attention RMSNorm (vector, T*HIDDEN).
 - qkv/latent projection GEMM [T, KV=16384, HIDDEN] (O(T)).
 - flash-attention core (causal): ops = N_HEADS * T^2 * (qk_head_dim + v_head_dim), compute-bound at
   large T; ops/peak. qk=192, v=256, N_HEADS=64 (estimates; noted). O(T^2) -> DOMINATES at long context.
Then the decode layer_time(config): mla_o (+res1/norm epilogue per config) + router + FFN + res2.
tokens/expert = T*top_k/experts = T/32 (large in prefill -> FFN compute-bound, unlike decode weight-bound).

## Expected difference from decode
Throughput vs T: rises (FFN fills / amortizes) then FALLS as attention O(T^2) dominates (throughput =
T/O(T^2) = O(1/T)). Fusion benefit (a % of the FFN, a shrinking fraction of the layer) -> shrinks toward
~0 at long context. Best config still the amortized-all-fused one, but margin over unfused decreases with T.

## Steps
1. [x] Build fusion_configs_prefill.py; 2. [x] Sweep T [512..131072] -> prefill_configs.json.
3. [ ] Chart. 4. [ ] Workflow verify.

## Result (H100 prefill, 40 configs)
BEST = S3-N4-r2f (== S5-N4-r2f) -> 0.7177 Mtok/s @ T=12288. SAME config family as decode (fold all
vector ops, keep FFN GEMMs amortized; N4>N5; r2f). WORST = S1-N6 0.232.
KEY prefill-specific findings:
 - Throughput vs T is NON-MONOTONIC: rises to 0.718 @ T=12288, then FALLS to 0.205 @ 131072, because
   attention O(T^2) dominates (throughput = T/O(T^2) = O(1/T)). flash %: 0%@512 -> 26%@12288 -> 78%@131072.
 - Fusion benefit (best/unf) also non-monotonic: 1.005x@512 -> PEAK ~1.039x@8192 -> 1.010x@131072.
   SMALLER than decode's +6.2%: the O(T^2) attention dilutes the FFN's share AND the FFN is compute-bound
   in prefill (large tokens/expert=T/32), so the vector-op fusion is hidden under compute.
 - Peak throughput 0.718 (prefill) < 1.215 (decode) -- attention adds cost.
Same 40-config ranking/families as decode (amortized scales-then-falls; F6 capped). Best config robust
across BOTH decode and prefill; only the MARGIN differs (prefill +4% peak vs decode +6%).
