# Exhaustive fusion-configuration enumeration + throughput-vs-batch (GLM-5.2 MoE decode, H100)

Goal: enumerate ALL fusion configs (finer space), sweep batch <= 16384, throughput = tokens/s = B/time,
plot throughput-vs-batch per config, find the best config. Estimation (snowcat-roofline, H100 model).

## Config space (finer, user-chosen)
Decode main path: mla_o -> +residual1 -> RMSNorm -> {router, up_gate} -> SwiGLU -> down -> +residual2.
Fusable vector ops and homes:
 - ATTN state (residual1 + RMSNorm placement), 5 states:
   S1 res1 sep, norm sep | S2 res1->mla_o(F1), norm sep | S3 res1+norm->mla_o(F2) |
   S4 res1 sep, norm->up_gate-prologue(F3) | S5 res1->mla_o(F1), norm->up_gate-prologue(F3)
 - FFN state (SwiGLU + GEMM structure), 4 states:
   N0 up_gate + swiglu + down (all separate) | N4 up_gate+SwiGLU(F4, half-width out) + down |
   N5 up_gate + SwiGLU+down(F5, 2x-wide in) | N6 on-chip up_gate+SwiGLU+down (F6, weights re-read)
 - RES2 (post-FFN residual): {separate | fused into down/F6 epilogue} = 2
=> 5 x 4 x 2 = 40 configs. router always standalone (also consumes the norm). F6 infeasible at some batch.

Modeling: reuse fusion_time_estimator (estimate_gemm_grouped / estimate_fused_gemm / estimate_ffn_fused /
estimate_vector_kernel) via batch_sweep.set_batch(B). Epilogue deltas: F1 res read + aux tile; F2 +rms;
F3 norm-prologue (aux m0*4, input already read); F4 out_factor=0.5; F5 a_factor=2.0; res2 like res1.

## Steps
1. [x] Build fusion_configs.py (40 = 5x4x2), test.
2. [x] Sweep batch [128..16384] -> fusion_configs.json. (Fixed res2 bug: it's a post-COMBINE residual,
   not per-expert; was over-charged x256 into the grouped down. Now handled once at layer level:
   standalone 3xB*HIDDEN vs fused 1x.)
3. [ ] Chart throughput-vs-batch (artifact). 4. [ ] Workflow verify.

## Result (H100, 40 configs, batch<=16384)
BEST = S3-N4-r2f -> 1.2076 Mtok/s @ B=12288. Top 6 within ~1% (S3/S5/S2 x N4/N5, all r2f): fold ALL
four vector ops (residual1, RMSNorm, SwiGLU, residual2) into GEMM epilogues while keeping up_gate+down
as separate weight-AMORTIZED grouped GEMMs; N4 (F4 half-width SwiGLU write) slightly > N5 (F5 2x-wide
read); r2f > r2s by ~1%; norm home S3(mla_o-epi)==S5(up_gate-pro) equivalent. WORST = S1-N6-r2s 0.247
(on-chip F6, no vector fusion) -- ~5x worse. Two families: amortization-preserving (N0/N4/N5, scale to
~1.2) vs on-chip F6 (N6, capped ~0.17-0.25). The best is the maximal epilogue-only fusion.
