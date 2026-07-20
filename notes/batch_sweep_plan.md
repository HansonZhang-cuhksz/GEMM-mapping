# Batch-size sweep: unfused vs fused throughput (GLM-5.2 MoE full decode layer, H100)

Hypothesis (user): unfused pass prefers LARGE batch (each kernel has full SMEM, amortizes weights),
fused pass prefers SMALL batch (holds intermediate in SMEM; large batch -> weight re-reads). Find each
pass's optimal batch, compare best throughput. Throughput = tokens/s = B / layer_time.

Model: batch_sweep.py monkeypatches fusion_time_estimator's batch globals; reuses its validated F2
(mla_o+residual+rms) + F6 (on-chip FFN) fused physics. Full layer = mla_o + router + up_gate + down +
residual + rmsnorm + swiglu (unfused) vs F2_fused + router + F6_ffn (fused). MoE: tokens/expert = B/32.

## Result (H100, batch_sweep_h100.json)
| batch | tok/exp | unf Mtok/s | fus Mtok/s | fus/unf |
|---|---|---|---|---|
| 256   | 8   | 0.044 | INFEAS | -- |
| 512   | 16  | 0.086 | 0.086 | 1.006x |
| 1024  | 32  | 0.166 | 0.168 | 1.012x |
| 2048  | 64  | 0.312 | 0.168 | 0.538x |
| 4096  | 128 | 0.558 | 0.168 | 0.301x |
| 8192  | 256 | 0.919 | 0.168 | 0.183x |
| 16384+| 512+| ~1.104 (saturates) | 0.168 (flat) | 0.15x |

UNFUSED optimal: batch 131072 -> 1.104 Mtok/s. FUSED optimal: batch 1024 -> 0.168 Mtok/s.
**Best-throughput ratio fused/unfused = 0.152x** (unfused's best is 6.6x higher).

## Mechanism (the key finding)
Hypothesis on PREFERENCES confirmed, but the throughput CAP is the story:
 - UNFUSED reads each weight ONCE and amortizes over all B tokens -> throughput SCALES with batch
   (0.04 -> 1.10 Mtok/s), saturating when the FFN GEMMs turn compute-bound (B>=16384).
 - FUSED (F6) holds activated[m0,INT] on-chip; m0 <= (SMEM-oh)/(INT*bpe) ~ 52 on H100, but Me=B/32 is
   a power of 2 so m0=32 (largest divisor <=52), mt=Me/32=B/1024. It RE-READS all weights mt x, so
   fused time ∝ B -> throughput is CAPPED at 0.168 Mtok/s (constant) for B>=1024. The on-chip fusion
   FORFEITS weight amortization -> its throughput can't scale.
 - So fused ties unfused only at tiny batch (B<=~1664, Me<=52, mt=1); beyond that unfused pulls away
   6.6x. Fused prefers small batch (before re-reads); unfused prefers large batch (amortization) -- but
   unfused's best wins decisively.

## Steps
1. [x] Build + run batch_sweep.py (H100).
2. [ ] Workflow: verify model + throughput-cap mechanism; refine crossover (Me~52 boundary); explore
   regimes (intermediate width, SMEM size, dense vs MoE) where fused-best could beat unfused-best; synth.
