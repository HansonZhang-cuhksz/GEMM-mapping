# Fused-vs-unfused kernel time on real GPUs (GLM-5.2 decode, batch 2048)

Real-world fusion decision: on a **fixed real GPU**, is the fused optimal kernel faster than
the unfused optimal kernels it replaces? Estimated analytically with GEMM-mapping's
latency-aware snowcat-roofline model (`gemm_time_estimator.py`, validated to 72–96% of
measured on a real RTX 4060), extended for fused epilogues + the memory-bound vector kernels
(`fusion_time_estimator.py`; see `notes/fusion_time_estimator_plan.md`). No kernel is run.

`unfused = Σ(optimal GEMM times) + Σ(vector-kernel times)`; `fused = optimal fused-kernel
time`. Both use the estimator's own optimal-mapping search (64×64×32 tensor-sensible floor).
FFN GEMMs are grouped over all 256 experts (they fill the GPU together).

## Results

### H100 SXM5 (989 TFLOP/s, 3.35 TB/s, 132 SMs, 227 KiB SMEM/block, ridge OI 295)

| Fusion | unfused ms | fused ms | speedup | verdict |
|---|---:|---:|---:|:--:|
| F1  FlashAttn + residual | 0.5302 | 0.5152 | **1.029×** | FUSE |
| F2  FlashAttn + residual + RMSNorm | 0.5377 | 0.5152 | **1.044×** | FUSE |
| F3  RMSNorm + up_gate | 3.9539 | 3.9464 | 1.002× | FUSE |
| F4  up_gate + activation | 4.0065 | 3.9264 | **1.020×** | FUSE |
| F5  activation + down | 2.0633 | 2.0233 | **1.020×** | FUSE |
| F6  up_gate + activation + down | 6.0097 | **11.6589** | **0.515×** | **SKIP** |

### RTX 4060 Laptop, measured profile (18.4 TFLOP/s, 170 GB/s, 24 SMs, 99 KiB SMEM/block)

| Fusion | unfused ms | fused ms | speedup | verdict |
|---|---:|---:|---:|:--:|
| F1  FlashAttn + residual | 22.8137 | 22.3696 | **1.020×** | FUSE |
| F2  FlashAttn + residual + RMSNorm | 22.9618 | 22.3696 | **1.026×** | FUSE |
| F3  RMSNorm + up_gate | 77.9154 | 77.7673 | 1.002× | FUSE |
| F4  up_gate + activation | 78.9516 | 77.3726 | **1.020×** | FUSE |
| F5  activation + down | 40.6601 | 39.8706 | **1.020×** | FUSE |
| F6  up_gate + activation + down | 118.4274 | **457.1298** | **0.259×** | **SKIP** |

## Findings

1. **F1–F5 are worth fusing on both GPUs** — a modest 0.2–4.4% kernel-time win. The
   mechanism is the same everywhere: the fusion eliminates a small memory-bound kernel
   (residual / RMSNorm / SwiGLU) whose traffic is tiny next to the weight-streaming GEMM, so
   the win is small in %, but it is **always positive** (and the framework should take it —
   the real launch-overhead saving, not modeled here, only adds to it).
   - F1/F2 win a bit more on H100 (0.029–0.044×): there `mla_o` is memory-bound, so removing
     the residual/RMSNorm round-trips matters slightly more. On the 4060 `mla_o` is
     compute-bound (18 TFLOP/s roof), so the eliminated traffic is fully hidden and the win
     is exactly the removed vector kernels (0.020–0.026×).

2. **F6 (full FFN fusion) is NOT worth it on real GPUs — it is 2–4× SLOWER.** This is the
   headline, and it **reverses the area study**, where F6 was the *strongest* fusion (−384
   MiB, whole intermediate eliminated). The reason is SMEM:
   - The down GEMM contracts over K = INTERMEDIATE = 2048, so the fused kernel must hold the
     full `activated[m0, :INTERMEDIATE]` row resident (`m0·2048·2 B`). Real SMEM/block is
     only **99 KiB (4060) / 227 KiB (H100)**, so the row-block is capped at **m0 = 16 (4060)
     / 32 (H100)**, versus M = 64 tokens/expert.
   - With `m0 < M`, the kernel processes `mt = M/m0 = 4 (4060) / 2 (H100)` row-blocks and
     **re-reads both weight matrices `mt×`** → weight traffic ×4 / ×2 → memory-bound time
     ×4 / ×2. Hence 0.259× and 0.515×.
   - The area study didn't see this because its hypothetical single-SM chip had ~1.3 MiB of
     SMEM (≈10× a real GPU's per-block budget), enough to hold `m0 = 64` and read weights
     once. **Real per-block SMEM is the binding constraint that kills F6.** (This is exactly
     why `ffn_fused_area_latency.py` and real MoE kernels fuse up_gate+SwiGLU and leave down
     standard.)

3. **Epilogue still beats prologue, on real hardware too.** F4 (SwiGLU into up_gate epilogue,
   writes the half-width activated output) and F5 (SwiGLU into down prologue, reads the
   2×-wide gate+up) both net +2.0%, but F4 does it by *reducing* output traffic while F5
   *adds* input traffic; F4 is the cleaner win and never risks going negative.

## Caveats

- **Analytical, not measured.** The base GEMM estimator is validated to 72–96% of measured
  RTX 4060; the fused numbers reuse it plus first-principles traffic deltas. The **fused/
  unfused ratio** (the verdict) is more robust than the absolute ms, since both sides share
  the estimator and its biases.
- **H100 is a spec-sheet profile** (nothing measured); RTX4060-measured is the calibrated one.
- Kernel **launch overhead is not modeled** — it only strengthens every FUSE verdict.
- Tile search uses the 64×64×32 tensor-sensible floor to avoid the estimator's tiny-tile
  collapse (TODO.md); the F6 SMEM overhead (~16 KiB streaming) is an estimate — but F6 loses
  by 2–4×, far outside any overhead uncertainty.

## Reproduce
```
conda run -n area python fusion_time_estimator.py --verbose        # both GPUs (area env: py3.11+)
conda run -n area python fusion_time_estimator.py --gpu h100-sxm
```
