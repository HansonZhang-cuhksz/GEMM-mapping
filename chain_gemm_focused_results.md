# 2-GEMM chain fusion — focused sweep (the balanced regime), H100

The 27-shape sweep forced every dim ≥ 2048, so the fused kernel could never hold a useful
row-block and fusion almost always lost. This focused sweep puts the shapes in the
**flash-attention regime** — narrow output `N2` (hold a small `E[m0,N2]` accumulator, stream
the wide intermediate) — so the fusion tradeoff is *tunable* and lands right at the balance.

**Fixed M=8192.** Narrow output `N2 ∈ {128, 256}`, wide intermediate `N1 ∈ {4096, 8192}`
(so `C = M·N1` is 64–128 MB > 30 MB eff-L2 → the unfused always pays an HBM round-trip), and
**`K1` is the balance knob** — weight `B = K1·N1` sweeps from L2-resident to HBM. All times
from the snowcat-roofline estimator (`unfused_time`/`fused_time` in `chain_gemm_fusion.py`).
Code: `chain_gemm_focused.py`. Table: `chain_gemm_focused_table.md`.

## Result: **FUSE 12 / unfuse 12** — a clean 50/50 balance

| N1 | N2 | K1 | B (MiB) | B loc | m0×mt | unfused ms | fused ms | speedup | winner |
|---:|---:|---:|---:|:--:|---|---:|---:|---:|:--:|
| 4096 | 128 | 512 | 4 | L2 | 16×512 | 0.056 | 0.045 | 1.256× | **FUSE** |
| 4096 | 128 | 1024 | 8 | L2 | 16×512 | 0.093 | 0.081 | 1.149× | **FUSE** |
| 4096 | 128 | 2048 | 16 | L2 | 16×512 | 0.164 | 0.152 | 1.079× | **FUSE** |
| 4096 | 128 | 4096 | 32 | HBM | 256×32 | 0.307 | 0.591 | 0.520× | unfuse |
| 4096 | 128 | 8192 | 64 | HBM | 256×32 | 0.594 | 1.164 | 0.510× | unfuse |
| 4096 | 128 | 16384 | 128 | HBM | 256×32 | 1.415 | 2.310 | 0.612× | unfuse |
| 4096 | 256 | 512–2048 | 4–16 | L2 | 16×512 | — | — | 1.02–1.06× | **FUSE** ×3 |
| 4096 | 256 | 4096–8192 | 32–64 | HBM | 256×32 | — | — | 0.60–0.87× | unfuse ×2 |
| 4096 | 256 | 16384 | 128 | HBM | 256×32 | 1.416 | 1.364 | 1.038× | **FUSE** |
| 8192 | 128 | 512–1024 | 8–16 | L2 | 16×512 | — | — | 1.15–1.24× | **FUSE** ×2 |
| 8192 | 128 | 2048–16384 | 32–256 | HBM | 256×32 | — | — | 0.51–0.59× | unfuse ×4 |
| 8192 | 256 | 512–1024 | 8–16 | L2 | 16×512 | — | — | 1.04–1.05× | **FUSE** ×2 |
| 8192 | 256 | 2048–8192 | 32–128 | HBM | 256×32 | — | — | 0.90–0.99× | unfuse ×3 |
| 8192 | 256 | 16384 | 256 | HBM | 256×32 | 2.725 | 2.647 | 1.029× | **FUSE** |

(Full 24 rows in `chain_gemm_focused_table.md`.)

## The crossover is driven by one thing: does weight B fit L2?

For each (N1, N2), fusion **wins while `B = K1·N1 ≤ 30 MB`** and **loses once B spills to HBM**:

- **N1=4096:** B crosses 30 MB at K1≈3840 → FUSE at K1 ≤ 2048, unfuse at K1 ≥ 4096.
- **N1=8192:** B crosses 30 MB at K1≈1920 → FUSE at K1 ≤ 1024, unfuse at K1 ≥ 2048.

When B fits L2 the fused kernel re-reads it `mt×` **from L2 (≈ free)** and pockets the saved
64–128 MB C round-trip → up to **1.26×**. When B spills, the same `mt = 32×` re-read is **from
DRAM** (e.g. 32 × 64 MB = 2 GB) and swamps the C saving → down to **0.51×**. This is the
central chained-GEMM fusion law: *fusion is worth it exactly when the re-read weight is
cache-resident.*

## A non-monotonic twist: fusion re-wins at very large K1 (if the output is wide enough)

At the largest K1=16384, `N2=256` **flips back to FUSE** (1.03–1.04×) even though B is deep in
HBM. Reason: with K1 that large the chain becomes **compute-bound** (GEMM1's `2·M·N1·K1` ops
dominate), so the `mt×` weight re-read is *hidden under the compute roof*, and fusion again
banks the C round-trip. The same K1 with `N2=128` stays a loss (0.59–0.61×) — a 128-wide output
gives only ~64 output tiles for 132 SMs, so the fused kernel is **under-occupied** and its
compute roof rises faster than the saving. So the third regime needs both compute-bound *and*
enough output tiles to fill the GPU.

## Summary — three regimes as K1 grows (fixed narrow N2, wide N1)

| regime | K1 | bottleneck | verdict | why |
|---|---|---|---|---|
| **B in L2** | small | memory | **FUSE** (1.02–1.26×) | re-read free from L2; save the C round-trip |
| **B in HBM** | mid | memory | unfuse (0.51–0.99×) | `mt×` DRAM weight re-read swamps the C saving |
| **compute-bound** | huge | compute | FUSE again (≈1.03×) | re-read hidden under compute; C saving recovered — *iff output wide enough to occupy the GPU* |

The takeaway matches real practice: chained-GEMM fusion (flash-attention, fused MLP) pays off
when the intermediate is narrow enough to hold a big row-block **and** the streamed weights are
L2-resident — or when the kernel is so compute-bound the re-read is free. Push the weights past
L2 in the memory-bound regime and fusion loses.

## Reproduce
```
conda run -n area python chain_gemm_focused.py --out chain_gemm_focused_table.md   # FUSE 12 / unfuse 12
```
