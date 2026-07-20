# 2-GEMM chain fusion across 27 shapes — results (H100, snowcat-roofline)

Chain `C[M,N1] = A[M,K1] @ B[K1,N1]`, `E[M,N2] = C[M,N1] @ D[N1,N2]`. A, B, D ∈
{square, tall, wide} (f=4 chained; large axis = 16384, short dims to 256; dims ∈ {256, 1024,
4096, 16384}). **Every GEMM is timed by the estimator** — `unfused_time` calls
`optimal_mapping_by_time`/`estimate_gemm_time` for both GEMMs; `fused_time` calls it per
row-block and reads the per-operand snowcat+L2 DRAM (`l2_breakdown`), then applies the
estimator's occupancy roofline. Code: `chain_gemm_fusion.py`.

Traffic (H100 effective L2 = 30 MB):
- **unfused** C: stays in L2 (0 HBM) if `M·N1·2 ≤ 30 MB`, else round-trips HBM (`2·M·N1`).
- **fused** C: on chip. GEMM2 reduces over the full N1, so each output row-block (`m0 ≤
  SMEM/(min(N1,N2)·2)` rows) needs the **full B and full D** → B, D are re-read once per block
  (`mt = M/m0` blocks) — exactly like flash attention re-reads K,V per query block. A weight
  fitting L2 is re-read from L2 (1× DRAM); otherwise mt× DRAM.

## Verdict counts (of 27): **FUSE 3 / unfuse 19 / infeasible 5**

(A global "no L2 for everyone" mode was tried and removed: a real tiled GEMM reuses its A/B
panels in SMEM/registers regardless of L2, so forcing raw-DRAM re-reads only mis-penalizes the
*unfused* pair — an adversarial review confirmed it biased the comparison toward fusion. DRAM
vs L2 is now decided **per operand by size**, which is what a real cache does.)

## Your question, directly

The intuition "full DRAM → bigger penalty avoided → more fuse wins" is **half right**. Fusion's
*saving* is the unfused C round-trip — and that is real only when **C is large (> 30 MB L2), so
it genuinely goes to full DRAM**. Those large-C cases are exactly where fusion can win. So yes,
C-in-full-DRAM favors fusion — captured here by C's size (see the "C" column: HBM vs L2).

But the fusion also has a *cost* the intuition omits: **the fused kernel re-reads B and D once
per output row-block — `mt = M/m0` times** (because GEMM2 reduces over the full N1, so each block
needs the entire weights; flash attention re-reads K,V per query block the same way). If you
*also* push the weights to full DRAM (a truly "everything DRAM" world), that re-read becomes
`mt × B` and `mt × D` of DRAM — and since `mt` is 64–1024, it dwarfs the single C round-trip.
That is why "everything full DRAM" gives *fewer* wins, not more. Two DRAM breakdowns
(`--verbose`) make it concrete:

- **tall-tall-square** (M=16384, K1=4096, N1=1024, N2=1024) — **FUSE 1.02×**: weights B=8 MiB,
  D=2 MiB both **fit L2 → re-read 1×**; fused DRAM = A 128 + B 8 + D 2 + E 32 ≈ 170 MiB and it
  is compute-bound; unfused pays a C=32 MiB HBM round-trip. Fusion wins.
- **tall-square-square** (M=16384, K1=4096, N1=4096, N2=4096) — **unfuse 0.056×**: B=32 MiB > L2
  → **re-read mt=1024× → 32 TiB** (!) of DRAM; fused is hopelessly memory-bound (20 ms) vs
  unfused 1.1 ms. The weight re-read is the whole story.

Had B's 8 MiB *not* been L2-resident (a truly all-DRAM world), it would be re-read 1024× from
DRAM and the first case would lose too. So the fused weight re-read — not the C round-trip — is
the dominant term, and its being **L2-resident (small enough to fit)** is what lets fusion win
at all. This is the opposite of "full DRAM helps fusion".

## Where fusion wins: **A tall + B tall** (large C, small L2-resident weights)

| A | B | D | M | K1 | N1 | N2 | speedup |
|---|---|---|---:|---:|---:|---:|---:|
| tall | tall | wide | 16384 | 4096 | 1024 | 4096 | 1.030× |
| tall | tall | tall | 16384 | 4096 | 1024 | 256 | 1.024× |
| tall | tall | square | 16384 | 4096 | 1024 | 1024 | 1.024× |

Recipe: **B tall → narrow intermediate N1 (=1024)** → feasible (holds ~16–64 rows) *and* B, D
small enough to be **L2-resident** (no re-read penalty); **A tall → large M (=16384)** → the
intermediate `C = 32 MB > L2` so the *unfused* pays a C round-trip that fusion avoids. Wins are
marginal (~1–3%) because on H100 these GEMMs are compute-bound, so the saved C round-trip is a
small fraction. **Infeasible (5):** both N1, N2 = 16384 (wide intermediate *and* output) — can't
hold 16 rows of a 32 KiB slice.

## Conclusion

Chain fusion helps only when **both** (i) the intermediate is narrow enough to hold a row-block
in SMEM (so `mt` is not huge) and (ii) the re-read weights are **L2-resident** (so the mt× re-read
is free). That is the tall-A + tall-B corner, and even there ~1–3% on H100. When weights spill
L2 (square/wide K1,N2), the fused mt× DRAM re-read dominates and fusion loses badly; when the
intermediate is wide it is infeasible. Full DRAM makes fusion *worse*, not better, because it
un-caches the fused weight re-reads. (This matches real practice: flash attention fuses because
head_dim is narrow *and* K/V are L2-resident.)

## Reproduce
```
conda run -n area python chain_gemm_fusion.py --verbose        # 3 fuse / 19 unfuse / 5 infeasible
```
