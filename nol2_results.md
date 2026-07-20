# All fusion tests rerun with L2 removed (l2_bytes=0)

"Remove L2 entirely" = every access goes to HBM, consistently for fused AND unfused: the
inter-kernel intermediate always round-trips HBM, the fused kernel's weight re-reads always hit
HBM, and each GEMM also loses its cross-tile operand caching. Implemented by zeroing the GPU's L2
capacity (`dataclasses.replace(g, l2_bytes=0)`) — the estimator then charges all snowcat traffic to
DRAM (verified: a 4096³ GEMM goes 143µs → 491µs, a 3.4× hit that quantifies L2's real value), and
`_eff_l2 → 0` makes the intermediate always spill and every weight `w_big`. No model code changed.
Driver: `rerun_nol2.py`.

## Headline: removing L2 makes fusion WORSE, not better

| test | L2-on | NO-L2 |
|---|---|---|
| chain 27-shape | FUSE **3** / unfuse 19 / infeas 5 | FUSE **0** / unfuse 22 / infeas 5 |
| focused flash-attn (24) | FUSE 12 / unfuse 12 | FUSE **14** / unfuse 10 |
| multi-depth w=128 | fuse-all opt, **4.585×** | fuse-all opt, **2.001×** |
| multi-depth w=256/512/1024 | fuse-all opt (2.3/1.15/1.01×) | fuse-all **NOT opt** (0.69/0.74/0.37×) |
| SMEM N* (seq), w=512/1024/2048 | none (fuse-all always opt) | **N*=2** (fusion fails immediately) |
| square n=1024/2048 | tie / tie | **unfuse 0.38× / 0.18×** |
| square n=4096 | infeasible | infeasible |

## Why — L2's main job here is caching the FUSED kernel's re-read weights

The fused kernel processes M in `mt = M/m0` row-blocks, and **each block re-reads the full
weights**. With L2 those re-reads are cache hits (≈1× DRAM); **without L2 they all hit HBM → mt×
DRAM.** The unfused kernel has no such re-read — it loses only its cross-tile operand caching (a
bounded factor). So the fused kernel leans on L2 *far harder* than the unfused, and removing L2
hurts fusion much more. Concretely (square n=1024, L=3): unfused slows 0.0067→0.0269 ms (4×, lost
tile caching), but fused slows 0.0067→0.0707 ms (**10.5×**, mt× weight re-read now from HBM) → a
tie becomes a 0.38× loss.

The intermediate-round-trip saving that "no L2" *adds* to fusion's ledger is real, but it is
**smaller than the weight-re-read cost that L2 was hiding** — so the net moves against fusion. This
is the clean, consistent settlement of the earlier "does full DRAM give more fuse wins?" question:
**fewer** (3→0 on the 27-shape), because the fused `mt×` weight re-read dominates the saved
intermediate round-trip.

## The two exceptions, and what they teach

- **w=128 still fuses (2.0×) without L2:** the weights are so tiny (32 KB) that even `mt=512×` from
  HBM is cheap, while the intermediate saving is large. So when weights are negligible, fusion
  survives L2 removal.
- **Focused flash-attn regime gains 2 wins (12→14):** there the win is dominated by a large
  intermediate round-trip with small streamed weights; removing L2 slows the *unfused* baseline
  (lost tile caching) a touch faster than the already-HBM-bound fused kernel, flipping two
  borderline cases. It is a marginal, regime-specific shift, not a reversal of the headline.

## The real lesson

**L2 is what makes chain fusion viable — it caches the weights the fused kernel re-reads once per
row-block.** Remove L2 and you don't unlock fusion for small (previously L2-resident) intermediates
— you *break* fusion, because its characteristic cost (the `mt×` weight re-read) is exactly the
thing L2 was absorbing. Fusion depends on L2 more than the split baseline does.

Note the square ties did NOT become wins under L2 removal (they became losses) — so the earlier
"SMEM > L2, an L2-resident intermediate should still fuse well" intuition is only true if the
weight re-read stays cached; with the *whole* cache gone, the re-read cost swamps the SMEM benefit.
The realistic model remains L2-on; this run characterizes L2's contribution by removing it.

## Reproduce
```
conda run -n area python rerun_nol2.py
```
