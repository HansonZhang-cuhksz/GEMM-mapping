# Chained-GEMM (2-GEMM) fusion study — plan & log

Question: for a 2-GEMM chain `C[M,N1] = A[M,K1] @ B[K1,N1]`, `E[M,N2] = C[M,N1] @ D[N1,N2]`,
across GEMM shapes, is **fusing** (keep C on chip, never touch HBM) faster than the
**unfused** pair (GEMM1 writes C to HBM, GEMM2 reads it back)? On **H100-SXM**, using
GEMM-mapping's latency-aware snowcat-roofline estimator. Sweep each of A, B, D over
{square, tall, wide} = **27 cases**.

## Shape parametrization (confirmed: aspect factor f=4, chained; centered)

The three matrices share dims: `A[M,K1]`, `B[K1,N1]`, `D[N1,N2]` — cols of one = rows of the
next. So aspects chain: fix relative `M`, then each next dim = prev × {square:×1, wide:×f,
tall:×1/f} (wide = more cols than rows; tall = more rows than cols). f=4.

  rel: M=1 ; K1 = M·a(A) ; N1 = K1·a(B) ; N2 = N1·a(D)   where a ∈ {1, 4, 1/4}

The literal "M=32768, f=4" compounds to dims up to 32768·4³ = 2.1M (matrices ~137 GB,
> HBM — unrealizable). So each case is **centered**: scale all four dims uniformly so the
**minimum dim = 2048**. Result: every dim ∈ {2048, 8192, 32768, 131072} (all 2^k, all
tile-unlimited); biggest matrix ~8.6 GB. Shapes/ratios preserved, only absolute scale
normalized per case. (e.g. all-wide → M,K1,N1,N2 = 2048, 8192, 32768, 131072;
all-tall → 131072, 32768, 8192, 2048; square-square-square → 2048².)

## Fusion model (confirmed: C on chip; B/D re-reads = full DRAM, no L2)

Both fused and unfused use the estimator with the **L2 model OFF** (`l2=False`, raw snowcat
traffic = "L2 always miss") — the pessimistic/consistent reading of "re-reads always full
DRAM".

**Unfused** = `time(GEMM1: A@B→C) + time(GEMM2: C@D→E)`, each via the estimator's optimal
mapping (min estimated time over tensor-sensible tiles). The intermediate C round-trips HBM:
GEMM1 writes it (its OUT), GEMM2 reads it (its A). Counted automatically.

**Fused** = one kernel, C never in HBM. The chain reduces over N1 (GEMM2's contraction), so a
row-block of `m0` rows must hold the intermediate/output slice resident. A smart kernel holds
whichever of the C-row (`m0·N1`) or the E-accumulator (`m0·N2`) is smaller, streaming the
other — so `resident = m0 · min(N1,N2) · bpe` (+ ~16 KiB tiles). `m0 ≤ SMEM/block / (min(N1,N2)
·bpe)`; if that is < the 16-row MMA minimum the fusion is **INFEASIBLE**. With `mt = M/m0`
row-blocks, each block reads full B and full D once → **B, D re-read mt× (full DRAM)**:

  fused DRAM = M·K1·bpe (A once) + mt·K1·N1·bpe (B) + mt·N1·N2·bpe (D) + M·N2·bpe (E once)
  fused ops  = 2·M·K1·N1 + 2·M·N1·N2
  time       = occupancy-aware roofline(ops, DRAM, tiles, m0-block)

Fusion helps iff it saves more than it costs:
  saved (C round-trip) = 2·M·N1·bpe   >   penalty (extra weight reads) = (mt−1)·(K1·N1 + N1·N2)·bpe

## Expected drivers
- **min(N1,N2) small** → large m0 → few blocks → few re-reads → fusion feasible & good.
  Large min(N1,N2) → m0 tiny/infeasible. The *narrow-intermediate* condition.
- **M large** → big C round-trip saved → favors fusion.
- **K1, N2 large** (big B, D weights) → big re-read penalty → hurts fusion.

## Steps
1. [x] Confirm parametrization (f=4, centered) + fusion model (C on-chip, DRAM re-reads).
2. [x] Implement `chain_gemm_fusion.py` (dims, fused, unfused, 27-case sweep, report).
3. [x] Run on h100-sxm; tabulate. Report -> `chain_gemm_fusion_results.md`.
4. [x] Write results report.

## CORRECTION (user review round 2)

Two real defects the user caught:
1. **The code did NOT use the snowcat-roofline estimator.** The prior version hand-rolled a
   `_roofline` and hand-computed traffic, never calling `estimate_gemm_time`/
   `optimal_mapping_by_time`. Rewritten: unfused = two `optimal_mapping_by_time` GEMMs; fused =
   per-row-block `optimal_mapping_by_time` sub-GEMMs whose per-operand snowcat+L2 DRAM
   (`l2_breakdown`) is aggregated, then the estimator's occupancy roofline (copied exactly).
2. **The "full DRAM vs L2" axis was mislabeled.** The user's "full DRAM" = the *unfused* C
   round-tripping HBM (expensive unfused). I had applied it to the *fused* weight re-reads.
   Fixed to a single realistic L2-aware model + a `--no-l2` bound.

**Verification workflow (8 agents) confirmed 1 real defect:** the `--no-l2` mode was
apples-to-oranges — the fused sub-GEMMs were forced to `l2=True` (a real tiled GEMM keeps
panels in SMEM), but the unfused used `l2=False`, making the unfused unphysically slow (16384^3
GEMM = 30.9 ms vs ~9 ms compute roof) and biasing `--no-l2` TOWARD fusion (its lone win was an
artifact; consistent no-L2 = FUSE 0). **Fix:** removed the unphysical global `--no-l2` toggle;
DRAM-vs-L2 is now purely size-dependent (a real cache), applied consistently to both sides.
The L2-on tally (FUSE 3) was always consistent and unaffected.

**Answer:** fusion's saving (unfused C round-trip) is real only when C is large (> 30 MB L2, so
genuinely in full DRAM) — that part matches the user's intuition. But the fused kernel re-reads
B, D `mt = M/m0` times per output row-block (flash-attn-style); that dominates unless the
weights are small enough to stay L2-resident. So fusion wins only where **C is large (DRAM,
avoided) AND weights are small (L2, cheap re-read)** = tall-A + tall-B, ~1-3%. "Everything full
DRAM" makes fusion WORSE (un-caches the mt x weight re-read), not better.

## Log / result
- First cut set the UNFUSED baseline to L2-off too, which made everything memory-bound with
  artificial re-reads (uniform ~0.38x). Fixed: **unfused = realistic L2-on** (its intra-kernel
  re-reads cache; C still round-trips HBM between the two kernels), which is the correct
  "pessimistic-for-fusion" framing. Fused keeps the chosen full-DRAM re-read model, plus an
  L2-cached-re-read complement.
- **Key structural fact (drives everything):** GEMM2 contracts over N1, so the fused kernel
  must materialize the intermediate/output slice for a row-block. Holding `m0 * min(N1,N2)`
  in 227 KiB SMEM caps `m0 <= ~50` when min(N1,N2)=2048 (the floor). So `mt = M/m0 >= M/50`
  row-blocks -> the weights B, D are re-read ~M/50 times. INFEASIBLE when min(N1,N2) >= 8192
  (can't hold even 16 rows).
- **Result (H100):** full-DRAM re-reads -> **FUSE 0** / skip 18 / infeasible 9 (fusion never
  wins; 4-9x slower). L2-cached re-reads -> **FUSE 1** / skip 17 / infeasible 9, the lone win
  (tall-square-square, 1.016x) marginal & compute-bound. So with **all dims large (>=2048),
  2-GEMM chain fusion is essentially not useful on H100**: the intermediate can't be held in
  large enough blocks, so weights are re-read. Fusion of chained GEMMs needs a *narrow* shared
  /output dim (e.g. attention's head_dim=128) — which the "large dims" requirement excludes.
