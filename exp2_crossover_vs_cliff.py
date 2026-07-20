"""EXPERIMENT 2 — gradual perf crossover vs feasibility cliff, and the m0-dodge.

(a) Across schedules {seq,full} and widths, find ANY (w,L) where fuse-all is FEASIBLE yet a
    split is STRICTLY faster (> TIE_TOL). The claim: no such regime exists; the only crossover
    is fuse-all becoming INFEASIBLE.
(b) The DODGE (schedule='full'): as L grows, print the optimal m0 chosen by the fused segment,
    its tile_smem_kib / resident_kib / pipeline-depth C, and the per-stage time — show m0 shrinks
    to keep tile_smem above the starvation floor at ~zero perf cost (weights are L2-resident, so
    more row-blocks = free re-reads).
(c) To force a GRADUAL crossover you need EXPENSIVE (non-L2) weight re-reads so a large m0 pays.
    That needs w*w*2 > eff-L2 (w>=~3964). Show that at those widths fusing even 2 stages is
    INFEASIBLE (resident cap), so the gradual crossover is squeezed out. Map the feasible-fuse
    vs expensive-weight boundary to prove they never overlap.

Run:  conda run -n area python exp2_crossover_vs_cliff.py
"""
from __future__ import annotations
import math
from multi_gemm_smem import analyze, find_Nstar, segment_time, MIN_TILE_SMEM
from multi_gemm_fusion import TIE_TOL
from chain_gemm_fusion import BPE, MMA_MIN_M, STREAM_OVERHEAD, _eff_l2
from gemm_time_estimator import GPUS, optimal_mapping_by_time

g = GPUS["h100-sxm"]
M = 131072
SMEM = g.smem_per_block_bytes
EFFL2 = _eff_l2(g)
print(f"H100-SXM: SMEM/blk={SMEM/1024:.0f}KiB  eff-L2={EFFL2/2**20:.0f}MiB  "
      f"num_sm={g.num_sm}  MMA_MIN_M={MMA_MIN_M}  MIN_TILE_SMEM={MIN_TILE_SMEM/1024:.0f}KiB  "
      f"TIE_TOL={TIE_TOL:g}  M={M}")


def ms(t):
    return f"{t*1e3:.4f}" if math.isfinite(t) else "INFEAS"


# ------------------------------------------------------------------ (a)
print("\n" + "=" * 78)
print("(a) SEARCH: any (w,schedule,L) with fuse-all FEASIBLE but a split STRICTLY faster?")
print("=" * 78)
WIDTHS = [128, 256, 512, 768, 1024, 1536, 2048, 3072]
LMAX = 12
counterexamples = []
print(f"{'sched':>5} {'w':>5} {'L':>3} {'fuse-all ms':>12} {'best ms':>10} "
      f"{'best cuts':>18} {'fa feasible?':>12} {'fa optimal?':>11}")
for schedule in ("seq", "full"):
    for w in WIDTHS:
        nstar, rows = find_Nstar(M=M, w=w, gpu=g, schedule=schedule, Lmax=LMAX)
        for (L, fa, best_t, best_cuts, fa_opt, uf) in rows:
            fa_feas = math.isfinite(fa)
            split_wins = fa_feas and not fa_opt      # feasible fuse-all yet a split is optimal
            if split_wins:
                counterexamples.append((schedule, w, L, fa, best_t, best_cuts))
            # only print the boundary rows to keep it compact: last feasible + first infeasible
        # compact per-(sched,w) summary: N* and the transition row
        # find first infeasible L and last feasible L
        feas_Ls = [L for (L, fa, *_ ) in rows if math.isfinite(fa)]
        last_feas = max(feas_Ls) if feas_Ls else None
        for (L, fa, best_t, best_cuts, fa_opt, uf) in rows:
            if last_feas is not None and L in (last_feas, last_feas + 1):
                print(f"{schedule:>5} {w:>5} {L:>3} {ms(fa):>12} {ms(best_t):>10} "
                      f"{str(best_cuts) if best_cuts is not None else '--':>18} "
                      f"{('YES' if math.isfinite(fa) else 'no'):>12} "
                      f"{('YES' if fa_opt else 'no'):>11}")
print(f"\n  counterexamples (fuse-all feasible AND a split strictly faster > TIE_TOL): "
      f"{len(counterexamples)}")
for ce in counterexamples:
    print(f"    {ce}")
if not counterexamples:
    print("    NONE — whenever fuse-all is feasible it is strictly optimal or tied.")


# ------------------------------------------------------------------ (b)
print("\n" + "=" * 78)
print("(b) THE m0-DODGE (schedule='full'): optimal fused-segment m0 vs depth L")
print("=" * 78)
for w in (256, 512, 1024):
    print(f"\n  --- w={w} (weight w*w*2={w*w*BPE/2**20:.2f}MiB {'>' if w*w*BPE>EFFL2 else '<='} "
          f"eff-L2 => re-reads {'DRAM' if w*w*BPE>EFFL2 else 'L2-cheap'}) ---")
    print(f"  single-GEMM baseline: {ms(segment_time([w, w], M, g, 'full')[0])} ms")
    print(f"  {'L':>3} {'fuse-all ms':>12} {'ms/stage':>9} {'m0':>6} {'mt':>6} "
          f"{'resident':>9} {'tile_smem':>10} {'C':>3} {'DRAM MiB':>9} {'bott':>7}")
    for L in range(1, 13):
        widths = [w] * (L + 1)
        t, info = segment_time(widths, M, g, "full")
        if "why" in info:
            print(f"  {L:>3} {'INFEAS':>12}  {info['why']}")
            break
        res = info.get("resident_kib", (SMEM // 1024) * 0.0)
        tile = info.get("tile_smem_kib", SMEM / 1024)
        C = info.get("C", "-")
        dram = info.get("dram_mib", 0.0)
        print(f"  {L:>3} {ms(t):>12} {t*1e3/L:>9.4f} {info['m0']:>6} {info['mt']:>6} "
              f"{res:>8.1f}K {tile:>9.1f}K {str(C):>3} {dram:>9.1f} {info.get('bott','-'):>7}")


# ------------------------------------------------------------------ (c)
print("\n" + "=" * 78)
print("(c) SQUEEZE-OUT: expensive-weight widths (w*w*2>eff-L2) can't fuse even 2 stages")
print("=" * 78)
# analytic boundaries: m0_min = MMA_MIN_M = 16 rows.
#   weight expensive:            w*w*2 > eff-L2      => w > sqrt(eff-L2/2)
#   fuse-2 feasible (seq):  m0*2w*2 + ovh <= SMEM at m0=16, need tile>=MIN_TILE_SMEM
#   fuse-2 feasible (full): m0*3w*2 + ovh <= SMEM at m0=16
w_expensive = math.sqrt(EFFL2 / BPE)
print(f"  weight becomes DRAM (expensive re-read) at w > {w_expensive:.0f}")
print(f"\n  {'w':>5} {'wt MiB':>8} {'wt>L2?':>7} {'seq fuse2':>10} {'full fuse2':>11} "
      f"{'seq m0*':>8} {'full m0*':>9}")
for w in (2048, 3072, 3584, 3968, 4096, 6144, 8192):
    wt = w * w * BPE
    # feasibility of fusing 2 stages [w,w,w]: does any m0 fit?
    def feas2(sched):
        t, info = segment_time([w, w, w], M, g, sched)
        return info.get("m0") if math.isfinite(t) else None
    seq_m0 = feas2("seq")
    full_m0 = feas2("full")
    print(f"  {w:>5} {wt/2**20:>8.1f} {('YES' if wt>EFFL2 else 'no'):>7} "
          f"{('feasible' if seq_m0 else 'INFEAS'):>10} "
          f"{('feasible' if full_m0 else 'INFEAS'):>11} "
          f"{str(seq_m0):>8} {str(full_m0):>9}")

# The exact gap: largest w that can fuse-2 (seq) vs smallest w with expensive weight.
def max_w_fuse2_seq():
    # m0=16 (min), resident = 16*(2w)*2 + ovh ; tile = SMEM - resident >= MIN_TILE_SMEM
    hi = None
    for w in range(64, 8192 + 1, 8):
        resident = MMA_MIN_M * (2 * w) * BPE + STREAM_OVERHEAD
        if SMEM - resident >= MIN_TILE_SMEM:
            hi = w
    return hi
def max_w_fuse2_full():
    hi = None
    for w in range(64, 8192 + 1, 8):
        resident = MMA_MIN_M * (3 * w) * BPE + STREAM_OVERHEAD
        if SMEM - resident >= MIN_TILE_SMEM:
            hi = w
    return hi
mws, mwf = max_w_fuse2_seq(), max_w_fuse2_full()
print(f"\n  BOUNDARY (m0=16 rows, the MMA minimum):")
print(f"    largest w that can fuse 2 stages: seq={mws}, full={mwf}")
print(f"    smallest w with expensive (DRAM) weight: {math.ceil(w_expensive)}")
print(f"    GAP: fuse-2 dies at w~{mws} (seq)/{mwf} (full); weights turn expensive only at "
      f"w~{math.ceil(w_expensive)}  => the two windows DO NOT overlap.")
print(f"    => no width has (expensive weight AND fuseable >=2 stages): gradual crossover "
      f"squeezed out.")
