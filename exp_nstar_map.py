"""EXPERIMENT 1 - N* MAP.

For BOTH schedules {full, seq}, sweep w in {128,256,512,1024,2048} at M=131072, Lmax=16.
Report N*(w,schedule) = smallest L where fuse-all is NOT optimal, plus WHY (INFEASIBLE vs a
split strictly faster). Check the FULL scaling law N* ~ SMEM/(m0_min*w*bpe): predicted vs observed.

Run:  conda run -n area python exp_nstar_map.py
"""
from __future__ import annotations

import math

from gemm_time_estimator import GPUS
from chain_gemm_fusion import BPE, MMA_MIN_M, STREAM_OVERHEAD, _eff_l2
from multi_gemm_smem import MIN_TILE_SMEM, analyze, find_Nstar, segment_time

M = 131072
WS = [128, 256, 512, 1024, 2048]
LMAX = 16
SCHEDULES = ["full", "seq"]
g = GPUS["h100-sxm"]
SMEM = g.smem_per_block_bytes

# ---- FULL scaling-law prediction ---------------------------------------------------------
# fuse-all of L stages (uniform w): resident = m0*(L+1)*w*BPE + STREAM_OVERHEAD.
# tile feasibility: SMEM - resident >= MIN_TILE_SMEM  =>  m0*(L+1)*w*BPE <= BUDGET.
BUDGET = SMEM - STREAM_OVERHEAD - MIN_TILE_SMEM          # bytes available to resident activations
# at smallest m0 = MMA_MIN_M:  (L+1) <= BUDGET/(m0_min*w*BPE) == K/w,  K = BUDGET/(m0_min*BPE)
K = BUDGET / (MMA_MIN_M * BPE)                            # = "6368" ; N*(first infeasible) = floor(K/w)


def predicted_nstar_full(w: int) -> int:
    # largest feasible L: L+1 <= K/w  => L <= K/w - 1 ; first infeasible L = that + 1 = floor(K/w)
    max_feasible_L = math.floor(K / w - 1)
    return max_feasible_L + 1                             # first infeasible depth


def why_at(w: int, L: int, schedule: str) -> str:
    """Explain why fuse-all is not optimal at depth L (assumes it is not)."""
    a = analyze(M, w, L, g, schedule)
    fa_t = a["fuse_all"][0]
    best_t, best_cuts = a["best"] if a["best"] else (float("inf"), None)
    if not math.isfinite(fa_t):
        # find the fuse-all seg info reason
        _, info = segment_time([w] * (L + 1), M, g, schedule)
        return f"INFEASIBLE ({info.get('why','')})"
    return f"SPLIT strictly faster: best cuts={best_cuts} {best_t*1e3:.4f}ms < fuse-all {fa_t*1e3:.4f}ms ({fa_t/best_t:.4f}x)"


def fa_m0_at(w: int, L: int, schedule: str):
    """Return m0 the fuse-all segment used at depth L (None if infeasible)."""
    _, info = segment_time([w] * (L + 1), M, g, schedule)
    return info.get("m0"), info


def main():
    print(f"=== EXPERIMENT 1: N* MAP  (H100-sxm, M={M}, Lmax={LMAX}) ===")
    print(f"SMEM/blk={SMEM/1024:.0f}KiB  STREAM_OVERHEAD={STREAM_OVERHEAD/1024:.0f}KiB  "
          f"MIN_TILE_SMEM={MIN_TILE_SMEM/1024:.0f}KiB  m0_min={MMA_MIN_M}  BPE={BPE}")
    print(f"FULL budget for resident acts = SMEM-overhead-tilefloor = {BUDGET}B ; "
          f"K=BUDGET/(m0_min*BPE)={K:.1f}  => predicted N*(full)=floor(K/w)\n")

    summary = {}
    for schedule in SCHEDULES:
        print(f"\n############## SCHEDULE = {schedule.upper()} ##############")
        for w in WS:
            nstar, rows = find_Nstar(M, w, g, schedule, LMAX)
            summary[(schedule, w)] = (nstar, rows)
            pred = predicted_nstar_full(w) if schedule == "full" else None
            hdr = f"--- w={w}"
            if schedule == "full":
                hdr += f"   predicted N*(full)=floor({K:.0f}/{w})={pred}"
            print(hdr)
            print(f"  {'L':>3} {'fuse-all ms':>12} {'best ms':>11} {'best cuts':>18} "
                  f"{'fa opt?':>8} {'fa m0':>7} {'resid KiB':>9} {'tile KiB':>9}")
            for (L, fa, bt, bc, opt, uf) in rows:
                m0, info = fa_m0_at(w, L, schedule)
                fam = f"{fa*1e3:.4f}" if math.isfinite(fa) else "INFEAS"
                btm = f"{bt*1e3:.4f}" if math.isfinite(bt) else "INFEAS"
                resid = f"{info.get('resident_kib', float('nan')):.1f}" if 'resident_kib' in info else "--"
                tile = f"{info.get('tile_smem_kib', float('nan')):.1f}" if 'tile_smem_kib' in info else "--"
                bcs = str(bc) if bc is not None else "--"
                print(f"  {L:>3} {fam:>12} {btm:>11} {bcs:>18} "
                      f"{('YES' if opt else 'no'):>8} {str(m0):>7} {resid:>9} {tile:>9}")
            if nstar is not None:
                print(f"  >> N*({schedule},w={w}) = {nstar}   WHY: {why_at(w, nstar, schedule)}")
            else:
                print(f"  >> N*({schedule},w={w}) = None (fuse-all stays optimal through L={LMAX})")

    # ---- final compact summary table -----------------------------------------------------
    print("\n\n================ SUMMARY: N*(w, schedule) ================")
    print(f"{'w':>6} | {'pred N*(full)':>13} | {'obs N*(full)':>12} | {'why(full@N*)':>14} "
          f"| {'obs N*(seq)':>11} | {'why(seq@N*)':>14}")
    for w in WS:
        pn = predicted_nstar_full(w)
        pn_s = str(pn) if pn <= LMAX else f"{pn}(>Lmax)"
        nf, _ = summary[("full", w)]
        ns, _ = summary[("seq", w)]
        whyf = "-"
        if nf is not None:
            whyf = "INFEAS" if "INFEASIBLE" in why_at(w, nf, "full") else "split<fa"
        whys = "-"
        if ns is not None:
            whys = "INFEAS" if "INFEASIBLE" in why_at(w, ns, "seq") else "split<fa"
        print(f"{w:>6} | {pn_s:>13} | {str(nf):>12} | {whyf:>14} | {str(ns):>11} | {whys:>14}")


if __name__ == "__main__":
    main()
