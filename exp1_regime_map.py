"""EXPERIMENT 1 — REGIME MAP over (L, w) at M=131072 on H100-SXM.

Builds on the validated multi_gemm_fusion harness (imported, not reimplemented).
For each (L, w) cell: fuse-all time, best-partition (time+cuts+max-seg-len),
fully-unfused time, speedup(unfused/fuse-all), fuse-all optimal? and a regime class.
"""
import math
from multi_gemm_fusion import analyze
from gemm_time_estimator import GPUS

GPU = GPUS['h100-sxm']
M = 131072
Ls = [2, 3, 4, 5, 6]
ws = [128, 256, 512, 1024, 2048, 4096]

def ms(t):
    return f"{t*1e3:.4f}" if math.isfinite(t) else "INFEAS"

# classification thresholds
def classify(spd, fa_optimal, fa_feasible):
    if not fa_feasible:
        return "INFEASIBLE"
    if spd >= 1.10 and fa_optimal:
        return "MEM-BOUND"
    if spd < 1.05:
        return "COMPUTE-BOUND"
    return "TRANSITION"

rows = []
print(f"{'L':>2} {'w':>5} | {'fuse_all_ms':>12} {'best_ms':>12} {'unfused_ms':>12} "
      f"{'spd(uf/fa)':>10} {'best_cuts':>16} {'maxseg':>6} {'fa_opt':>7} {'class':>12}")
print("-"*130)

# store per-(w) fuse_all times to compute per-stage marginal benefit in mem-bound regime
fa_by_w = {w: {} for w in ws}
uf_by_w = {w: {} for w in ws}

for w in ws:
    for L in Ls:
        a = analyze(M, w, L, GPU)
        fa_t = a['fuse_all'][0]
        uf_t = a['unfused'][0]
        best = a['best']
        fa_feasible = math.isfinite(fa_t)
        if best is not None:
            bt, bcuts, _ = best
            maxseg = a['maxseg'](bcuts)
            fa_opt = (bcuts == ())
        else:
            bt, bcuts, maxseg, fa_opt = float('inf'), None, None, False
        spd = (uf_t / fa_t) if fa_feasible else float('nan')
        cls = classify(spd, fa_opt, fa_feasible)
        fa_by_w[w][L] = fa_t
        uf_by_w[w][L] = uf_t
        # best beats fa factor
        beat = (fa_t / bt) if (fa_feasible and math.isfinite(bt)) else float('nan')
        cutstr = str(bcuts) if bcuts is not None else "-"
        print(f"{L:>2} {w:>5} | {ms(fa_t):>12} {ms(bt):>12} {ms(uf_t):>12} "
              f"{spd:>10.3f} {cutstr:>16} {str(maxseg):>6} {str(fa_opt):>7} {cls:>12}")
        rows.append((L, w, fa_t, bt, uf_t, spd, bcuts, maxseg, fa_opt, cls, beat))
    print("-"*130)

# ---- per-stage marginal analysis in mem-bound regime ----
print("\n\n=== PER-STAGE MARGINAL: fuse-all time vs L (does adding stages keep helping?) ===")
for w in ws:
    print(f"\n w={w}:")
    print(f"   {'L':>2} {'fuseall_ms':>11} {'unfused_ms':>11} {'spd':>7} "
          f"{'d(fa)/dL_ms':>12} {'d(uf)/dL_ms':>12} {'uf-fa saved_ms':>14}")
    prev_fa = prev_uf = None
    for L in Ls:
        fa = fa_by_w[w][L]; uf = uf_by_w[w][L]
        spd = (uf/fa) if math.isfinite(fa) else float('nan')
        dfa = (fa - prev_fa)*1e3 if (prev_fa is not None and math.isfinite(fa) and math.isfinite(prev_fa)) else float('nan')
        duf = (uf - prev_uf)*1e3 if (prev_uf is not None) else float('nan')
        saved = (uf - fa)*1e3 if math.isfinite(fa) else float('nan')
        print(f"   {L:>2} {ms(fa):>11} {ms(uf):>11} {spd:>7.3f} "
              f"{dfa:>12.4f} {duf:>12.4f} {saved:>14.4f}")
        prev_fa = fa if math.isfinite(fa) else prev_fa
        prev_uf = uf
