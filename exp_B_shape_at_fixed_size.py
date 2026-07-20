"""EXPERIMENT B - SHAPE SWEEP AT FIXED SIZE (isolate SHAPE).

Hold SIZE (geomean of the 4 dims) EXACTLY constant at 3 levels and vary SHAPE (aspect ratios).
Dims are powers of 2, exponents = G + delta with sum(delta)=0 -> geomean = 2^G exactly.
Same delta grid at every G => identical *shapes* at different *sizes*.

Question: at each fixed size, do different SHAPES produce different verdicts? Which shape
feature predicts FUSE? Quantify how much verdict variance at fixed size is explained by shape.
"""
from __future__ import annotations
import itertools, math
import numpy as np
from chain_gemm_probe import probe
from gemm_time_estimator import GPUS

GPU = GPUS["h100-sxm"]
DMIN, DMAX = 8, 16          # 2^8=256 .. 2^16=65536
DELTA = (-2, -1, 0, 1, 2)   # shape offset per dim (log2), must keep exponent in [8,16]
SIZE_LEVELS = {"small": 10, "medium": 12, "large": 14}  # G -> geomean 2^G = 1024/4096/16384

# --- shape grid: (dm,dk,dn,dp) with sum 0 (holds geomean exactly), each in DELTA -----------
shapes = [d for d in itertools.product(DELTA, repeat=4) if sum(d) == 0]
print(f"# shapes per size level = {len(shapes)}  ; total probe() calls = {len(shapes)*len(SIZE_LEVELS)}")

# threshold reference (all SIZE-vs-hardware; shape moves the products across these at fixed G):
#   f_C>1 (C round-trip saved)  <=> M*N1*2 > 30MB <=> (m+n) >= 24  [2^23.9]
#   f_B>1 (weight B re-read x mt) <=> (k+n) >= 24 ; f_D>1 <=> (n+p) >= 24
#   infeasible <=> min(N1,N2) >= 8192 <=> min(n,p) >= 13
records = []
for level, G in SIZE_LEVELS.items():
    for (dm, dk, dn, dp) in shapes:
        m, k, n, p = G+dm, G+dk, G+dn, G+dp
        assert DMIN <= min(m, k, n, p) and max(m, k, n, p) <= DMAX
        M, K1, N1, N2 = 2**m, 2**k, 2**n, 2**p
        pr = probe(M, K1, N1, N2, GPU)
        records.append(dict(
            level=level, G=G, dm=dm, dk=dk, dn=dn, dp=dp,
            M=M, K1=K1, N1=N1, N2=N2,
            winner=pr["winner"], feasible=pr["feasible"],
            speedup=pr["speedup"], bott=pr["bott"],
            # scale-free shape features (identical across size levels):
            aA=pr["aA_log2"], aB=pr["aB_log2"], aD=pr["aD_log2"],
            # products the mechanism depends on (as exponent sums, shape part = delta sum):
            mn=m+n, kn=k+n, npx=n+p, minNp=min(n, p),
            f_held=pr["f_held"], f_B=pr["f_B"], f_D=pr["f_D"], f_C=pr["f_C"],
        ))

def counts(rs):
    c = {"FUSE": 0, "unfuse": 0, "infeasible": 0}
    for r in rs:
        c[r["winner"]] += 1
    return c

print("\n================  VERDICT DISTRIBUTION PER FIXED SIZE  ================")
print(f"{'size':>7} {'geomean':>8} | {'FUSE':>5} {'unfuse':>6} {'infeas':>6} | "
      f"{'medianSp':>8} {'maxSp':>6} | shapes vary verdict?")
for level, G in SIZE_LEVELS.items():
    rs = [r for r in records if r["level"] == level]
    c = counts(rs)
    sp = [r["speedup"] for r in rs if r["feasible"]]
    nz = sum(1 for v in c.values() if v > 0)
    print(f"{level:>7} {2**G:>8} | {c['FUSE']:>5} {c['unfuse']:>6} {c['infeasible']:>6} | "
          f"{np.median(sp) if sp else float('nan'):>8.3f} {max(sp) if sp else float('nan'):>6.3f} | "
          f"{'YES ('+str(nz)+' verdicts)' if nz>1 else 'NO (all same)'}")

# --- verdict ENTROPY at each fixed size: >0 bits => shape alone moves the verdict -----------
def entropy(rs):
    c = counts(rs); n = len(rs)
    return -sum((v/n)*math.log2(v/n) for v in c.values() if v > 0)
print("\n  verdict entropy at fixed size (bits; >0 => SHAPE alone changes the verdict):")
for level in SIZE_LEVELS:
    rs = [r for r in records if r["level"] == level]
    print(f"    {level:>7}: H={entropy(rs):.3f} bits over {len(rs)} shapes")

# ------------------------------------------------------------------------------------------
# WHICH SHAPE FEATURE PREDICTS THE VERDICT?  (mutual info + best single-feature threshold)
# ------------------------------------------------------------------------------------------
def mutual_info(xs, ys):
    """MI(bits) between discrete feature xs and label ys."""
    n = len(xs); mi = 0.0
    xv, yv = set(xs), set(ys)
    for x in xv:
        px = sum(1 for a in xs if a == x)/n
        for y in yv:
            py = sum(1 for b in ys if b == y)/n
            pxy = sum(1 for a, b in zip(xs, ys) if a == x and b == y)/n
            if pxy > 0:
                mi += pxy*math.log2(pxy/(px*py))
    return mi

SHAPE_FEATS = {
    "dm(M rel)":  lambda r: r["dm"],
    "dk(K1 rel)": lambda r: r["dk"],
    "dn(N1 rel)": lambda r: r["dn"],
    "dp(N2 rel)": lambda r: r["dp"],
    "aA=lg M/K1": lambda r: r["aA"],
    "aB=lg K1/N1": lambda r: r["aB"],
    "aD=lg N1/N2": lambda r: r["aD"],
    "dm+dn (~f_C)": lambda r: r["dm"]+r["dn"],
    "dk+dn (~f_B)": lambda r: r["dk"]+r["dn"],
    "dn+dp (~f_D)": lambda r: r["dn"]+r["dp"],
    "min(dn,dp)feas": lambda r: min(r["dn"], r["dp"]),
}

def stump_acc(vals, labels):
    """best threshold rule val>=t -> majority label; return accuracy of best split (binary FUSE/notFUSE style handled by caller)."""
    order = sorted(set(vals)); best = 0.0
    for t in order:
        # predict class by side using majority on each side
        left = [l for v, l in zip(vals, labels) if v < t]
        right = [l for v, l in zip(vals, labels) if v >= t]
        acc = 0
        for grp in (left, right):
            if grp:
                maj = max(set(grp), key=grp.count)
                acc += sum(1 for l in grp if l == maj)
        best = max(best, acc/len(labels))
    # also the trivial single-group (t below min): majority baseline
    maj = max(set(labels), key=labels.count)
    best = max(best, sum(1 for l in labels if l == maj)/len(labels))
    return best

print("\n================  SHAPE-FEATURE -> VERDICT (pooled over all shapes, per size)  ============")
print("MI in bits of the 3-way verdict; stump = best single-threshold accuracy of that feature.")
for level in SIZE_LEVELS:
    rs = [r for r in records if r["level"] == level]
    labels = [r["winner"] for r in rs]
    base = max(set(labels), key=labels.count)
    base_acc = sum(1 for l in labels if l == base)/len(labels)
    print(f"\n  [{level}]  base-rate(majority='{base}') acc={base_acc:.3f}  (H={entropy(rs):.3f} bits)")
    rows = []
    for name, fn in SHAPE_FEATS.items():
        vals = [fn(r) for r in rs]
        rows.append((mutual_info(vals, labels), stump_acc(vals, labels), name))
    for mi, acc, name in sorted(rows, reverse=True):
        print(f"     {name:>15}:  MI={mi:.3f} bits   stump_acc={acc:.3f}")

# ------------------------------------------------------------------------------------------
# HOW MUCH VERDICT VARIANCE AT FIXED SIZE IS EXPLAINED BY SHAPE?
#   (1) at fixed size, size features are CONSTANT -> any variance is 100% shape by construction
#   (2) predictability: fit log2(speedup) ~ shape features (feasible only) -> R^2
#   (3) a shape-only 2-rule classifier accuracy for the 3-way verdict
# ------------------------------------------------------------------------------------------
print("\n================  VARIANCE-EXPLAINED-BY-SHAPE AT FIXED SIZE  ================")
for level in SIZE_LEVELS:
    rs = [r for r in records if r["level"] == level]
    feas = [r for r in rs if r["feasible"]]
    # (2) linear regression of log2(speedup) on shape deltas (3 free dof; use all 4 + intercept via lstsq)
    if len(feas) > 5 and len(set(r["speedup"] for r in feas)) > 1:
        X = np.array([[1, r["dm"], r["dk"], r["dn"], r["dp"]] for r in feas], float)
        y = np.array([math.log2(r["speedup"]) for r in feas], float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y-yhat)**2)); ss_tot = float(np.sum((y-y.mean())**2))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    else:
        r2, beta = float("nan"), None
    # (3) mechanistic shape rule: FUSE iff (dm+dn high) & (dk+dn low) & (dn+dp low) & feasible
    #     implement as: predict FUSE if feasible and (dm+dn - dk) >= tuned; else infeasible if min>=13 else unfuse
    labels = [r["winner"] for r in rs]
    # simple 3-way shape classifier:
    def predict(r):
        if r["minNp"] >= 13:            # shape -> infeasible (min(N1,N2)>=8192)
            return "infeasible"
        # FUSE favored when C big (mn>=24) and weights fit (kn<24 and npx<24)
        if r["mn"] >= 24 and r["kn"] < 24 and r["npx"] < 24:
            return "FUSE"
        return "unfuse"
    acc = sum(1 for r in rs if predict(r) == r["winner"])/len(rs)
    print(f"  [{level}] shape-only mechanistic classifier acc={acc:.3f} "
          f"| log2(speedup)~shape R^2={r2:.3f}"
          + (f"  beta(dm,dk,dn,dp)=[{beta[1]:+.2f},{beta[2]:+.2f},{beta[3]:+.2f},{beta[4]:+.2f}]" if beta is not None else ""))

# --- pooled mechanistic classifier accuracy (all sizes) -----------------------------------
def predict(r):
    if r["minNp"] >= 13: return "infeasible"
    if r["mn"] >= 24 and r["kn"] < 24 and r["npx"] < 24: return "FUSE"
    return "unfuse"
acc_all = sum(1 for r in records if predict(r) == r["winner"])/len(records)
print(f"\n  POOLED shape-threshold classifier accuracy (all {len(records)} cases) = {acc_all:.3f}")

# ------------------------------------------------------------------------------------------
# WHAT SHAPE => FUSE ? profile FUSE vs unfuse vs infeasible on shape features
# ------------------------------------------------------------------------------------------
print("\n================  MEAN SHAPE FEATURE BY VERDICT (pooled)  ================")
print(f"{'verdict':>10} {'n':>4} | {'dm':>5} {'dk':>5} {'dn':>5} {'dp':>5} | "
      f"{'aA':>5} {'aB':>5} {'aD':>5} | {'minNp':>6} {'medSp':>6}")
for w in ("FUSE", "unfuse", "infeasible"):
    rs = [r for r in records if r["winner"] == w]
    if not rs:
        print(f"{w:>10}    0 |  (none)"); continue
    def mean(key): return np.mean([r[key] for r in rs])
    sp = [r["speedup"] for r in rs if r["feasible"]]
    print(f"{w:>10} {len(rs):>4} | {mean('dm'):>5.2f} {mean('dk'):>5.2f} {mean('dn'):>5.2f} {mean('dp'):>5.2f} | "
          f"{mean('aA'):>5.2f} {mean('aB'):>5.2f} {mean('aD'):>5.2f} | {mean('minNp'):>6.2f} "
          f"{np.median(sp) if sp else float('nan'):>6.3f}")

# ------------------------------------------------------------------------------------------
# COMPACT QUOTABLE TABLE: at medium size, a few illustrative shapes (square/tall/wide)
# ------------------------------------------------------------------------------------------
print("\n================  ILLUSTRATIVE SHAPES @ medium (geomean=4096, size FIXED)  ============")
print(f"{'shape(dm,dk,dn,dp)':>20} | {'M':>6}x{'K1':>6}x{'N1':>6}x{'N2':>6} | {'verdict':>10} {'sp':>6} "
      f"| f_C  f_B  f_D  minNp bott")
illus = [(0,0,0,0),(2,-2,0,0),(2,0,-2,0),(2,0,0,-2),(-2,2,0,0),(0,0,2,-2),(0,-2,2,0),(0,0,-2,2),(2,-1,-1,0),(-2,0,2,0)]
mid = {(r["dm"],r["dk"],r["dn"],r["dp"]): r for r in records if r["level"]=="medium"}
for s in illus:
    r = mid.get(s)
    if not r: continue
    sp = f"{r['speedup']:.3f}" if r["feasible"] else "  --  "
    print(f"{str(s):>20} | {r['M']:>6}x{r['K1']:>6}x{r['N1']:>6}x{r['N2']:>6} | {r['winner']:>10} {sp:>6} "
          f"| {r['f_C']:.2f} {r['f_B']:.2f} {r['f_D']:.2f} {r['minNp']:>5} {r['bott']}")
