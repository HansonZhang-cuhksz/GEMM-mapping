"""EXPERIMENT C -- RATIO COLLAPSE.

Test whether the FUSE-vs-not verdict is a clean function of the dimensionless
size-vs-hardware ratios (f_held, f_B, f_C, AI/ridge), independent of raw shape/size.

Sample ~300 random power-of-2 points, get winner + ratios via the shared probe, then
fit hand-rolled depth<=3 CART decision trees (sklearn unavailable) on THREE feature sets:
  RATIO-only, SHAPE-only, SIZE-only. Compare train accuracy on the binary FUSE-vs-not task.
Also report the clean feasibility rule (infeasible predicted by f_held) and the single
most predictive ratio.
"""
from __future__ import annotations
import math, random
from collections import Counter
from chain_gemm_probe import probe
from gemm_time_estimator import GPUS

random.seed(20260717)
G = GPUS["h100-sxm"]
EXPS = list(range(8, 17))          # 2^8=256 .. 2^16=65536
N = 300

# ---- sample N unique power-of-2 points ----
seen = set()
pts = []
while len(pts) < N:
    e = (random.choice(EXPS), random.choice(EXPS), random.choice(EXPS), random.choice(EXPS))
    if e in seen:
        continue
    seen.add(e)
    pts.append(tuple(1 << x for x in e))

rows = []
for (M, K1, N1, N2) in pts:
    p = probe(M, K1, N1, N2, G)
    ai = p["AI_fused"]
    ai_over_ridge = (ai / p["ridge"]) if (p["feasible"] and math.isfinite(ai)) else 0.0
    rows.append({
        "M": M, "K1": K1, "N1": N1, "N2": N2,
        "winner": p["winner"], "feasible": p["feasible"],
        "fuse": 1 if p["winner"] == "FUSE" else 0,        # binary target: FUSE vs not(unfuse|infeasible)
        # RATIO features (dimensionless, size-vs-hardware)
        "f_held": p["f_held"], "f_B": p["f_B"], "f_C": p["f_C"], "f_D": p["f_D"],
        "ai_over_ridge": ai_over_ridge,
        # SHAPE features (scale-free aspect ratios)
        "aA": p["aA_log2"], "aB": p["aB_log2"], "aD": p["aD_log2"],
        # SIZE features (absolute magnitude)
        "lg_geo": math.log2(p["geomean_dim"]), "lg_tf": math.log2(p["tflops"]),
        "bott": p["bott"],
    })

# ---- class distribution ----
wc = Counter(r["winner"] for r in rows)
nfuse = sum(r["fuse"] for r in rows)
print("=" * 78)
print(f"EXPERIMENT C: {N} random power-of-2 pts, dims in [256,65536], H100-SXM")
print(f"class counts: FUSE={wc['FUSE']}  unfuse={wc['unfuse']}  infeasible={wc['infeasible']}")
print(f"binary target FUSE-vs-not: FUSE={nfuse}  not={N-nfuse}  (base-rate acc={max(nfuse,N-nfuse)/N:.3f})")
print("=" * 78)

# =================== hand-rolled CART (Gini), depth<=3 ===================
def gini(ys):
    n = len(ys)
    if n == 0:
        return 0.0
    c = Counter(ys)
    return 1.0 - sum((v / n) ** 2 for v in c.values())

def best_split(X, y, feat_idx):
    n = len(y); base = gini(y); best = None
    for fi in feat_idx:
        vals = sorted(set(row[fi] for row in X))
        for i in range(len(vals) - 1):
            thr = (vals[i] + vals[i + 1]) / 2.0
            L = [j for j in range(n) if X[j][fi] <= thr]
            R = [j for j in range(n) if X[j][fi] > thr]
            if not L or not R:
                continue
            wg = (len(L) * gini([y[j] for j in L]) + len(R) * gini([y[j] for j in R])) / n
            if best is None or wg < best[0] - 1e-12:
                best = (wg, fi, thr, L, R)
    if best is None or best[0] >= base - 1e-9:
        return None
    return best

class Node:
    __slots__ = ("leaf", "cls", "fi", "thr", "L", "R")
    def __init__(self):
        self.leaf = True; self.cls = None; self.fi = None; self.thr = None; self.L = None; self.R = None

def build(X, y, depth, feat_idx):
    nd = Node()
    nd.cls = Counter(y).most_common(1)[0][0]
    if depth == 0 or len(set(y)) == 1:
        return nd
    sp = best_split(X, y, feat_idx)
    if sp is None:
        return nd
    _, fi, thr, L, R = sp
    nd.leaf = False; nd.fi = fi; nd.thr = thr
    nd.L = build([X[j] for j in L], [y[j] for j in L], depth - 1, feat_idx)
    nd.R = build([X[j] for j in R], [y[j] for j in R], depth - 1, feat_idx)
    return nd

def predict(nd, x):
    while not nd.leaf:
        nd = nd.L if x[nd.fi] <= nd.thr else nd.R
    return nd.cls

def accuracy(nd, X, y):
    return sum(predict(nd, X[j]) == y[j] for j in range(len(y))) / len(y)

def print_tree(nd, names, ind=0):
    pad = "  " * ind
    if nd.leaf:
        print(f"{pad}-> predict {'FUSE' if nd.cls==1 else 'not'}")
        return
    print(f"{pad}if {names[nd.fi]} <= {nd.thr:.4g}:")
    print_tree(nd.L, names, ind + 1)
    print(f"{pad}else ({names[nd.fi]} > {nd.thr:.4g}):")
    print_tree(nd.R, names, ind + 1)

FEATS = {
    "RATIO": ["f_held", "f_B", "f_C", "f_D", "ai_over_ridge"],
    "SHAPE": ["aA", "aB", "aD"],
    "SIZE":  ["lg_geo", "lg_tf"],
}
y = [r["fuse"] for r in rows]

print("\n--- FUSE-vs-not, hand-rolled CART depth<=3, TRAIN accuracy ---")
results = {}
trees = {}
for name, feats in FEATS.items():
    X = [[r[f] for f in feats] for r in rows]
    tree = build(X, y, 3, list(range(len(feats))))
    acc = accuracy(tree, X, y)
    results[name] = acc
    trees[name] = (tree, feats)
    print(f"  {name:6s} ({','.join(feats)}): acc={acc:.3f}")

# =================== single-feature stump accuracy (most predictive) ===================
def best_stump_acc(vals, y):
    uq = sorted(set(vals)); best = 0.0; bthr = None; bdir = None
    for i in range(len(uq) - 1):
        thr = (uq[i] + uq[i + 1]) / 2.0
        # rule A: predict FUSE if <= thr
        a = sum((1 if vals[j] <= thr else 0) == y[j] for j in range(len(y))) / len(y)
        # rule B: predict FUSE if > thr
        b = sum((1 if vals[j] > thr else 0) == y[j] for j in range(len(y))) / len(y)
        if a > best:
            best, bthr, bdir = a, thr, "<=thr=>FUSE"
        if b > best:
            best, bthr, bdir = b, thr, ">thr=>FUSE"
    return best, bthr, bdir

print("\n--- single-ratio stump accuracy (FUSE-vs-not) ---")
allfeats = ["f_held", "f_B", "f_C", "f_D", "ai_over_ridge"]
stump = {}
for f in allfeats:
    vals = [r[f] for r in rows]
    acc, thr, d = best_stump_acc(vals, y)
    stump[f] = (acc, thr, d)
    print(f"  {f:14s}: acc={acc:.3f}  ({d} @ {thr:.4g})")
best_ratio = max(stump, key=lambda k: stump[k][0])
print(f"  => single most predictive ratio: {best_ratio}  (acc={stump[best_ratio][0]:.3f})")

# =================== feasibility cleanly by f_held ===================
print("\n--- feasibility as a function of f_held (ratio) ---")
fh_feas = [(r["f_held"], r["feasible"]) for r in rows]
# infeasible iff f_held > ~1/16; find empirical clean threshold
inf_max_fh = max((fh for fh, fe in fh_feas if not fe), default=float("nan"))
feas_min_fh = min((fh for fh, fe in fh_feas if fe), default=float("nan"))
# accuracy of rule: feasible iff f_held <= t, sweep
uq = sorted(set(r["f_held"] for r in rows))
best = (0, None)
for i in range(len(uq) - 1):
    t = (uq[i] + uq[i + 1]) / 2
    acc = sum(((r["f_held"] <= t) == r["feasible"]) for r in rows) / N
    if acc > best[0]:
        best = (acc, t)
print(f"  rule 'feasible iff f_held <= {best[1]:.4g}': acc={best[0]:.3f}")
print(f"  max f_held among infeasible pts = {inf_max_fh:.4g}; "
      f"min f_held among feasible pts = {feas_min_fh:.4g}  (theory boundary ~0.0625=1/16)")

# shape/size cannot predict feasibility (min(N1,N2) is absolute, not an aspect or overall size)
Xs = [[r[f] for f in FEATS["SHAPE"]] for r in rows]
ys = [1 if r["feasible"] else 0 for r in rows]
ts = build(Xs, ys, 3, list(range(3)))
print(f"  SHAPE-only tree predicting feasibility: acc={accuracy(ts, Xs, ys):.3f}")
Xz = [[r[f] for f in FEATS["SIZE"]] for r in rows]
tz = build(Xz, ys, 3, list(range(2)))
print(f"  SIZE-only  tree predicting feasibility: acc={accuracy(tz, Xz, ys):.3f}")
Xr = [[r[f] for f in FEATS["RATIO"]] for r in rows]
tr = build(Xr, ys, 3, list(range(5)))
print(f"  RATIO-only tree predicting feasibility: acc={accuracy(tr, Xr, ys):.3f}")

# =================== print the RATIO tree in words ===================
print("\n--- discovered RATIO decision boundary (FUSE-vs-not tree) ---")
tree, feats = trees["RATIO"]
print_tree(tree, feats)

# =================== summary ===================
print("\n" + "=" * 78)
print("SUMMARY  (train accuracy, FUSE-vs-not, depth<=3)")
print(f"  RATIO-only : {results['RATIO']:.3f}")
print(f"  SHAPE-only : {results['SHAPE']:.3f}")
print(f"  SIZE-only  : {results['SIZE']:.3f}")
print(f"  base-rate  : {max(nfuse,N-nfuse)/N:.3f}")
print(f"  most predictive single ratio: {best_ratio} (stump acc {stump[best_ratio][0]:.3f})")
print("=" * 78)
