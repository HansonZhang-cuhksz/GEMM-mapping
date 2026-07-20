"""EXPERIMENT D — Feature importance: SHAPE vs SIZE vs RATIO for 2-GEMM fusion preference.

Sample ~300 power-of-2 chain shapes, get the estimator's FUSE/unfuse verdict via the shared
probe, then quantify per-feature predictive importance grouped into SHAPE / SIZE / RATIO.

sklearn is NOT available -> hand-rolled: binned mutual information, best single-split info gain,
a bagged random forest with Gini (impurity-decrease) importances, and group-only classifier
accuracy (direct explanatory power of each feature group).
"""
from __future__ import annotations
import math, random
from chain_gemm_probe import probe
from gemm_time_estimator import GPUS

SEED = 1234
N_SAMPLE = 320                      # <= ~350 probe calls
EXPS = list(range(8, 17))           # 2^8=256 .. 2^16=65536
G = GPUS["h100-sxm"]

# ---------------- sample ----------------
rng = random.Random(SEED)
seen = set()
rows = []
while len(rows) < N_SAMPLE:
    M  = 1 << rng.choice(EXPS)
    K1 = 1 << rng.choice(EXPS)
    N1 = 1 << rng.choice(EXPS)
    N2 = 1 << rng.choice(EXPS)
    key = (M, K1, N1, N2)
    if key in seen:
        continue
    seen.add(key)
    p = probe(M, K1, N1, N2, G)
    rows.append(p)

n_feas = sum(1 for p in rows if p["feasible"])
n_fuse = sum(1 for p in rows if p["winner"] == "FUSE")
n_unf  = sum(1 for p in rows if p["winner"] == "unfuse")
n_inf  = sum(1 for p in rows if p["winner"] == "infeasible")
print(f"probe calls = {len(rows)}  feasible={n_feas}  FUSE={n_fuse}  unfuse={n_unf}  infeasible={n_inf}")

# ---------------- features ----------------
def feats(p):
    M, K1, N1, N2 = p["M"], p["K1"], p["N1"], p["N2"]
    f = {
        # SHAPE (scale-free aspect ratios)
        "aA_log2": p["aA_log2"], "aB_log2": p["aB_log2"], "aD_log2": p["aD_log2"],
        # SIZE (absolute magnitude)
        "lg_geomean": math.log2(p["geomean_dim"]),
        "lg_M": math.log2(M), "lg_K1": math.log2(K1), "lg_N1": math.log2(N1), "lg_N2": math.log2(N2),
        "lg_tflops": math.log2(p["tflops"]),
        # RATIO (size-vs-hardware); monotone log transforms -> tree/MI unaffected
        "lg_f_held": math.log2(p["f_held"]),
        "lg_f_B": math.log2(p["f_B"]),
        "lg_f_C": math.log2(p["f_C"]),
        "lg_AI_ridge": math.log2(p["AI_fused"] / p["ridge"]) if p["feasible"] and math.isfinite(p["AI_fused"]) else float("nan"),
    }
    return f

GROUPS = {
    "SHAPE": ["aA_log2", "aB_log2", "aD_log2"],
    "SIZE":  ["lg_geomean", "lg_M", "lg_K1", "lg_N1", "lg_N2", "lg_tflops"],
    "RATIO": ["lg_f_held", "lg_f_B", "lg_f_C", "lg_AI_ridge"],
}
FEAT_NAMES = [f for g in GROUPS.values() for f in g]
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}

# Build dataset. Label target = 1 if FUSE preferred.  Restrict to FEASIBLE points so
# AI/ridge (and the FUSE/unfuse contrast) are well defined -- feasibility here is ~100%.
X, y = [], []
for p in rows:
    if not p["feasible"]:
        continue
    fv = feats(p)
    if any(math.isnan(v) for v in fv.values()):
        continue
    X.append([fv[k] for k in FEAT_NAMES])
    y.append(1 if p["winner"] == "FUSE" else 0)

n = len(y)
pos = sum(y)
print(f"dataset n={n}  FUSE(+)={pos}  unfuse(-)={n-pos}  base_rate={pos/n:.3f}")

# ================= hand-rolled importances =================
def gini(labels):
    m = len(labels)
    if m == 0:
        return 0.0
    p = sum(labels) / m
    return 2 * p * (1 - p)

def best_split(col, labels):
    """Return (info_gain_gini, threshold, acc) for best single split on one feature column."""
    order = sorted(range(len(col)), key=lambda i: col[i])
    vals = [col[i] for i in order]
    labs = [labels[i] for i in order]
    m = len(labs)
    parent = gini(labs)
    tot_pos = sum(labs)
    best_g, best_t, best_acc = -1.0, None, max(tot_pos, m - tot_pos) / m
    left_pos = 0
    for i in range(1, m):
        left_pos += labs[i - 1]
        if vals[i] == vals[i - 1]:
            continue
        nl, nr = i, m - i
        gl = 2 * (left_pos / nl) * (1 - left_pos / nl)
        rp = tot_pos - left_pos
        gr = 2 * (rp / nr) * (1 - rp / nr)
        ig = parent - (nl * gl + nr * gr) / m
        # accuracy of this threshold stump (predict majority on each side)
        acc = (max(left_pos, nl - left_pos) + max(rp, nr - rp)) / m
        if ig > best_g:
            best_g, best_t, best_acc = ig, 0.5 * (vals[i] + vals[i - 1]), acc
    return best_g, best_t, best_acc

def mutual_info_binned(col, labels, nbins=8):
    """MI(feature;label) with equal-frequency bins (nats)."""
    m = len(col)
    order = sorted(range(m), key=lambda i: col[i])
    bin_of = [0] * m
    for rank, i in enumerate(order):
        bin_of[i] = min(nbins - 1, rank * nbins // m)
    # joint counts
    py = [0, 0]
    pb = [0] * nbins
    pjb = [[0, 0] for _ in range(nbins)]
    for i in range(m):
        b = bin_of[i]; l = labels[i]
        py[l] += 1; pb[b] += 1; pjb[b][l] += 1
    mi = 0.0
    for b in range(nbins):
        for l in (0, 1):
            c = pjb[b][l]
            if c == 0:
                continue
            pj = c / m
            mi += pj * math.log(pj / ((pb[b] / m) * (py[l] / m)))
    return mi

# ---- per-feature stump + MI ----
cols = [[X[i][j] for i in range(n)] for j in range(len(FEAT_NAMES))]
per_feat = {}
for j, name in enumerate(FEAT_NAMES):
    ig, t, acc = best_split(cols[j], y)
    mi = mutual_info_binned(cols[j], y)
    per_feat[name] = {"stump_ig": ig, "stump_acc": acc, "mi": mi}

# ================= bagged random forest, Gini importances =================
class Node:
    __slots__ = ("feat", "thr", "left", "right", "pred")

def build_tree(idx, depth, max_depth, min_leaf, mtry, trng, imp):
    node = Node()
    labs = [y[i] for i in idx]
    node.pred = 1 if sum(labs) * 2 >= len(labs) else 0
    if depth >= max_depth or len(idx) < 2 * min_leaf or len(set(labs)) == 1:
        node.feat = None
        return node
    parent_g = gini(labs)
    m = len(idx)
    feat_subset = trng.sample(range(len(FEAT_NAMES)), mtry)
    best = None  # (ig, feat, thr, left_idx, right_idx)
    for j in feat_subset:
        pairs = sorted(((X[i][j], y[i]) for i in idx))
        vals = [v for v, _ in pairs]
        labl = [l for _, l in pairs]
        tot_pos = sum(labl); left_pos = 0
        for k in range(1, m):
            left_pos += labl[k - 1]
            if vals[k] == vals[k - 1] or k < min_leaf or m - k < min_leaf:
                continue
            nl, nr = k, m - k
            gl = 2 * (left_pos / nl) * (1 - left_pos / nl)
            rp = tot_pos - left_pos
            gr = 2 * (rp / nr) * (1 - rp / nr)
            ig = parent_g - (nl * gl + nr * gr) / m
            if best is None or ig > best[0]:
                best = (ig, j, 0.5 * (vals[k] + vals[k - 1]))
    if best is None or best[0] <= 0:
        node.feat = None
        return node
    ig, j, thr = best
    imp[j] += m * ig            # weighted impurity decrease (Gini importance)
    left_idx = [i for i in idx if X[i][j] <= thr]
    right_idx = [i for i in idx if X[i][j] > thr]
    node.feat, node.thr = j, thr
    node.left = build_tree(left_idx, depth + 1, max_depth, min_leaf, mtry, trng, imp)
    node.right = build_tree(right_idx, depth + 1, max_depth, min_leaf, mtry, trng, imp)
    return node

def predict(node, xrow):
    while node.feat is not None:
        node = node.left if xrow[node.feat] <= node.thr else node.right
    return node.pred

N_TREES = 300
MAX_DEPTH = 8
MIN_LEAF = 3
MTRY = max(1, int(round(math.sqrt(len(FEAT_NAMES)))))
frng = random.Random(SEED + 7)
imp = [0.0] * len(FEAT_NAMES)
oob_correct = 0; oob_total = 0
oob_votes = [[0, 0] for _ in range(n)]
for _ in range(N_TREES):
    boot = [frng.randrange(n) for _ in range(n)]
    inbag = set(boot)
    tree = build_tree(boot, 0, MAX_DEPTH, MIN_LEAF, MTRY, frng, imp)
    for i in range(n):
        if i not in inbag:
            oob_votes[i][predict(tree, X[i])] += 1
for i in range(n):
    if sum(oob_votes[i]) > 0:
        pred = 1 if oob_votes[i][1] >= oob_votes[i][0] else 0
        oob_correct += (pred == y[i]); oob_total += 1
imp_sum = sum(imp) or 1.0
rf_imp = {FEAT_NAMES[j]: imp[j] / imp_sum for j in range(len(FEAT_NAMES))}
print(f"RF: trees={N_TREES} depth<={MAX_DEPTH} mtry={MTRY}  OOB_acc={oob_correct/oob_total:.3f}  (base={max(pos,n-pos)/n:.3f})")

# ================= group-only classifier accuracy (explanatory power) =================
def train_tree_full(idx, feat_idxs, depth, max_depth, min_leaf):
    node = Node()
    labs = [y[i] for i in idx]
    node.pred = 1 if sum(labs) * 2 >= len(labs) else 0
    if depth >= max_depth or len(idx) < 2 * min_leaf or len(set(labs)) == 1:
        node.feat = None; return node
    m = len(idx); parent_g = gini(labs)
    best = None
    for j in feat_idxs:
        pairs = sorted(((X[i][j], y[i]) for i in idx))
        vals = [v for v, _ in pairs]; labl = [l for _, l in pairs]
        tot_pos = sum(labl); left_pos = 0
        for k in range(1, m):
            left_pos += labl[k - 1]
            if vals[k] == vals[k - 1] or k < min_leaf or m - k < min_leaf:
                continue
            nl, nr = k, m - k
            gl = 2 * (left_pos / nl) * (1 - left_pos / nl)
            rp = tot_pos - left_pos; gr = 2 * (rp / nr) * (1 - rp / nr)
            ig = parent_g - (nl * gl + nr * gr) / m
            if best is None or ig > best[0]:
                best = (ig, j, 0.5 * (vals[k] + vals[k - 1]))
    if best is None or best[0] <= 0:
        node.feat = None; return node
    _, j, thr = best
    li = [i for i in idx if X[i][j] <= thr]; ri = [i for i in idx if X[i][j] > thr]
    node.feat, node.thr = j, thr
    node.left = train_tree_full(li, feat_idxs, depth + 1, max_depth, min_leaf)
    node.right = train_tree_full(ri, feat_idxs, depth + 1, max_depth, min_leaf)
    return node

# 5-fold CV accuracy using only each group's features
splitrng = random.Random(SEED + 99)
perm = list(range(n)); splitrng.shuffle(perm)
folds = [perm[i::5] for i in range(5)]
def cv_acc(feat_idxs, max_depth=6):
    corr = 0
    for f in range(5):
        test = set(folds[f]); train = [i for i in range(n) if i not in test]
        tr = train_tree_full(train, feat_idxs, 0, max_depth, MIN_LEAF)
        for i in folds[f]:
            corr += (predict(tr, X[i]) == y[i])
    return corr / n
group_acc = {}
for gname, gfeats in GROUPS.items():
    gidx = [FEAT_NAMES.index(fn) for fn in gfeats]
    group_acc[gname] = cv_acc(gidx)
all_acc = cv_acc(list(range(len(FEAT_NAMES))))
base_acc = max(pos, n - pos) / n

# ================= report =================
print("\n=== PER-FEATURE IMPORTANCE (ranked by RF Gini) ===")
print(f"{'feature':<12}{'group':<7}{'RF_imp':>8}{'MI(nats)':>10}{'stump_IG':>10}{'stump_acc':>10}")
ranked = sorted(FEAT_NAMES, key=lambda f: rf_imp[f], reverse=True)
for f in ranked:
    print(f"{f:<12}{GROUP_OF[f]:<7}{rf_imp[f]:>8.3f}{per_feat[f]['mi']:>10.3f}{per_feat[f]['stump_ig']:>10.3f}{per_feat[f]['stump_acc']:>10.3f}")

print("\n=== GROUP SUMMED IMPORTANCE MASS ===")
mi_sum = sum(per_feat[f]['mi'] for f in FEAT_NAMES) or 1.0
print(f"{'group':<7}{'RF_mass':>9}{'MI_mass':>9}{'CVacc(only)':>12}")
for gname in GROUPS:
    rfm = sum(rf_imp[f] for f in GROUPS[gname])
    mim = sum(per_feat[f]['mi'] for f in GROUPS[gname]) / mi_sum
    print(f"{gname:<7}{rfm:>9.3f}{mim:>9.3f}{group_acc[gname]:>12.3f}")
print(f"{'ALL':<7}{1.0:>9.3f}{1.0:>9.3f}{all_acc:>12.3f}   base_acc={base_acc:.3f}")

# ---- machine-readable summary line ----
print("\nSUMMARY_JSON:", {
    "n": n, "base_rate": round(pos / n, 3), "oob_acc": round(oob_correct / oob_total, 3),
    "rf_mass": {g: round(sum(rf_imp[f] for f in GROUPS[g]), 3) for g in GROUPS},
    "mi_mass": {g: round(sum(per_feat[f]['mi'] for f in GROUPS[g]) / mi_sum, 3) for g in GROUPS},
    "cv_acc_only": {g: round(group_acc[g], 3) for g in GROUPS}, "cv_acc_all": round(all_acc, 3),
    "top5_rf": [(f, round(rf_imp[f], 3)) for f in ranked[:5]],
})
