"""EXPERIMENT E - HARDWARE CAUSAL TEST (size-vs-capacity, not raw dims).

Hypothesis: the FUSE/unfuse verdict is governed by absolute size RELATIVE TO hardware
capacity (dimensionless ratios ~1), not by raw dims. Causal test: DOUBLE the L2 and the
per-block SMEM. If the lever is size-vs-capacity, every boundary must translate to ~2x
larger dims for the SAME shape. If it were raw dims (capacity-independent), boundaries
would not move.

Two capacities are doubled:
  L2  : governs weight residency  -> b_big/d_big -> FUSE<->unfuse flip
  SMEM: governs held row-block    -> m0_max      -> feasible<->infeasible flip

We report boundaries three ways:
  (A) 4 shapes, uniform power-of-2 scaling (lambda on ALL dims) -> literal 'same shape' test.
  (B) single-lever sweeps that make the binding capacity ratio LINEAR, so the 2x is crisp:
        B1: sweep K1  -> f_B = K1*N1*2/eff_L2 crosses 1  (weight-B L2 flip)
        B2: sweep N2  -> f_D = N1*N2*2/eff_L2 crosses 1  (weight-D L2 flip)
        B3: sweep N   (N1=N2=N) -> f_held/m0_max crosses feasibility (SMEM)
All timing via the shared probe (snowcat roofline). power-of-2 dims in [256, 65536], BPE=2.
"""
from __future__ import annotations
import dataclasses, math
from gemm_time_estimator import GPUS
from chain_gemm_fusion import _eff_l2
from chain_gemm_probe import probe

H = GPUS["h100-sxm"]
BIG = dataclasses.replace(H, l2_bytes=H.l2_bytes * 2, smem_per_block_bytes=H.smem_per_block_bytes * 2)
GPU = {"h100": H, "big2x": BIG}
NCALLS = 0

def P(M, K1, N1, N2, g):
    global NCALLS
    NCALLS += 1
    return probe(M, K1, N1, N2, GPU[g])

def w(p):
    return "F" if p["winner"] == "FUSE" else ("u" if p["winner"] == "unfuse" else "X")

print(f"eff_L2  MB:  h100={_eff_l2(H)/1e6:.2f}  big2x={_eff_l2(BIG)/1e6:.2f}")
print(f"SMEM/blk B:  h100={H.smem_per_block_bytes}  big2x={BIG.smem_per_block_bytes}\n")

# ---------------------------------------------------------------- helpers
def last_fuse_and_feas(rows):
    """rows: list of (lever_value, winner_char) in increasing lever order.
    Returns (last_lever_with_FUSE, first_lever_that_is_infeasible)."""
    last_fuse = None
    first_infeas = None
    for v, ch in rows:
        if ch == "F":
            last_fuse = v
        if ch == "X" and first_infeas is None:
            first_infeas = v
    return last_fuse, first_infeas

# ================================================================ (A) UNIFORM SCALING, 4 SHAPES
# each shape = base log2 dims (M,K1,N1,N2); scale by 2^t on ALL dims (t integer).
# chosen so that at small scale it is FUSE-feasible and scaling up -> unfuse -> infeasible.
SHAPES = {
    # name          M   K1  N1  N2   (log2)   character
    "B-heavy":     (13,  9, 12,  7),   # K1*N1 crosses L2 first ; N2 tiny keeps feasible
    "D-heavy":     (12,  8, 11, 11),   # N1*N2 crosses L2 ; wide output
    "square":      (12, 10, 10,  9),   # balanced
    "C-heavy":     (14,  9, 11,  8),   # tall M, big intermediate C
}
TS = range(-4, 6)  # scale exponent t

print("=" * 108)
print("(A) UNIFORM SCALING per shape  (dims = 2^(base+t) on ALL four dims)   F=FUSE u=unfuse X=infeasible")
print("=" * 108)
uniform_summary = []
for name, base in SHAPES.items():
    print(f"\nshape {name}: base log2 (M,K1,N1,N2)={base}")
    print(f"  {'t':>3} {'M':>6} {'K1':>6} {'N1':>6} {'N2':>6} | {'geo':>7} | "
          f"{'h100':>4} {'sp':>6} f_B/f_D/f_held | {'big2x':>5} {'sp':>6} f_B/f_D/f_held")
    rows = {"h100": [], "big2x": []}
    geo_of = {}
    for t in TS:
        dims = [2 ** (b + t) for b in base]
        if min(dims) < 256 or max(dims) > 65536:
            continue
        M, K1, N1, N2 = dims
        geo = (M * K1 * N1 * N2) ** 0.25
        geo_of[t] = geo
        cells = {}
        for g in ("h100", "big2x"):
            p = P(M, K1, N1, N2, g)
            cells[g] = p
            rows[g].append((t, w(p)))
        ph, pb = cells["h100"], cells["big2x"]
        print(f"  {t:>3} {M:>6} {K1:>6} {N1:>6} {N2:>6} | {geo:>7.0f} | "
              f"{w(ph):>4} {ph['speedup']:>6.3f} {ph['f_B']:.2f}/{ph['f_D']:.2f}/{ph['f_held']:.2f} | "
              f"{w(pb):>5} {pb['speedup']:>6.3f} {pb['f_B']:.2f}/{pb['f_D']:.2f}/{pb['f_held']:.2f}")
    # boundaries in t (log2), convert shift to per-dim factor
    lf_h, fi_h = last_fuse_and_feas(rows["h100"])
    lf_b, fi_b = last_fuse_and_feas(rows["big2x"])
    def geo_at(t):
        return geo_of.get(t)
    uniform_summary.append((name, lf_h, lf_b, fi_h, fi_b, geo_of))
    def sf(th, tb):
        if th is None or tb is None:
            return "n/a"
        return f"{2.0 ** (tb - th):.2f}x"
    print(f"   -> FUSE->unfuse boundary  last-FUSE t: h100={lf_h} big2x={lf_b}  per-dim shift={sf(lf_h, lf_b)}"
          + (f"  (geo {geo_at(lf_h):.0f}->{geo_at(lf_b):.0f})" if lf_h is not None and lf_b is not None else ""))
    print(f"   -> feasibility boundary   first-infeas t: h100={fi_h} big2x={fi_b}  per-dim shift={sf(fi_h, fi_b)}"
          + (f"  (geo {geo_at(fi_h):.0f}->{geo_at(fi_b):.0f})" if fi_h is not None and fi_b is not None else ""))

# ================================================================ (B) SINGLE-LEVER (crisp 2x)
def sweep_lever(title, gen, values, lever_name):
    """gen(v)->(M,K1,N1,N2). Print winner vs lever on both GPUs; return boundaries dict."""
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    print(f"  {lever_name:>7} {'M':>6} {'K1':>6} {'N1':>6} {'N2':>6} | "
          f"{'h100 win':>8} {'sp':>6} {'f_B':>5} {'f_D':>5} {'f_held':>6} | "
          f"{'big win':>8} {'sp':>6} {'f_B':>5} {'f_D':>5} {'f_held':>6}")
    rows = {"h100": [], "big2x": []}
    for v in values:
        M, K1, N1, N2 = gen(v)
        cs = {}
        for g in ("h100", "big2x"):
            p = P(M, K1, N1, N2, g)
            cs[g] = p
            rows[g].append((v, w(p)))
        ph, pb = cs["h100"], cs["big2x"]
        print(f"  {v:>7} {M:>6} {K1:>6} {N1:>6} {N2:>6} | "
              f"{ph['winner']:>8} {ph['speedup']:>6.3f} {ph['f_B']:>5.2f} {ph['f_D']:>5.2f} {ph['f_held']:>6.3f} | "
              f"{pb['winner']:>8} {pb['speedup']:>6.3f} {pb['f_B']:>5.2f} {pb['f_D']:>5.2f} {pb['f_held']:>6.3f}")
    lf_h, fi_h = last_fuse_and_feas(rows["h100"])
    lf_b, fi_b = last_fuse_and_feas(rows["big2x"])
    def shift(a, b):
        if a is None or b is None:
            return "n/a"
        return f"{b / a:.2f}x"
    print(f"   -> last-FUSE {lever_name}: h100={lf_h} big2x={lf_b}  shift={shift(lf_h, lf_b)}")
    print(f"   -> first-INFEAS {lever_name}: h100={fi_h} big2x={fi_b}  shift={shift(fi_h, fi_b)}")
    return {"last_fuse": (lf_h, lf_b), "first_infeas": (fi_h, fi_b)}

# B1: sweep K1 -> weight-B L2 flip (f_B = K1*N1*2/eff_L2 crosses 1). N1=4096, N2=128, M=8192.
b1 = sweep_lever(
    "(B1) L2 lever: sweep K1  (weight B = K1xN1).  Fixed M=8192 N1=4096 N2=128.  Expect last-FUSE 2x on big.",
    lambda K1: (8192, K1, 4096, 128),
    [256, 512, 1024, 2048, 4096, 8192, 16384],
    "K1")

# B2: sweep N2 -> weight-D L2 flip (f_D = N1*N2*2/eff_L2 crosses 1). N1=2048 (feasible always), K1=512, M=8192.
b2 = sweep_lever(
    "(B2) L2 lever: sweep N2  (weight D = N1xN2).  Fixed M=8192 K1=512 N1=2048.  Expect last-FUSE 2x on big.",
    lambda N2: (8192, 512, 2048, N2),
    [256, 512, 1024, 2048, 4096, 8192, 16384],
    "N2")

# B3: sweep N (N1=N2=N) -> SMEM feasibility (m0_max<16). K1=1024, M=8192.
b3 = sweep_lever(
    "(B3) SMEM lever: sweep N=N1=N2  (held row-block = min(N1,N2)).  Fixed M=8192 K1=1024.  Expect infeas edge 2x.",
    lambda N: (8192, 1024, N, N),
    [512, 1024, 2048, 4096, 8192, 16384],
    "N")

# ================================================================ SUMMARY
print("\n" + "=" * 108)
print("SUMMARY OF BOUNDARY SHIFTS  (h100 -> big2x)")
print("=" * 108)
print(f"{'test':<34}{'boundary':<20}{'h100':>10}{'big2x':>10}{'shift':>8}")
def line(tag, bnd, a, b):
    s = f"{b/a:.2f}x" if (a and b) else "n/a"
    print(f"{tag:<34}{bnd:<20}{a if a else '-':>10}{b if b else '-':>10}{s:>8}")
line("B1 sweep K1 (weight-B L2)", "last-FUSE K1", *b1["last_fuse"])
line("B2 sweep N2 (weight-D L2)", "last-FUSE N2", *b2["last_fuse"])
line("B3 sweep N  (SMEM feas)", "last-feasible N", (b3["first_infeas"][0] // 2 if b3["first_infeas"][0] else None),
     (b3["first_infeas"][1] // 2 if b3["first_infeas"][1] else None))
line("B3 sweep N  (SMEM feas)", "first-infeas N", *b3["first_infeas"])
print(f"\nTotal probe() calls: {NCALLS}")
