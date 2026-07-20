"""EXPERIMENT A - SCALE-INVARIANCE (isolate SIZE from SHAPE).

If the fuse/unfuse verdict depended ONLY on shape (aspect ratios), it would be
scale-invariant: multiply all four dims by a common lambda=2^k (shape fixed:
aA_log2, aB_log2, aD_log2 constant) and the verdict must not change.

We take 6 FIXED shapes (fixed aspect ratios) and scale all four dims by lambda
over the widest feasible power-of-2 range in [256,65536] (min(N1,N2)<=4096 for
feasibility; M is always a power of 2 so divisible). At each (shape,lambda) we
record winner, speedup, f_held, f_B, f_D, f_C, bott, feasibility, and flag flips.
"""
from __future__ import annotations
import math
from chain_gemm_probe import probe
from gemm_time_estimator import GPUS

G = GPUS["h100-sxm"]

# Each shape = base tuple at the SMALLEST scale (min dim = 256), fixed aspect ratios.
# ratios shown as M:K1:N1:N2.  We scale by lambda=2^k upward until infeasible / dim>65536.
SHAPES = [
    ("flash-attn  16:1:16:1 (wideN1,narrowN2)", (4096, 256, 4096, 256)),
    ("square       1:1:1:1",                     (256,  256, 256,  256)),
    ("tall-A      16:1:1:1  (M large)",          (4096, 256, 256,  256)),
    ("wide-B       1:1:16:1 (N1 wide)",          (256,  256, 4096, 256)),
    ("big-weight   1:16:16:1(K1*N1 huge)",       (256,  4096,4096, 256)),
    ("wide-D       1:1:1:16 (N2 large)",         (256,  256, 256,  4096)),
]

LAMBDAS = [1, 2, 4, 8, 16, 32]  # 2^k; we stop a shape once a dim would exceed 65536.
MAXDIM = 65536
MINDIM = 256


def scaled(base, lam):
    return tuple(d * lam for d in base)


def in_range(dims):
    return all(MINDIM <= d <= MAXDIM for d in dims)


def run():
    print(f"# EXPERIMENT A - scale-invariance test on {G.name if hasattr(G,'name') else 'h100-sxm'}")
    print(f"# feasibility needs min(N1,N2)<=~6752 (=>4096 feasible, 8192 infeasible)")
    print(f"# f_B>1 <=> weight B re-read from DRAM (K1*N1*2>30MB); f_D>1 <=> weight D from DRAM\n")

    summary = []  # (shape_name, seq, flips, flip_info)
    total_calls = 0

    for name, base in SHAPES:
        print("=" * 118)
        print(f"SHAPE {name}   base(M,K1,N1,N2)={base}")
        # confirm shape (aspect ratios) is constant: print log2 aspects from base
        M, K1, N1, N2 = base
        print(f"   aA_log2=log2(M/K1)={math.log2(M/K1):+.0f}  aB_log2=log2(K1/N1)={math.log2(K1/N1):+.0f}  aD_log2=log2(N1/N2)={math.log2(N1/N2):+.0f}   (FIXED across scale)")
        print(f"   {'lam':>4} {'M':>6} {'K1':>6} {'N1':>6} {'N2':>6} | {'winner':>10} {'speedup':>8} | "
              f"{'f_held':>7} {'f_B':>7} {'f_D':>7} {'f_C':>7} | {'AI':>6} {'ridge':>5} | {'bott':>7} {'feas':>5}")
        seq = []          # list of (lam, winner, feasible, ratios)
        for lam in LAMBDAS:
            dims = scaled(base, lam)
            if not in_range(dims):
                # still probe the FIRST out-of-feasible-min case to show infeasibility if only min(N1,N2) grows,
                # but if any dim exceeds MAXDIM we cannot form a legal problem -> skip (report as dim-cap).
                if any(d > MAXDIM for d in dims):
                    print(f"   {lam:>4} {dims[0]:>6} {dims[1]:>6} {dims[2]:>6} {dims[3]:>6} |  (dim>65536: out of legal range, stop)")
                    break
            M, K1, N1, N2 = dims
            p = probe(M, K1, N1, N2, G)
            total_calls += 1
            sp = p["speedup"]
            sp_s = f"{sp:.3f}" if math.isfinite(sp) else "  nan"
            ai = p["AI_fused"]
            ai_s = f"{ai:.0f}" if math.isfinite(ai) else "nan"
            print(f"   {lam:>4} {M:>6} {K1:>6} {N1:>6} {N2:>6} | {p['winner']:>10} {sp_s:>8} | "
                  f"{p['f_held']:>7.3f} {p['f_B']:>7.2f} {p['f_D']:>7.2f} {p['f_C']:>7.2f} | "
                  f"{ai_s:>6} {p['ridge']:>5.0f} | {p['bott']:>7} {str(p['feasible']):>5}")
            seq.append((lam, p["winner"], p["feasible"],
                        dict(f_held=p["f_held"], f_B=p["f_B"], f_D=p["f_D"], f_C=p["f_C"])))
            # stop after the first infeasible point (scaling further only stays infeasible)
            if not p["feasible"]:
                break

        # ---- flip analysis for this shape ----
        winners = [w for (_, w, _, _) in seq]
        # fuse-preference flip: change among FUSE/unfuse (ignoring infeasible tail for the *preference* flip)
        pref = [(lam, w, r) for (lam, w, feas, r) in seq if w in ("FUSE", "unfuse")]
        flip_events = []
        for i in range(1, len(pref)):
            if pref[i][1] != pref[i - 1][1]:
                lam0, w0, r0 = pref[i - 1]
                lam1, w1, r1 = pref[i]
                flip_events.append((lam0, w0, lam1, w1, r0, r1))
        # feasibility loss event
        feas_loss = None
        for i in range(1, len(seq)):
            if seq[i - 1][2] and not seq[i][2]:
                feas_loss = (seq[i - 1][0], seq[i][0])
        seq_str = " -> ".join(f"{lam}:{w}" for (lam, w, _, _) in seq)
        print(f"   VERDICT SEQUENCE (by lambda): {seq_str}")
        if flip_events:
            for (lam0, w0, lam1, w1, r0, r1) in flip_events:
                print(f"   *** FUSE-PREF FLIP {w0}->{w1} between lambda {lam0} and {lam1}: "
                      f"f_B {r0['f_B']:.2f}->{r1['f_B']:.2f}, f_D {r0['f_D']:.2f}->{r1['f_D']:.2f}, "
                      f"f_C {r0['f_C']:.2f}->{r1['f_C']:.2f}, f_held {r0['f_held']:.3f}->{r1['f_held']:.3f}")
        else:
            print(f"   *** NO fuse-preference flip (verdict constant across feasible scale)")
        if feas_loss:
            print(f"   *** FEASIBILITY LOST between lambda {feas_loss[0]} and {feas_loss[1]} (min(N1,N2)>~6752)")
        flipped = len(flip_events) > 0
        summary.append((name, seq_str, flipped, flip_events, feas_loss))
        print()

    # ---------- overall summary ----------
    print("=" * 118)
    print("SUMMARY  (SHAPE fixed within each row; only SIZE=lambda varies)")
    n_flip = 0
    for (name, seq_str, flipped, flip_events, feas_loss) in summary:
        tag = "FLIPS" if flipped else "no-flip"
        if flipped:
            n_flip += 1
            fe = flip_events[0]
            drv = "f_B~1" if (fe[4]['f_B'] < 1 <= fe[5]['f_B']) else ("f_D~1" if (fe[4]['f_D'] < 1 <= fe[5]['f_D']) else "other")
            extra = f"  (driver: {drv} at lambda {fe[0]}->{fe[2]})"
        else:
            extra = ""
        print(f"  {tag:>7} | {name:<42} | {seq_str}{extra}")
    print(f"\n  ==> {n_flip} of {len(SHAPES)} shapes FLIP fuse-preference as scale (lambda) grows, at FIXED shape.")
    print(f"  ==> If the verdict were shape-only it would be scale-INVARIANT (0 flips).")
    print(f"  ==> total probe() calls = {total_calls}")


if __name__ == "__main__":
    run()
