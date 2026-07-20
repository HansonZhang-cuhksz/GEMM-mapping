"""Experiment 2: partition structure + the user's literal question.

(a) mem-bound (w=128, w=256), L=6: total time ~linear in NUMBER of segments and
    ~independent of WHERE the cuts are. Group partitions by segment-count, report
    mean/spread. Quantify per-cut cost in ms and MB (one intermediate round-trip).
(b) L=3 literal question: fuse-all-3 vs fuse-[1,2]+3 vs fuse-1+[2,3] vs fully unfused.
(c) generalize the rule (is fewest-segments always optimal in mem-bound regime?).
Then repeat (b) for compute-bound w=1024 to show the four times ~coincide.
"""
from __future__ import annotations
import math, statistics
from collections import defaultdict
from multi_gemm_fusion import analyze, partition_time
from gemm_time_estimator import GPUS
from chain_gemm_fusion import BPE, _eff_l2

GPU = GPUS['h100-sxm']
M = 131072

def ms(t): return t * 1e3 if math.isfinite(t) else float('inf')

def nseg(cuts): return len(cuts) + 1

def group_by_segcount(a):
    """Return dict seg_count -> list of times (ms), plus per-partition detail."""
    groups = defaultdict(list)
    for t, cuts, segs in a['results']:
        if math.isfinite(t):
            groups[nseg(cuts)].append(t * 1e3)
    return groups

def part_a(w):
    print(f"\n########## (a) mem-bound w={w}, L=6 — group by segment count ##########")
    a = analyze(M, w, 6, GPU)
    Ci_mib = M * w * BPE / 2**20
    print(f"eff-L2={_eff_l2(GPU)/2**20:.0f}MiB  C_i=M*w*2={Ci_mib:.1f}MiB "
          f"({'spills' if M*w*BPE>_eff_l2(GPU) else 'fits'} L2)")
    groups = group_by_segcount(a)
    print(f"{'#seg':>4} {'n_part':>6} {'mean_ms':>9} {'min_ms':>9} {'max_ms':>9} {'spread%':>8}")
    means = {}
    for s in sorted(groups):
        v = groups[s]
        mean = statistics.mean(v); lo = min(v); hi = max(v)
        spread = (hi - lo) / mean * 100
        means[s] = mean
        print(f"{s:>4} {len(v):>6} {mean:>9.4f} {lo:>9.4f} {hi:>9.4f} {spread:>7.2f}%")
    # per-cut cost from slope of mean vs seg-count
    ks = sorted(means)
    diffs = [means[k+1] - means[k] for k in ks if (k+1) in means]
    per_cut_ms = statistics.mean(diffs)
    print(f"\nper-cut (per-added-segment) cost: diffs_ms={[f'{d:.4f}' for d in diffs]}")
    print(f"  mean per-cut = {per_cut_ms:.4f} ms")
    # MB interpretation: a cut writes+reads one intermediate C_i
    roundtrip_mib = 2 * Ci_mib
    print(f"  one intermediate C_i={Ci_mib:.1f}MiB; round-trip(write+read)=2*C_i={roundtrip_mib:.1f}MiB")
    print(f"  implied BW = roundtrip/per_cut = {roundtrip_mib*2**20/(per_cut_ms*1e-3)/1e12:.2f} TB/s "
          f"(HBM peak={GPU.bw_bytes_per_s/1e12:.2f} TB/s)")
    # linearity check: fit time = base + slope*(#seg-1)
    base = means[1]
    pred = {s: base + per_cut_ms*(s-1) for s in ks}
    maxerr = max(abs(means[s]-pred[s])/means[s]*100 for s in ks)
    print(f"  linear fit t(ms)={base:.4f}+{per_cut_ms:.4f}*(#seg-1); max rel err vs group-mean={maxerr:.2f}%")
    return a, per_cut_ms, roundtrip_mib

def seg_dram(a, cuts):
    """Sum DRAM MiB across segments for a given partition, to measure real HBM traffic.
    length-1 (plain unfused) segments don't report dram_mib -> compute analytically:
    read input C_in (M*w_in) + read weight (w_in*w_out) + write output C_out (M*w_out)."""
    widths = a['widths']
    t, segs = partition_time(widths, M, GPU, cuts)
    tot = 0.0
    for (g0, g1, _, info) in segs:
        if 'dram_mib' in info:
            tot += info['dram_mib']
        else:  # k==1 length-1 segment: gemm g0 uses widths[g0-1]->widths[g0]
            w_in, w_out = widths[g0-1], widths[g0]
            tot += (M*w_in + w_in*w_out + M*w_out) * BPE / 2**20
    return t*1e3, tot

def part_a_dram(a, w):
    """Show actual aggregated DRAM MiB grows by ~one round-trip per cut."""
    print(f"\n--- (a) actual HBM traffic (sum of segment dram_mib) w={w} ---")
    print(f"{'#seg':>4} {'cuts':>14} {'time_ms':>9} {'HBM_MiB':>9}")
    # pick the fuse-all + a representative chain of nested cuts
    reps = [(), (3,), (2,4), (2,3,4)] if False else None
    # one representative per seg-count (contiguous cuts)
    by_seg = {}
    for t, cuts, segs in a['results']:
        s = nseg(cuts)
        if s not in by_seg:
            by_seg[s] = cuts
    prev_mib = None
    for s in sorted(by_seg):
        cuts = by_seg[s]
        tm, mib = seg_dram(a, cuts)
        d = "" if prev_mib is None else f"  (+{mib-prev_mib:.1f})"
        print(f"{s:>4} {str(cuts or '()'):>14} {tm:>9.4f} {mib:>9.1f}{d}")
        prev_mib = mib

def part_b(w, label):
    print(f"\n########## (b) L=3 literal question — {label} w={w} ##########")
    a = analyze(M, w, 3, GPU)
    Ci_mib = M * w * BPE / 2**20
    print(f"C_i={Ci_mib:.1f}MiB ({'spills' if M*w*BPE>_eff_l2(GPU) else 'fits'} L2)")
    named = {
        (): "fuse-ALL-3        [G1,G2,G3]",
        (2,): "fuse-[1,2]+3      [G1,G2]|[G3]",
        (1,): "fuse-1+[2,3]      [G1]|[G2,G3]",
        (1,2): "fully UNFUSED     [G1]|[G2]|[G3]",
    }
    rows = []
    for cuts, name in named.items():
        r = next(x for x in a['results'] if x[1] == cuts)
        rows.append((name, cuts, ms(r[0])))
    fa = next(v for n,c,v in rows if c==())
    print(f"{'partition':<34} {'#seg':>4} {'time_ms':>9} {'vs_fuseall':>10}")
    for name, cuts, v in rows:
        rel = v/fa if math.isfinite(v) and fa else float('nan')
        print(f"{name:<34} {nseg(cuts):>4} {v:>9.4f} {rel:>9.3f}x")
    best = min(rows, key=lambda r: r[2])
    print(f"  BEST = {best[0].split('  ')[0].strip()}  ({best[2]:.4f} ms)")
    return rows

if __name__ == "__main__":
    # (a) + per-cut cost for the two mem-bound widths
    a128, pc128, rt128 = part_a(128)
    part_a_dram(a128, 128)
    a256, pc256, rt256 = part_a(256)
    part_a_dram(a256, 256)
    # (b) mem-bound L=3
    part_b(128, "MEM-BOUND")
    part_b(256, "MEM-BOUND")
    # contrast: compute-bound L=3
    part_b(1024, "COMPUTE-BOUND")
    # also L=6 compute-bound group summary for the rule
    print("\n########## (c) contrast: compute-bound w=1024 L=6 group-by-segcount ##########")
    ac = analyze(M, 1024, 6, GPU)
    g = group_by_segcount(ac)
    print(f"{'#seg':>4} {'n_part':>6} {'mean_ms':>9} {'min_ms':>9} {'max_ms':>9}")
    for s in sorted(g):
        v=g[s]; print(f"{s:>4} {len(v):>6} {statistics.mean(v):>9.4f} {min(v):>9.4f} {max(v):>9.4f}")
    bt,bc,_ = ac['best']; fa = next(r for r in ac['results'] if r[1]==())
    print(f"  best cuts={bc or '()'} {ms(bt):.4f}ms ; fuse-all {ms(fa[0]):.4f}ms ; "
          f"fuse-all optimal? {bc==()}")
