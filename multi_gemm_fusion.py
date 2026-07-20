"""Multi-GEMM chain fusion: is fusing MORE stages better than fusing fewer + leaving some split?

Chain of L GEMMs, uniform narrow widths (the fusion-preferring regime from the shape-vs-size
study): X[M,w] @ W1[w,w] @ ... @ WL[w,w] -> Y[M,w]. Large M so every intermediate C_i = M*w
spills eff-L2 (an unfused round-trip worth avoiding); narrow w so weights are tiny/L2-resident
and the row-block stays feasible.

A "partition" splits the L contiguous stages into fused SEGMENTS; boundaries between segments
are materialized to HBM (a round-trip). partition = set of internal cut positions (2^(L-1) of
them). No cuts = fuse-all (1 kernel); all cuts = fully unfused (L kernels).

Fused-segment model (the confirmed generalization of the validated 2-GEMM fused_time):
  * process m0 rows at a time; the segment's intermediates stay resident in SMEM. Block-
    sequential: hold two adjacent activations at once -> peak SMEM = m0 * max_s(K_s + K_{s+1}) * bpe.
    m0 <= (SMEM_block - overhead) / (peak_pair * bpe); if < 16-row MMA min -> INFEASIBLE.
  * mt = M/m0 row-blocks; each block re-reads every weight in the segment once -> weight W_s
    re-read mt x (from L2 if K_{s-1}*K_s*bpe <= eff-L2 else DRAM). Input activation read once
    total, output written once total, internal activations never touch HBM.
  * time = estimator occupancy-roofline(total ops, aggregated DRAM, output tiles, resident, C).
  A length-1 "segment" is just the plain unfused GEMM (via estimate_gemm_time): reads input,
  writes output, no re-read, no round-trip saving. So a cut boundary's activation is written by
  one segment and read by the next = the HBM round-trip (valid here since every C_i > eff-L2).

All GEMMs are timed by the snowcat-roofline estimator, exactly as in chain_gemm_fusion.py.

Scope caveats (from the model audit — the headlines are robust but bounded):
  * Residency uses peak_pair = 2w (hold two adjacent activations), MORE CONSERVATIVE than the
    2-GEMM model's hold=min(N1,N2)=w. It only sets the feasibility cap (w>=4096 infeasible) and a
    latency term that never binds here; feasible-case timings are unchanged. So this generalizes,
    but does NOT literally reproduce, the 2-GEMM fused_time.
  * For UNIFORM widths the fused weight re-read cost (w_big = w*w*2 > eff-L2) is inert for every
    FEASIBLE segment (w_big only at w>=4096, which is always infeasible). So feasible fused
    segments carry ~zero modeled cost beyond the saved round-trips -> fuse-all is optimal-or-tied
    by construction for uniform-narrow chains. A genuine "fuse-fewer-wins" crossover needs
    NON-UNIFORM / large weights (some segment weight > eff-L2). This module answers the uniform case.
  * Assumes each intermediate C_i > eff-L2 (spills), so a cut is a full HBM round-trip with no
    inter-kernel L2 credit. Enforced with a warning below; violated only for small M*w.

Run:  conda run -n area python multi_gemm_fusion.py --L 6 --w 128 --M 131072 --verbose
"""

from __future__ import annotations

import argparse
import itertools
import math

from gemm_time_estimator import GPUS, GpuModel, optimal_mapping_by_time
from snowcat_demo.model.workload import divisors
from chain_gemm_fusion import BPE, MMA_MIN_M, STREAM_OVERHEAD, _dram_of, _eff_l2, _roofline

TIE_TOL = 1e-6   # partitions within this rel-tol of the fastest are treated as ties (compute-bound)


def segment_time(widths: list[int], M: int, gpu: GpuModel) -> tuple[float, dict]:
    """Time one fused segment spanning GEMMs with boundary `widths` = [K_in, ..., K_out]."""
    k = len(widths) - 1                      # number of GEMMs in this segment
    if k == 1:                               # length-1 segment == plain unfused GEMM
        _, e = optimal_mapping_by_time(M, widths[1], widths[0], gpu, l2=True)
        return e.time_s, {"m0": M, "mt": 1, "k": 1, "bott": "gemm",
                          "tile": f"{e.mapping.bm}x{e.mapping.bn}x{e.mapping.bk}"}
    eff_l2 = _eff_l2(gpu)
    peak_pair = max(widths[s] + widths[s + 1] for s in range(k))      # SMEM residency driver
    m0_max = (gpu.smem_per_block_bytes - STREAM_OVERHEAD) // (peak_pair * BPE)
    if m0_max < MMA_MIN_M:
        return float("inf"), {"why": f"INFEASIBLE: peak_pair={peak_pair}, m0_max={m0_max}<{MMA_MIN_M}", "k": k}
    ops = sum(2 * M * widths[s] * widths[s + 1] for s in range(k))
    best = None
    for m0 in divisors(M):
        if m0 < MMA_MIN_M or m0 > m0_max:
            continue
        mt = M // m0
        dram = 0.0
        last_bn = last_c = 1
        for s in range(k):
            w_in, w_out = widths[s], widths[s + 1]
            _, eb = optimal_mapping_by_time(m0, w_out, w_in, gpu, l2=True)   # A[m0,w_in] @ W[w_in,w_out]
            w_big = (w_in * w_out * BPE) > eff_l2
            dram += (mt if w_big else 1) * _dram_of(eb, "W")                 # weight re-read
            if s == 0:
                dram += mt * _dram_of(eb, "A")                              # input read once total
            if s == k - 1:
                dram += mt * _dram_of(eb, "OUT")                            # output written once total
                last_bn, last_c = max(eb.mapping.bn, 1), eb.num_stages
        out_tiles = mt * max(1, widths[-1] // last_bn)
        resident = m0 * peak_pair * BPE + STREAM_OVERHEAD
        t, bott = _roofline(gpu, ops, dram, out_tiles, resident, last_c)
        if best is None or t < best[0]:
            best = (t, {"m0": m0, "mt": mt, "k": k, "bott": bott, "dram_mib": dram / 2**20})
    return best if best else (float("inf"), {"why": "no m0 fits", "k": k})


def partition_time(widths: list[int], M: int, gpu: GpuModel, cuts: tuple[int, ...]) -> tuple[float, list]:
    """Total time for a partition given internal cut positions (1..L-1)."""
    L = len(widths) - 1
    bounds = [0] + list(cuts) + [L]
    total, segs = 0.0, []
    for a, b in zip(bounds, bounds[1:]):
        t, info = segment_time(widths[a:b + 1], M, gpu)
        total += t
        segs.append((a + 1, b, t, info))          # GEMMs a+1..b (1-indexed)
    return total, segs


def all_partitions(L: int):
    for r in range(L):                             # r cuts
        for cuts in itertools.combinations(range(1, L), r):
            yield cuts


def analyze(M: int, w: int, L: int, gpu: GpuModel, verbose: bool = False) -> dict:
    if M * w * BPE <= _eff_l2(gpu):
        print(f"  [warn] C_i=M*w*2={M*w*BPE/2**20:.0f}MiB <= eff-L2 {_eff_l2(gpu)/2**20:.0f}MiB: "
              f"cut boundaries would stay in L2; the 'cut=HBM round-trip' assumption over-penalizes "
              f"splits here (use larger M*w).")
    widths = [w] * (L + 1)                          # uniform: X[M,w] @ (w,w) xL -> [M,w]
    results = []
    for cuts in all_partitions(L):
        t, segs = partition_time(widths, M, gpu, cuts)
        results.append((t, cuts, segs))
    feasible = [r for r in results if math.isfinite(r[0])]
    feasible.sort(key=lambda r: r[0])
    # Tie-break: among partitions within TOL of the fastest (compute-bound cells are exact ties
    # to ~1 ULP), prefer the FEWEST cuts (fuse-all). Avoids reporting float-noise as a "winner".
    if feasible:
        tmin = feasible[0][0]
        near = [r for r in feasible if r[0] <= tmin * (1 + TIE_TOL)]
        best = min(near, key=lambda r: len(r[1]))
    else:
        best = None
    fuse_all = next(r for r in results if r[1] == ())
    unfused = next(r for r in results if r[1] == tuple(range(1, L)))
    # longest fused segment length in each partition
    def maxseg(cuts):
        bounds = [0] + list(cuts) + [L]
        return max(b - a for a, b in zip(bounds, bounds[1:]))
    return {"M": M, "w": w, "L": L, "widths": widths, "results": results,
            "best": best, "fuse_all": fuse_all, "unfused": unfused, "maxseg": maxseg}


def print_report(a: dict, gpu: GpuModel, verbose: bool = False) -> None:
    L, w, M = a["L"], a["w"], a["M"]
    print(f"\n===== Multi-GEMM chain fusion — {gpu.name} =====")
    print(f"L={L} GEMMs, uniform w={w}, M={M}. eff-L2={_eff_l2(gpu)/2**20:.0f}MiB, "
          f"C_i=M*w*2={M*w*BPE/2**20:.0f}MiB ({'spills' if M*w*BPE>_eff_l2(gpu) else 'fits'} L2).")
    fa_t = a["fuse_all"][0]; uf_t = a["unfused"][0]; best = a["best"]
    def ms(t): return f"{t*1e3:.4f}" if math.isfinite(t) else "INFEAS"
    print(f"  fuse-ALL ({L} stages): {ms(fa_t)} ms   |   fully UNFUSED: {ms(uf_t)} ms   "
          f"|   speedup(unfused/fuseall)={uf_t/fa_t:.3f}x" if math.isfinite(fa_t) else f"  fuse-ALL INFEAS")
    if best:
        bt, bcuts, _ = best
        print(f"  BEST partition: cuts={bcuts or 'none (fuse-all)'}  {ms(bt)} ms  "
              f"(max fused segment = {a['maxseg'](bcuts)} stages)")
        print(f"  fuse-all is {'OPTIMAL' if bcuts==() else f'NOT optimal (best beats it by {fa_t/bt:.3f}x)'}")
    if verbose:
        print("  all partitions (sorted):")
        for t, cuts, segs in sorted(a["results"], key=lambda r: (not math.isfinite(r[0]), r[0])):
            seglens = [b - (aa - 1) for aa, b, _, _ in segs]
            print(f"    {ms(t):>9} ms  segments={seglens}  cuts={cuts or '()'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=sorted(GPUS), default="h100-sxm")
    ap.add_argument("--M", type=int, default=131072)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    print_report(analyze(args.M, args.w, args.L, GPUS[args.gpu]), GPUS[args.gpu], args.verbose)


if __name__ == "__main__":
    main()
