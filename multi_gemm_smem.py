"""Multi-GEMM fusion WITH the SMEM-competition effect (tests the user's hypothesis).

Snowcat: a GEMM's tile pipeline depth C = SMEM/W caps effective bandwidth (bw_latency =
active_sm*C*W/latency), and larger SMEM -> bigger tiles / deeper pipeline -> faster (until BW
saturates). In a FUSED kernel the resident intermediate activations occupy SMEM, so each fused
stage's GEMM tile gets only `SMEM - resident`. Hypothesis: fusing MORE stages leaves less SMEM
per tile, so past some depth N* fuse-all stops being the best partition.

This module reduces each fused stage's tile-SMEM budget to (SMEM - resident) and re-times it with
the estimator (dataclasses.replace on smem_per_block_bytes), so the estimator itself picks the
smaller tile / shallower pipeline and reports the slowdown. Two resident schedules:
  * seq   : buffer-reuse (tiny-cuda-nn style) -> resident = m0 * max_s(K_s+K_{s+1}) * bpe
            (two adjacent activations). DEPTH-INDEPENDENT. **This is the REALISTIC model** — a
            linear chain only needs its input+output activation live at any stage.
  * full  : accumulate ALL boundary activations -> resident = m0 * sum(widths) * bpe, GROWS with
            depth. **Physically unrealistic** (a real kernel frees consumed activations) — kept
            only as an artificial upper bound that forces the resident to exceed SMEM at depth.

FINDINGS (adversarially audited — see notes/multi_gemm_smem_plan.md and multi_gemm_smem_results.md):
  The user's PREMISE (snowcat perf grows with tile SMEM) is true in general but DOES NOT ENGAGE in
  the narrow-w regime where chain fusion is attractive: optimal_mapping_by_time pins the 64x64x32
  floor tile (C=2, since c_sat=1) from 227 KiB down to ~32 KiB SMEM, so the SMEM->perf channel is
  saturated at the floor and the reduced budget is INERT on time for every feasible segment. The
  fused kernel also DODGES starvation by shrinking the row-block m0 (free, since narrow weights are
  L2-resident). So there is NO gradual perf crossover anywhere (0 of ~160 cells). A crossover
  appears ONLY under 'full' and ONLY as a hard FEASIBILITY cliff at N* = floor(SMEM/(m0_min*w*bpe))
  ~ floor(6368/w) (w=512->12, 1024->6, 2048->3) — a residency-formula threshold, not a perf
  optimum. Under the realistic 'seq' schedule there is no crossover at any width through L=16.

Everything else (weight re-read mt x, input/output once, cut = HBM round-trip, all GEMMs via the
estimator) is identical to multi_gemm_fusion.py.

Run:  conda run -n area python multi_gemm_smem.py --w 512 --M 131072 --Lmax 14 --smem seq
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math

from gemm_time_estimator import GPUS, GpuModel, optimal_mapping_by_time
from snowcat_demo.model.workload import divisors
from chain_gemm_fusion import BPE, MMA_MIN_M, STREAM_OVERHEAD, _dram_of, _eff_l2, _roofline
from multi_gemm_fusion import all_partitions, TIE_TOL

MIN_TILE_SMEM = 12 * 1024   # a 64x64x32 double-buffered tile needs ~16 KiB; floor below which infeasible

_SEG_CACHE: dict = {}       # memoize segment_time (segments of equal widths are identical) -> fast L sweep


def _resident_width(widths: list[int], schedule: str) -> int:
    k = len(widths) - 1
    if schedule == "seq":
        return max(widths[s] + widths[s + 1] for s in range(k))     # two adjacent activations
    return sum(widths)                                              # 'full': all boundary activations


def segment_time(widths: list[int], M: int, gpu: GpuModel, schedule: str) -> tuple[float, dict]:
    """Fused segment with SMEM competition: tile budget = SMEM - resident activations."""
    key = (tuple(widths), M, gpu.name, schedule)
    if key in _SEG_CACHE:
        return _SEG_CACHE[key]
    res = _segment_time_uncached(widths, M, gpu, schedule)
    _SEG_CACHE[key] = res
    return res


def _segment_time_uncached(widths: list[int], M: int, gpu: GpuModel, schedule: str) -> tuple[float, dict]:
    k = len(widths) - 1
    if k == 1:                                   # length-1 == plain unfused GEMM, full SMEM
        _, e = optimal_mapping_by_time(M, widths[1], widths[0], gpu, l2=True)
        return e.time_s, {"m0": M, "mt": 1, "k": 1, "bott": "gemm", "tile_smem_kib": gpu.smem_per_block_bytes // 1024}
    eff_l2 = _eff_l2(gpu)
    live_w = _resident_width(widths, schedule)
    ops = sum(2 * M * widths[s] * widths[s + 1] for s in range(k))
    best = None
    for m0 in divisors(M):
        if m0 < MMA_MIN_M:
            continue
        mt = M // m0
        resident = m0 * live_w * BPE + STREAM_OVERHEAD
        tile_smem = gpu.smem_per_block_bytes - resident              # SMEM left for the GEMM tile
        if tile_smem < MIN_TILE_SMEM:
            continue                                                # can't fit a tile at this m0
        gpu_r = dataclasses.replace(gpu, smem_per_block_bytes=int(tile_smem))
        dram, last_bn, last_c, last_w, ok = 0.0, 1, 2, resident, True
        for s in range(k):
            w_in, w_out = widths[s], widths[s + 1]
            try:
                _, eb = optimal_mapping_by_time(m0, w_out, w_in, gpu_r, l2=True)   # starved tile
            except Exception:
                ok = False
                break
            w_big = (w_in * w_out * BPE) > eff_l2
            dram += (mt if w_big else 1) * _dram_of(eb, "W")
            if s == 0:
                dram += mt * _dram_of(eb, "A")
            if s == k - 1:
                dram += mt * _dram_of(eb, "OUT")
                last_bn, last_c, last_w = max(eb.mapping.bn, 1), eb.num_stages, eb.working_set_bytes
        if not ok:
            continue
        out_tiles = mt * max(1, widths[-1] // last_bn)
        # roofline on the REAL hw; SMEM starvation enters via last_c (pipeline depth) and W=tile working set.
        t, bott = _roofline(gpu, ops, dram, out_tiles, last_w, last_c)
        if best is None or t < best[0]:
            best = (t, {"m0": m0, "mt": mt, "k": k, "bott": bott, "C": last_c,
                        "tile_smem_kib": tile_smem / 1024, "resident_kib": resident / 1024,
                        "dram_mib": dram / 2**20})
    return best if best else (float("inf"),
                              {"why": f"INFEASIBLE: resident({live_w}*m0*2) leaves < {MIN_TILE_SMEM//1024}KiB for tile", "k": k})


def partition_time(widths, M, gpu, cuts, schedule):
    L = len(widths) - 1
    bounds = [0] + list(cuts) + [L]
    total, segs = 0.0, []
    for a, b in zip(bounds, bounds[1:]):
        t, info = segment_time(widths[a:b + 1], M, gpu, schedule)
        total += t
        segs.append((a + 1, b, t, info))
    return total, segs


def analyze(M, w, L, gpu, schedule):
    widths = [w] * (L + 1)
    results = [(partition_time(widths, M, gpu, cuts, schedule)[0], cuts) for cuts in all_partitions(L)]
    feasible = sorted([r for r in results if math.isfinite(r[0])], key=lambda r: r[0])
    if feasible:
        tmin = feasible[0][0]
        near = [r for r in feasible if r[0] <= tmin * (1 + TIE_TOL)]
        best = min(near, key=lambda r: len(r[1]))          # fewest cuts among ties
    else:
        best = None
    fuse_all = next(r for r in results if r[1] == ())
    unfused = next(r for r in results if r[1] == tuple(range(1, L)))
    return {"M": M, "w": w, "L": L, "schedule": schedule, "results": results,
            "best": best, "fuse_all": fuse_all, "unfused": unfused}


def find_Nstar(M, w, gpu, schedule, Lmax=10):
    """Smallest L at which fuse-all is NOT the optimal partition (or None if it stays optimal)."""
    rows = []
    nstar = None
    for L in range(2, Lmax + 1):
        a = analyze(M, w, L, gpu, schedule)
        fa = a["fuse_all"][0]
        best_t, best_cuts = a["best"] if a["best"] else (float("inf"), None)
        fa_opt = math.isfinite(fa) and best_cuts == ()
        rows.append((L, fa, best_t, best_cuts, fa_opt, a["unfused"][0]))
        if nstar is None and not fa_opt:
            nstar = L
    return nstar, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=sorted(GPUS), default="h100-sxm")
    ap.add_argument("--M", type=int, default=131072)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--Lmax", type=int, default=10)
    ap.add_argument("--smem", choices=["seq", "full"], default="seq",
                    help="seq = realistic buffer-reuse (no crossover); full = artificial accumulate upper bound")
    args = ap.parse_args()
    gpu = GPUS[args.gpu]
    nstar, rows = find_Nstar(args.M, args.w, gpu, args.smem, args.Lmax)
    print(f"\n=== Multi-GEMM fusion + SMEM competition ({args.smem}) — {gpu.name} ===")
    print(f"M={args.M}, uniform w={args.w}, eff-L2={_eff_l2(gpu)/2**20:.0f}MiB, SMEM/blk={gpu.smem_per_block_bytes//1024}KiB")
    print(f"{'L':>3} {'fuse-all ms':>12} {'best ms':>10} {'best cuts':>16} {'fuse-all opt?':>14} {'unfused ms':>11}")
    for L, fa, bt, bc, opt, uf in rows:
        fam = f"{fa*1e3:.4f}" if math.isfinite(fa) else "INFEAS"
        btm = f"{bt*1e3:.4f}" if math.isfinite(bt) else "INFEAS"
        print(f"{L:>3} {fam:>12} {btm:>10} {str(bc) if bc is not None else '--':>16} "
              f"{('YES' if opt else 'no'):>14} {uf*1e3:>11.4f}")
    print(f"\n  ==> fuse-all stops being optimal at L = {nstar}" if nstar else
          f"\n  ==> fuse-all stays optimal through L={args.Lmax}")


if __name__ == "__main__":
    main()
