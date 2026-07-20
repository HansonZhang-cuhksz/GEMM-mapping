"""Is fusing a 2-GEMM chain worth it, across GEMM shapes?  (H100, analytical)

Chain:  C[M,N1] = A[M,K1] @ B[K1,N1]  ;  E[M,N2] = C[M,N1] @ D[N1,N2].
Sweep each of A, B, D over {square, tall, wide} = 27 cases. For each, compare the
latency-aware snowcat-roofline time of the FUSED kernel (C on chip) vs the UNFUSED pair.

**This version calls the snowcat-roofline estimator directly** (gemm_time_estimator.
estimate_gemm_time / optimal_mapping_by_time) for EVERY GEMM — the per-operand snowcat+L2
traffic and the calibrated occupancy roofline come from the estimator, not a hand model.

Traffic model (H100 effective L2 = 0.6*50 MB = 30 MB). A,B,D,E cross HBM; the intermediate C:
  * UNFUSED: GEMM1 writes C, GEMM2 reads it. If C <= eff-L2 it stays in L2 (0 HBM); else it
    round-trips HBM (2*M*N1).  <- the "unfused-C-via-L2-or-DRAM" axis.
  * FUSED: C never leaves the SM. But GEMM2 reduces over the FULL N1, so each output row-block
    (m0 rows, m0 <= SMEM/(min(N1,N2)*bpe)) needs the full B and full D -> B, D are re-read once
    PER BLOCK (mt = M/m0 blocks). This is the same re-read flash-attention does over K,V per
    query block. A weight that fits eff-L2 is re-read from L2 (1x DRAM); otherwise mt x DRAM.

So fusion trades: SAVE the unfused C round-trip (2*M*N1, only when C > L2) FOR the extra weight
re-reads ((mt-1)*(B_dram + D_dram)). It wins when C is large and the re-read weights are small
/ L2-resident.

A real tiled GEMM reuses its A/B panels in SMEM/registers regardless of L2, so a global
"no L2" is unphysical (it would only mis-penalize the unfused pair); DRAM vs L2 is decided
per operand by SIZE (fits eff-L2 or not), which is what a real cache does.

Run:  conda run -n area python chain_gemm_fusion.py --verbose
"""

from __future__ import annotations

import argparse
import itertools
import math

from gemm_time_estimator import (
    GPUS,
    GpuModel,
    Mapping,
    estimate_gemm_time,
    optimal_mapping_by_time,
)
from snowcat_demo.model.workload import divisors

BPE = 2
ASPECTS = ("square", "tall", "wide")
A_FACTOR = {"square": 1.0, "tall": 0.25, "wide": 4.0}
MAX_DIM = 16384
MMA_MIN_M = 16
STREAM_OVERHEAD = 16 * 1024


def case_dims(a_a: str, a_b: str, a_d: str) -> tuple[int, int, int, int]:
    """(M, K1, N1, N2). f=4 chained aspects, centered so max dim = MAX_DIM (dims are 2^k)."""
    rels = [1.0]
    for a in (a_a, a_b, a_d):
        rels.append(rels[-1] * A_FACTOR[a])
    scale = MAX_DIM / max(rels)
    return tuple(int(round(r * scale)) for r in rels)


def _eff_l2(gpu: GpuModel) -> float:
    return gpu.l2_capacity_alpha * gpu.l2_bytes


def _dram_of(e, op: str) -> float:
    """Per-operand DRAM (bytes) from the estimator's L2 breakdown: op in {A,W,OUT}."""
    idx = {"A": 0, "W": 1, "OUT": 2}[op]
    return e.l2_breakdown[idx][7]


# --------------------------------------------------------------------------- #
# UNFUSED: two optimal GEMMs via the estimator; C via L2 (HBM round-trip if > L2) #
# --------------------------------------------------------------------------- #
def unfused_time(M, K1, N1, N2, gpu: GpuModel) -> tuple[float, dict]:
    # Both GEMMs use the estimator's realistic L2 model (a real tiled GEMM reuses A/B panels
    # in SMEM/L2 regardless). The DRAM question is only the *inter-kernel* intermediate C.
    _, e1 = optimal_mapping_by_time(M, N1, K1, gpu, l2=True)   # C[M,N1] = A[M,K1] @ B[K1,N1]
    _, e2 = optimal_mapping_by_time(M, N2, N1, gpu, l2=True)   # E[M,N2] = C[M,N1] @ D[N1,N2]
    c_bytes = M * N1 * BPE
    c_in_l2 = c_bytes <= _eff_l2(gpu)                          # C stays in L2 iff it fits
    if c_in_l2:   # C stays in L2 across the two kernels: drop its HBM write (g1 OUT) + read (g2 A)
        t1 = max(e1.compute_time_eff_s, max(0.0, e1.traffic_bytes - c_bytes) / e1.bw_eff_bytes_per_s)
        t2 = max(e2.compute_time_eff_s, max(0.0, e2.traffic_bytes - c_bytes) / e2.bw_eff_bytes_per_s)
    else:
        t1, t2 = e1.time_s, e2.time_s
    info = {"t": t1 + t2, "g1_ms": t1 * 1e3, "g2_ms": t2 * 1e3,
            "C": "L2" if c_in_l2 else "HBM", "c_mib": c_bytes / 2**20,
            "map1": f"{e1.mapping.bm}x{e1.mapping.bn}x{e1.mapping.bk}",
            "map2": f"{e2.mapping.bm}x{e2.mapping.bn}x{e2.mapping.bk}"}
    return t1 + t2, info


# --------------------------------------------------------------------------- #
# FUSED: estimator per row-block sub-GEMMs (C on chip); B,D re-read per block     #
# --------------------------------------------------------------------------- #
def _roofline(gpu: GpuModel, ops, dram, out_tiles, w, c):
    """Exact copy of the estimator's occupancy roofline (gemm_time_estimator lines ~543-580)."""
    active_sm = min(max(out_tiles, 1), gpu.num_sm)
    waves = math.ceil(max(out_tiles, 1) / gpu.num_sm)
    sm_util = max(out_tiles, 1) / (waves * gpu.num_sm)
    per_sm_bw = gpu.bw_bytes_per_s / gpu.bw_saturation_sms
    bw_latency = active_sm * c * w / gpu.latency_seconds
    bw_eff = min(gpu.bw_bytes_per_s, per_sm_bw * active_sm, bw_latency)
    compute_eff = (ops / gpu.peak_tensor_flops) / sm_util if sm_util > 0 else ops / gpu.peak_tensor_flops
    memory = dram / bw_eff if bw_eff > 0 else float("inf")
    return max(compute_eff, memory), ("compute" if compute_eff >= memory else "memory")


def fused_time(M, K1, N1, N2, gpu: GpuModel) -> tuple[float, dict]:
    hold = min(N1, N2)
    m0_max = (gpu.smem_per_block_bytes - STREAM_OVERHEAD) // (hold * BPE)
    if m0_max < MMA_MIN_M:
        return float("inf"), {"why": f"INFEASIBLE: need {hold*BPE//1024} KiB/row, m0_max={m0_max}<{MMA_MIN_M}"}
    eff_l2 = _eff_l2(gpu)
    # A cross-block weight re-read hits L2 (1x DRAM) iff the whole weight fits eff-L2; else mt x.
    b_big = (K1 * N1 * BPE) > eff_l2
    d_big = (N1 * N2 * BPE) > eff_l2
    ops = 2 * M * N1 * K1 + 2 * M * N1 * N2
    best = None
    for m0 in divisors(M):
        if m0 < MMA_MIN_M or m0 > m0_max:
            continue
        mt = M // m0
        # per-row-block sub-GEMMs, via the estimator (snowcat + L2 breakdown), C never in HBM.
        # The blocks are tiny (m0 rows) so intra-block L2 barely matters; the L2 question that
        # governs the verdict is the CROSS-block weight re-read (b_big/d_big below), set by `l2`.
        _, eb1 = optimal_mapping_by_time(m0, N1, K1, gpu, l2=True)  # A[m0]@B -> C[m0]  (drop C=OUT)
        _, eb2 = optimal_mapping_by_time(m0, N2, N1, gpu, l2=True)  # C[m0]@D -> E[m0]  (drop C=A)
        a_read = _dram_of(eb1, "A")
        b_read = _dram_of(eb1, "W")
        d_read = _dram_of(eb2, "W")
        e_write = _dram_of(eb2, "OUT")
        # aggregate over mt blocks; a weight re-read hits L2 (1x DRAM) if it fits, else mt x.
        dram = (mt * a_read + (mt if b_big else 1) * b_read
                + (mt if d_big else 1) * d_read + mt * e_write)
        out_tiles = mt * max(1, N2 // max(eb2.mapping.bn, 1))
        resident = m0 * hold * BPE + STREAM_OVERHEAD
        t, bott = _roofline(gpu, ops, dram, out_tiles, resident, eb2.num_stages)
        if best is None or t < best[0]:
            best = (t, {"m0": m0, "mt": mt, "bott": bott,
                        "A": mt * a_read, "B": (mt if b_big else 1) * b_read,
                        "D": (mt if d_big else 1) * d_read, "E": mt * e_write,
                        "Bx": mt if b_big else 1, "Dx": mt if d_big else 1,
                        "g1": f"{eb1.mapping.bm}x{eb1.mapping.bn}x{eb1.mapping.bk}",
                        "g2": f"{eb2.mapping.bm}x{eb2.mapping.bn}x{eb2.mapping.bk}"})
    return (best[0], best[1]) if best else (float("inf"), {"why": "no divisor m0 fits"})


def run(gpu: GpuModel, verbose: bool = False) -> None:
    print(f"\n################  2-GEMM chain fusion — {gpu.name}  ################")
    print(f"peak {gpu.peak_tensor_flops/1e12:.0f} TFLOP/s | HBM {gpu.bw_bytes_per_s/1e12:.2f} TB/s | "
          f"SMEM/blk {gpu.smem_per_block_bytes//1024} KiB | eff-L2 {_eff_l2(gpu)/2**20:.0f} MiB | dims 256..{MAX_DIM}")
    print(f"  (all GEMMs via estimate_gemm_time. unfused C: in L2 if <= eff-L2 else HBM round-trip. "
          f"fused: C on-chip, B/D re-read mt x per row-block, from L2 if the weight fits else DRAM.)")
    print(f"\n{'A':>6} {'B':>6} {'D':>6} | {'M':>6} {'K1':>6} {'N1':>6} {'N2':>6} | "
          f"{'unfused':>8} {'fused':>8} {'speedup':>8} {'winner':>7} {'C':>4}")
    n_fuse = n_unfuse = n_infeas = 0
    wins = []
    for a_a, a_b, a_d in itertools.product(ASPECTS, repeat=3):
        M, K1, N1, N2 = case_dims(a_a, a_b, a_d)
        u_t, ui = unfused_time(M, K1, N1, N2, gpu)
        f_t, fi = fused_time(M, K1, N1, N2, gpu)
        if not math.isfinite(f_t):
            winner, sp, ft = "infeas", "--", "INFEAS"
            n_infeas += 1
        elif f_t < u_t:
            winner, sp, ft = "FUSE", f"{u_t/f_t:.3f}x", f"{f_t*1e3:.3f}"
            n_fuse += 1
            wins.append((a_a, a_b, a_d, M, K1, N1, N2, u_t / f_t, ui["C"]))
        else:
            winner, sp, ft = "unfuse", f"{u_t/f_t:.3f}x", f"{f_t*1e3:.3f}"
            n_unfuse += 1
        print(f"{a_a:>6} {a_b:>6} {a_d:>6} | {M:>6} {K1:>6} {N1:>6} {N2:>6} | "
              f"{u_t*1e3:>8.3f} {ft:>8} {sp:>8} {winner:>7} {ui['C']:>4}")
        if verbose:
            print(f"        unfused: g1 {ui['g1_ms']:.3f} + g2 {ui['g2_ms']:.3f} ms  "
                  f"tiles {ui['map1']},{ui['map2']}  C={ui['c_mib']:.0f}MiB->{ui['C']}")
            if math.isfinite(f_t):
                mib = 2**20
                print(f"        fused:   m0={fi['m0']} mt={fi['mt']} [{fi['bott']}]  DRAM(MiB): "
                      f"A {fi['A']/mib:.0f} + B {fi['B']/mib:.0f}(x{fi['Bx']}) + "
                      f"D {fi['D']/mib:.0f}(x{fi['Dx']}) + E {fi['E']/mib:.0f}")
            else:
                print(f"        fused:   {fi['why']}")
    print(f"\n  ==> FUSE wins {n_fuse}   unfuse wins {n_unfuse}   infeasible {n_infeas}   (of 27)")
    for a_a, a_b, a_d, M, K1, N1, N2, s, cloc in sorted(wins, key=lambda x: -x[7]):
        print(f"    FUSE: A={a_a:<6} B={a_b:<6} D={a_d:<6}  {s:.3f}x   "
              f"(M={M} K1={K1} N1={N1} N2={N2}; unfused C in {cloc})")


def emit_table(gpu: GpuModel, path: str | None = None) -> None:
    """Markdown table: per case, the unfused & fused kernel mappings, times, and winner."""
    lines = [
        f"# 2-GEMM chain fusion — per-case mappings & times ({gpu.name})",
        "",
        f"Chain `C=A@B, E=C@D`. All GEMMs timed by the snowcat-roofline estimator "
        f"(`estimate_gemm_time`). Tiles are `BM×BN×BK`. Unfused = GEMM1(A@B)+GEMM2(C@D), C via "
        f"L2 if ≤30 MB else HBM. Fused = one kernel over `mt=M/m0` row-blocks (C on chip; "
        f"B,D re-read once per block). f=4 shapes, dims 256..{MAX_DIM}, H100.",
        "",
        "| A | B | D | M×K1×N1×N2 | unfused G1 / G2 tile | unf ms | C | fused m0×mt, G1 / G2 tile | fus ms | speedup | winner |",
        "|---|---|---|---|---|---:|:--:|---|---:|---:|:--:|",
    ]
    nf = nu = ni = 0
    for a_a, a_b, a_d in itertools.product(ASPECTS, repeat=3):
        M, K1, N1, N2 = case_dims(a_a, a_b, a_d)
        u_t, ui = unfused_time(M, K1, N1, N2, gpu)
        f_t, fi = fused_time(M, K1, N1, N2, gpu)
        umap = f"{ui['map1']} / {ui['map2']}"
        if not math.isfinite(f_t):
            fmap, fms, sp, win = "INFEASIBLE (slice too wide)", "—", "—", "infeasible"
            ni += 1
        else:
            fmap = f"m0={fi['m0']}×{fi['mt']}, {fi['g1']} / {fi['g2']}"
            fms = f"{f_t*1e3:.3f}"
            sp = f"{u_t/f_t:.3f}×"
            if f_t < u_t:
                win, nf = "**FUSE**", nf + 1
            else:
                win, nu = "unfuse", nu + 1
        lines.append(f"| {a_a} | {a_b} | {a_d} | {M}×{K1}×{N1}×{N2} | {umap} | "
                     f"{u_t*1e3:.3f} | {ui['C']} | {fmap} | {fms} | {sp} | {win} |")
    lines += ["", f"**Totals: FUSE {nf} / unfuse {nu} / infeasible {ni}** (of 27). "
              f"Fusion wins only for tall-A + tall-B (large C round-trip avoided, weights L2-resident).", ""]
    out = "\n".join(lines)
    print(out)
    if path:
        with open(path, "w") as fh:
            fh.write(out + "\n")
        print(f"[wrote {path}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=sorted(GPUS), default="h100-sxm")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--table", action="store_true",
                    help="emit the per-case mapping/time/winner markdown table")
    ap.add_argument("--out", default=None, help="write the table to this file")
    args = ap.parse_args()
    if args.table:
        emit_table(GPUS[args.gpu], args.out)
    else:
        run(GPUS[args.gpu], verbose=args.verbose)


if __name__ == "__main__":
    main()
