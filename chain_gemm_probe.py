"""Shared probe for the 2-GEMM chain fusion 'shape vs size' analysis.

One entry point, `probe(M, K1, N1, N2, gpu)`, returns the estimator-backed fuse/unfuse verdict
AND the dimensionless drivers, so downstream experiments can test whether the verdict tracks
SHAPE (aspect ratios, scale-free) or SIZE (absolute magnitude vs fixed hardware capacities).

All timing goes through the SAME snowcat-roofline functions used everywhere else
(`unfused_time` / `fused_time` in chain_gemm_fusion.py) — no new physics.

Definitions
-----------
SHAPE (scale-free): the aspect ratios. We report log2 aspect of each matrix
    aA = log2(M/K1), aB = log2(K1/N1), aD = log2(N1/N2)
  and the normalized dim vector dims / geomean(dims).
SIZE (scale): the absolute magnitude. We report geomean(M,K1,N1,N2) and total FLOPs.

Dimensionless drivers (all are SIZE-vs-hardware ratios — the suspected true levers):
    f_held = min(N1,N2)*bpe / (SMEM_block - overhead)   # feasibility (row-block capacity)
    f_B    = K1*N1*bpe / eff_L2                          # weight-B L2 residency (re-read cost)
    f_D    = N1*N2*bpe / eff_L2                          # weight-D L2 residency
    f_C    = M*N1*bpe  / eff_L2                          # intermediate L2 residency (saving exists iff >1)
    AI_fused = fused_ops / fused_dram ; ridge = peak_flops / bw   # compute-vs-memory
"""

from __future__ import annotations

import math

from gemm_time_estimator import GPUS, GpuModel
from chain_gemm_fusion import BPE, STREAM_OVERHEAD, _eff_l2, fused_time, unfused_time


def probe(M: int, K1: int, N1: int, N2: int, gpu: GpuModel) -> dict:
    u_t, ui = unfused_time(M, K1, N1, N2, gpu)
    f_t, fi = fused_time(M, K1, N1, N2, gpu)
    feasible = math.isfinite(f_t)
    if feasible:
        winner = "FUSE" if f_t < u_t else "unfuse"
        speedup = u_t / f_t
    else:
        winner = "infeasible"
        speedup = float("nan")

    eff_l2 = _eff_l2(gpu)
    smem = gpu.smem_per_block_bytes - STREAM_OVERHEAD
    ops = 2 * M * N1 * K1 + 2 * M * N1 * N2
    fused_dram = (fi.get("A", 0) + fi.get("B", 0) + fi.get("D", 0) + fi.get("E", 0)) if feasible else float("nan")
    dims = [M, K1, N1, N2]
    geomean = math.exp(sum(math.log(d) for d in dims) / 4)

    return {
        "M": M, "K1": K1, "N1": N1, "N2": N2,
        # verdict
        "winner": winner, "feasible": feasible,
        "speedup": speedup,               # unfused/fused ; >1 => FUSE
        "u_ms": u_t * 1e3, "f_ms": f_t * 1e3 if feasible else float("nan"),
        "bott": fi.get("bott", "n/a"), "m0": fi.get("m0"), "mt": fi.get("mt"),
        # SHAPE (scale-free)
        "aA_log2": math.log2(M / K1), "aB_log2": math.log2(K1 / N1), "aD_log2": math.log2(N1 / N2),
        "norm_dims": [d / geomean for d in dims],
        # SIZE (scale)
        "geomean_dim": geomean, "tflops": ops / 1e12,
        # dimensionless drivers (size-vs-hardware)
        "f_held": min(N1, N2) * BPE / smem,
        "f_B": K1 * N1 * BPE / eff_l2,
        "f_D": N1 * N2 * BPE / eff_l2,
        "f_C": M * N1 * BPE / eff_l2,
        "AI_fused": ops / fused_dram if feasible and fused_dram else float("nan"),
        "ridge": gpu.peak_tensor_flops / gpu.bw_bytes_per_s,
    }


def fmt(p: dict) -> str:
    return (f"{p['M']:>6}x{p['K1']:>6}x{p['N1']:>6}x{p['N2']:>6} | {p['winner']:>10} "
            f"sp={p['speedup']:.3f} | f_held={p['f_held']:.3f} f_B={p['f_B']:.2f} "
            f"f_C={p['f_C']:.2f} AI={p['AI_fused']:.0f} ridge={p['ridge']:.0f} [{p['bott']}]")


if __name__ == "__main__":
    g = GPUS["h100-sxm"]
    for M, K1, N1, N2 in [
        (8192, 512, 4096, 128),    # FUSE (B in L2)
        (8192, 8192, 4096, 128),   # unfuse (B in HBM)
        (8192, 16384, 4096, 256),  # FUSE again (compute-bound)
        (16384, 4096, 4096, 4096), # unfuse (from 27-shape)
        (16384, 16384, 16384, 16384),  # infeasible
    ]:
        print(fmt(probe(M, K1, N1, N2, g)))
