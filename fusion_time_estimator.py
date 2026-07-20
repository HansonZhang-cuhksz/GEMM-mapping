"""Real-GPU fused-vs-unfused kernel time estimator (built on gemm_time_estimator.py).

The inference-framework question: on a FIXED real GPU, is the fused optimal kernel faster
than the unfused optimal kernels it replaces? Answers it for the six GLM-5.2 fusions in the
decode (batch 2048) regime, on any GpuModel from gemm_time_estimator (default: h100-sxm and
rtx4060-measured).

Reuses, out of place (import only; gemm_time_estimator.py is unchanged):
  * GpuModel + GPUS profiles, the snowcat+L2 per-operand traffic, and the
    occupancy/wave-quantization roofline of estimate_gemm_time().

Adds:
  * a memory-bound vector-kernel model (residual / RMSNorm / SwiGLU epilogues; their CUDA
    compute is hidden — arithmetic intensity <= 8 FLOP/byte << both GPUs' ridge OI);
  * a fused GEMM-with-epilogue model (per-operand traffic deltas: halved output for the
    SwiGLU epilogue, doubled A for the down prologue, extra residual/gamma/partial-RMS
    reads; plus the epilogue's on-chip aux state, which reduces the SMEM available to the
    GEMM pipeline);
  * a grouped-GEMM multiplier (count = 256 experts run together and fill the GPU);
  * the full-FFN GEMM-GEMM fusion (F6): intermediate on chip, weights re-read per row-block.

See notes/fusion_time_estimator_plan.md.  Run:
  conda run -n profiling python fusion_time_estimator.py
  conda run -n profiling python fusion_time_estimator.py --gpu h100-sxm --verbose
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from gemm_time_estimator import (
    GPUS,
    MMA_MIN_BM,
    MMA_MIN_BN,
    MMA_MIN_BK,
    GpuModel,
    Mapping,
    _auto_num_stages,
    estimate_gemm_time,
)
from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload, divisors

# --------------------------------------------------------------------------- #
# Decode batch-2048 workload (GLM-5.2)                                          #
# --------------------------------------------------------------------------- #
BPE = 2
HIDDEN = 6144
INTERMEDIATE = 2048
EXPERTS = 256
N_HEADS = 64
V_HEAD_DIM = 256
BATCH = 2048
TOP_K = 8
TOKENS_PER_EXPERT = BATCH * TOP_K // EXPERTS          # 64

MLA_O = (BATCH, HIDDEN, N_HEADS * V_HEAD_DIM)         # 2048 x 6144 x 16384
UP_GATE = (TOKENS_PER_EXPERT, 2 * INTERMEDIATE, HIDDEN)   # 64 x 4096 x 6144  (per expert)
DOWN = (TOKENS_PER_EXPERT, HIDDEN, INTERMEDIATE)          # 64 x 6144 x 2048  (per expert)

# Memory-bound vector/reduction kernel HBM traffic (bytes).
RESIDUAL_TRAFFIC = 3 * BATCH * HIDDEN * BPE              # read y, read x, write sum
RMSNORM_TRAFFIC = BATCH * HIDDEN * BPE + BATCH * 4       # read h, write per-row stat
ACTIVATION_TRAFFIC = (BATCH * TOP_K) * (2 * INTERMEDIATE + INTERMEDIATE) * BPE  # gate+up -> activated


# --------------------------------------------------------------------------- #
# Roofline (mirrors estimate_gemm_time; applied to aggregate ops/traffic/tiles) #
# --------------------------------------------------------------------------- #
@dataclass
class KTime:
    label: str
    time_s: float
    compute_s: float
    memory_s: float
    bottleneck: str
    traffic_bytes: float
    ops: float
    tiles: int
    detail: str = ""


def _roofline(gpu: GpuModel, ops: float, traffic: float, tiles: int, w: int, c: int):
    """Occupancy-aware roofline (same law as estimate_gemm_time) for aggregate work."""
    active_sm = min(max(tiles, 1), gpu.num_sm)
    waves = math.ceil(max(tiles, 1) / gpu.num_sm)
    sm_util = max(tiles, 1) / (waves * gpu.num_sm)
    per_sm_bw = gpu.bw_bytes_per_s / gpu.bw_saturation_sms
    inflight = active_sm * c * w
    bw_latency = inflight / gpu.latency_seconds if gpu.latency_seconds > 0 else gpu.bw_bytes_per_s
    bw_eff = min(gpu.bw_bytes_per_s, per_sm_bw * active_sm, bw_latency)
    compute_s = ops / gpu.peak_tensor_flops if gpu.peak_tensor_flops > 0 else 0.0
    compute_eff = compute_s / sm_util if sm_util > 0 else compute_s
    memory_s = traffic / bw_eff if bw_eff > 0 else float("inf")
    time_s = max(compute_eff, memory_s)
    bott = "compute" if compute_eff >= memory_s else "memory"
    return time_s, compute_eff, memory_s, bott, sm_util, waves


def estimate_vector_kernel(label: str, traffic_bytes: float, gpu: GpuModel) -> KTime:
    """Memory-bound elementwise/reduction kernel: many tiles -> occupancy ~1, compute hidden.

    time = traffic / bw_peak (the saturating-BW law delivers full peak once the many small
    tiles fill the chip; the CUDA-core FLOPs of these low-AI ops are hidden under memory).
    """
    time_s = traffic_bytes / gpu.bw_bytes_per_s
    return KTime(label, time_s, 0.0, time_s, "memory", traffic_bytes, 0.0,
                 tiles=gpu.num_sm, detail="vector (memory-bound, compute hidden)")


# --------------------------------------------------------------------------- #
# Plain GEMM (unfused), aggregated over `count` grouped instances               #
# --------------------------------------------------------------------------- #
def _candidate_tiles(m, n, k, gpu, min_bm=64, min_bn=64, min_bk=32):
    # 64x64x32 tensor-sensible floor (matches gemm_time_estimator.optimal_mapping_by_time);
    # avoids the estimator's tiny-tile collapse (TODO.md compute-efficiency term).
    wl = GemmWorkload(m=m, k=k, n=n, bytes_per_element=gpu.bytes_per_element)
    lo_bm, lo_bn, lo_bk = min(min_bm, m), min(min_bn, n), min(min_bk, k)
    for p in enumerate_mappings(wl):
        mp = p.mapping
        if mp.m0 >= lo_bm and mp.n0 >= lo_bn and mp.k0 >= lo_bk:
            yield mp


def estimate_gemm_grouped(label, m, n, k, count, gpu: GpuModel) -> KTime:
    """Optimal-mapping time of a grouped GEMM (count independent instances filling the GPU).

    Per-instance snowcat+L2 traffic from estimate_gemm_time; aggregate over count with
    occupancy from the total tile count.
    """
    best = None
    for mp in _candidate_tiles(m, n, k, gpu):
        try:
            e = estimate_gemm_time(m, n, k, Mapping(mp.m0, mp.n0, mp.k0, mp.loop_order), gpu)
        except ValueError:
            continue
        if not e.fits_smem:
            continue
        traffic = count * e.traffic_bytes
        ops = count * e.ops
        tiles = count * e.output_tiles
        t, ce, me, bott, _, _ = _roofline(gpu, ops, traffic, tiles, e.working_set_bytes, e.num_stages)
        if best is None or t < best[0]:
            best = (t, ce, me, bott, traffic, ops, tiles,
                    f"{mp.m0}x{mp.n0}x{mp.k0} {'-'.join(mp.loop_order)} C={e.num_stages}")
    if best is None:
        raise ValueError(f"{label}: no feasible mapping")
    t, ce, me, bott, traffic, ops, tiles, det = best
    return KTime(label, t, ce, me, bott, traffic, ops, tiles, det)


# --------------------------------------------------------------------------- #
# Fused GEMM + epilogue/prologue                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Epilogue:
    """Per-operand traffic deltas + aux SMEM for a fused GEMM (all relative to base GEMM)."""
    a_factor: float = 1.0            # A-read multiplier (F5 down reads 2x-wide gate+up)
    out_factor: float = 1.0          # OUT-write multiplier (F4 writes activated = 0.5x)
    extra_hbm_once: float = 0.0      # extra HBM read/write, once over the whole instance
    aux_smem_per_tile: object = None  # fn(bm, bn) -> extra on-chip bytes (residual tile, ...)


def estimate_fused_gemm(label, m, n, k, count, epi: Epilogue, gpu: GpuModel) -> KTime:
    """Optimal-mapping time of a GEMM with a fused epilogue/prologue.

    Uses estimate_gemm_time's per-operand L2/sector DRAM (A/W/OUT), applies the epilogue's
    per-operand factors + extra HBM, and charges the epilogue's aux SMEM against the pipeline
    budget (C*W + aux <= SMEM/block) -- the real-GPU analogue of the area study's starvation.
    """
    best = None
    for mp in _candidate_tiles(m, n, k, gpu):
        try:
            e = estimate_gemm_time(m, n, k, Mapping(mp.m0, mp.n0, mp.k0, mp.loop_order), gpu)
        except ValueError:
            continue
        aux = epi.aux_smem_per_tile(mp.m0, mp.n0) if epi.aux_smem_per_tile else 0
        if e.num_stages * e.working_set_bytes + aux > gpu.smem_per_block_bytes:
            continue                      # fused working set (pipeline + aux) must fit SMEM
        # per-operand DRAM from the base L2 model, then apply epilogue deltas
        dram = {op: row[7] for row, op in zip(e.l2_breakdown, ("A", "W", "OUT"))} if e.l2_breakdown else None
        if dram is None:      # L2 disabled path not used here
            continue
        t_inst = (epi.a_factor * dram["A"] + dram["W"] + epi.out_factor * dram["OUT"]
                  + e.reduction_bytes + epi.extra_hbm_once)
        traffic = count * t_inst
        ops = count * e.ops
        tiles = count * e.output_tiles
        t, ce, me, bott, _, _ = _roofline(gpu, ops, traffic, tiles, e.working_set_bytes, e.num_stages)
        if best is None or t < best[0]:
            best = (t, ce, me, bott, traffic, ops, tiles,
                    f"{mp.m0}x{mp.n0}x{mp.k0} {'-'.join(mp.loop_order)} C={e.num_stages}")
    if best is None:
        raise ValueError(f"{label}: no feasible fused mapping (aux SMEM too large?)")
    t, ce, me, bott, traffic, ops, tiles, det = best
    return KTime(label, t, ce, me, bott, traffic, ops, tiles, det)


# --------------------------------------------------------------------------- #
# F6: full-FFN GEMM-GEMM fusion (up_gate -> SwiGLU -> down), intermediate on-chip #
# --------------------------------------------------------------------------- #
def estimate_ffn_fused(label, count, gpu: GpuModel) -> KTime:
    """out[M,HIDDEN] = down(SwiGLU(up_gate(x[M,HIDDEN]))) as one kernel, intermediate on chip.

    Per m0-row-block: read x once, write out once, read BOTH weight matrices once; mt=M/m0
    blocks -> weights re-read mt x. Buffer holds the resident activated + out accumulators.
    Enumerate m0 (SMEM-gated, as in the area F6 model).
    """
    m = TOKENS_PER_EXPERT
    w_ug = HIDDEN * (2 * INTERMEDIATE) * BPE
    w_dn = INTERMEDIATE * HIDDEN * BPE
    x_out = 2 * m * HIDDEN * BPE
    ops_inst = (2 * m * (2 * INTERMEDIATE) * HIDDEN) + (2 * m * HIDDEN * INTERMEDIATE)
    # Realistic SMEM: the down GEMM contracts over K=INTERMEDIATE, so the FULL activated
    # row activated[m0,:INTERMEDIATE] must be resident for the block. That (m0*INTERMEDIATE
    # *bpe) is the binding constraint; add a modest overhead for the down-output accumulator
    # tile (m0 x 128) and double-buffered weight/input k-slice streams (~16 KiB).
    STREAM_OVERHEAD = 16 * 1024
    best = None
    for m0 in divisors(m):
        if m0 < MMA_MIN_BM:
            continue
        mt = m // m0
        resident = m0 * INTERMEDIATE * BPE + m0 * 128 * BPE + STREAM_OVERHEAD
        buffer = resident
        if buffer > gpu.smem_per_block_bytes:
            continue                                           # row-block doesn't fit SMEM
        traffic_inst = x_out + mt * (w_ug + w_dn)              # weights re-read mt x
        traffic = count * traffic_inst
        ops = count * ops_inst
        tiles = count * mt * max(1, HIDDEN // 128)             # out[m0,HIDDEN] tiled ~128 in N
        t, ce, me, bott, _, _ = _roofline(gpu, ops, traffic, tiles, buffer, 2)
        if best is None or t < best[0]:
            best = (t, ce, me, bott, traffic, ops, tiles, f"m0={m0} mt={mt} resident={resident//1024}KiB")
    if best is None:
        # No row-block fits: the full-FFN fusion is INFEASIBLE on this GPU's SMEM.
        return KTime(label, float("inf"), float("inf"), float("inf"), "infeasible",
                     0.0, 0.0, 0, detail=f"no m0>={MMA_MIN_BM} fits SMEM/block "
                     f"({gpu.smem_per_block_bytes//1024} KiB); needs >= {INTERMEDIATE*MMA_MIN_BM*BPE//1024} KiB activated")
    t, ce, me, bott, traffic, ops, tiles, det = best
    return KTime(label, t, ce, me, bott, traffic, ops, tiles, det)


# --------------------------------------------------------------------------- #
# The six fusions: (fused kernels) vs (unfused kernels)                          #
# --------------------------------------------------------------------------- #
def _residual_aux(m0, n0):
    return m0 * n0 * BPE                      # hold the residual output tile on chip


def _residual_rms_aux(m0, n0):
    return m0 * n0 * BPE + n0 * BPE + m0 * 4  # residual tile + gamma tile + fp32 RMS stat


def fusion_specs(gpu: GpuModel):
    """Return {name: (unfused_kernels[], fused_kernels[])} as callables producing KTime."""
    m_o, n_o, k_o = MLA_O
    m_u, n_u, k_u = UP_GATE
    m_d, n_d, k_d = DOWN
    return {
        "F1  FlashAttn + residual": (
            [lambda: estimate_gemm_grouped("mla_o", m_o, n_o, k_o, 1, gpu),
             lambda: estimate_vector_kernel("residual", RESIDUAL_TRAFFIC, gpu)],
            [lambda: estimate_fused_gemm("mla_o+residual", m_o, n_o, k_o, 1,
                Epilogue(extra_hbm_once=BATCH * HIDDEN * BPE, aux_smem_per_tile=_residual_aux), gpu)],
        ),
        "F2  FlashAttn + residual + RMSNorm": (
            [lambda: estimate_gemm_grouped("mla_o", m_o, n_o, k_o, 1, gpu),
             lambda: estimate_vector_kernel("residual", RESIDUAL_TRAFFIC, gpu),
             lambda: estimate_vector_kernel("rmsnorm", RMSNORM_TRAFFIC, gpu)],
            [lambda: estimate_fused_gemm("mla_o+residual+rms", m_o, n_o, k_o, 1,
                Epilogue(extra_hbm_once=BATCH * HIDDEN * BPE + BATCH * 4,
                         aux_smem_per_tile=_residual_rms_aux), gpu)],
        ),
        "F3  RMSNorm + up_gate": (
            [lambda: estimate_gemm_grouped("up_gate", m_u, n_u, k_u, EXPERTS, gpu),
             lambda: estimate_vector_kernel("rmsnorm", RMSNORM_TRAFFIC, gpu)],
            [lambda: estimate_fused_gemm("up_gate+rms", m_u, n_u, k_u, EXPERTS,
                Epilogue(aux_smem_per_tile=lambda m0, n0: m0 * 4), gpu)],
        ),
        "F4  up_gate + activation": (
            [lambda: estimate_gemm_grouped("up_gate", m_u, n_u, k_u, EXPERTS, gpu),
             lambda: estimate_vector_kernel("activation", ACTIVATION_TRAFFIC, gpu)],
            [lambda: estimate_fused_gemm("up_gate+swiglu", m_u, n_u, k_u, EXPERTS,
                Epilogue(out_factor=0.5), gpu)],   # write activated (N/2), not raw gate+up
        ),
        "F5  activation + down": (
            [lambda: estimate_gemm_grouped("down", m_d, n_d, k_d, EXPERTS, gpu),
             lambda: estimate_vector_kernel("activation", ACTIVATION_TRAFFIC, gpu)],
            [lambda: estimate_fused_gemm("swiglu+down", m_d, n_d, k_d, EXPERTS,
                Epilogue(a_factor=2.0), gpu)],      # down reads 2x-wide gate+up
        ),
        "F6  up_gate + activation + down": (
            [lambda: estimate_gemm_grouped("up_gate", m_u, n_u, k_u, EXPERTS, gpu),
             lambda: estimate_vector_kernel("activation", ACTIVATION_TRAFFIC, gpu),
             lambda: estimate_gemm_grouped("down", m_d, n_d, k_d, EXPERTS, gpu)],
            [lambda: estimate_ffn_fused("ffn(up_gate+swiglu+down)", EXPERTS, gpu)],
        ),
    }


def run(gpu: GpuModel, verbose: bool = False) -> None:
    print(f"\n################  {gpu.name}  ################")
    print(f"peak tensor {gpu.peak_tensor_flops/1e12:.1f} TFLOP/s | HBM {gpu.bw_bytes_per_s/1e9:.0f} GB/s | "
          f"{gpu.num_sm} SMs | ridge OI {gpu.peak_tensor_flops/gpu.bw_bytes_per_s:.0f} FLOP/B | decode batch {BATCH}")
    print(f"{'fusion':<36} {'unfused ms':>11} {'fused ms':>10} {'speedup':>8} {'verdict':>9}")
    for name, (unfused_fns, fused_fns) in fusion_specs(gpu).items():
        unf = [f() for f in unfused_fns]
        fus = [f() for f in fused_fns]
        unf_t = sum(k.time_s for k in unf)
        fus_t = sum(k.time_s for k in fus)
        if not math.isfinite(fus_t):
            print(f"{name:<36} {unf_t*1e3:>11.4f} {'INFEASIBLE':>10} {'--':>8} {'skip':>9}")
            if verbose:
                for k in fus:
                    print(f"      FUSED    {k.label:<26} {'infeasible':>12}  {k.detail}")
            continue
        speedup = unf_t / fus_t if fus_t > 0 else float("inf")
        verdict = "FUSE" if fus_t < unf_t else "skip"
        print(f"{name:<36} {unf_t*1e3:>11.4f} {fus_t*1e3:>10.4f} {speedup:>7.3f}x {verdict:>9}")
        if verbose:
            for k in unf:
                print(f"      unfused  {k.label:<26} {k.time_s*1e3:>9.4f} ms  [{k.bottleneck}]  {k.detail}")
            for k in fus:
                print(f"      FUSED    {k.label:<26} {k.time_s*1e3:>9.4f} ms  [{k.bottleneck}]  {k.detail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=sorted(GPUS) + ["both"], default="both",
                    help="GPU profile (default: both h100-sxm and rtx4060-measured)")
    ap.add_argument("--verbose", action="store_true", help="per-kernel breakdown")
    args = ap.parse_args()
    gpus = (["h100-sxm", "rtx4060-measured"] if args.gpu == "both" else [args.gpu])
    for g in gpus:
        run(GPUS[g], verbose=args.verbose)


if __name__ == "__main__":
    main()
