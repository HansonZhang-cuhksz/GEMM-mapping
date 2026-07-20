"""Batch-size sweep: GLM-5.2 MoE full decode layer, unfused vs fused pass, throughput = tokens/time.

Hypothesis: the UNFUSED pass gives each kernel full SMEM for one algorithmic pass, so it amortizes
weights better at LARGE batch; the FUSED pass (F6 on-chip FFN + F2 attention) holds the intermediate
in SMEM, so LARGE batch -> per-expert tokens exceed the SMEM row-block -> weight re-reads -> it prefers
SMALL batch. Find each pass's optimal batch and compare best throughput. H100 model.

Reuses fusion_time_estimator's validated kernel + F2/F6 fused physics, parametrized by batch.
"""
import argparse
import json
import math

import fusion_time_estimator as fte
from fusion_time_estimator import (
    Epilogue, _residual_rms_aux, estimate_ffn_fused, estimate_fused_gemm,
    estimate_gemm_grouped, estimate_vector_kernel,
)
import math as _m
from gemm_time_estimator import GPUS

HIDDEN, INT, EXPERTS, TOPK, BPE = fte.HIDDEN, fte.INTERMEDIATE, fte.EXPERTS, fte.TOP_K, fte.BPE
KV = fte.N_HEADS * fte.V_HEAD_DIM   # 16384


def set_batch(B):
    fte.BATCH = B
    fte.TOKENS_PER_EXPERT = B * TOPK // EXPERTS
    fte.MLA_O = (B, HIDDEN, KV)
    fte.UP_GATE = (fte.TOKENS_PER_EXPERT, 2 * INT, HIDDEN)
    fte.DOWN = (fte.TOKENS_PER_EXPERT, HIDDEN, INT)
    fte.RESIDUAL_TRAFFIC = 3 * B * HIDDEN * BPE
    fte.RMSNORM_TRAFFIC = B * HIDDEN * BPE + B * 4
    fte.ACTIVATION_TRAFFIC = (B * TOPK) * (2 * INT + INT) * BPE


def _g(label, m, n, k, count, gpu):
    try:
        return estimate_gemm_grouped(label, m, n, k, count, gpu).time_s
    except ValueError:
        return float("inf")


def unfused_layer(gpu):
    B, Me = fte.BATCH, fte.TOKENS_PER_EXPERT
    return (_g("mla_o", B, HIDDEN, KV, 1, gpu) + _g("router", B, EXPERTS, HIDDEN, 1, gpu)
            + _g("up_gate", Me, 2 * INT, HIDDEN, EXPERTS, gpu) + _g("down", Me, HIDDEN, INT, EXPERTS, gpu)
            + estimate_vector_kernel("residual", fte.RESIDUAL_TRAFFIC, gpu).time_s
            + estimate_vector_kernel("rmsnorm", fte.RMSNORM_TRAFFIC, gpu).time_s
            + estimate_vector_kernel("swiglu", fte.ACTIVATION_TRAFFIC, gpu).time_s)


def fused_layer(gpu):
    """F2 (mla_o+residual+rms) + router + F6 (on-chip FFN). Returns (time, ffn_feasible)."""
    B, Me = fte.BATCH, fte.TOKENS_PER_EXPERT
    try:
        attn = estimate_fused_gemm("mla_o+res+rms", B, HIDDEN, KV, 1,
                                   Epilogue(extra_hbm_once=B * HIDDEN * BPE + B * 4,
                                            aux_smem_per_tile=_residual_rms_aux), gpu).time_s
    except ValueError:
        attn = _g("mla_o", B, HIDDEN, KV, 1, gpu)   # fall back if fused mla_o infeasible
    router = _g("router", B, EXPERTS, HIDDEN, 1, gpu)
    ffn = estimate_ffn_fused("ffn", EXPERTS, gpu)
    return attn + router + ffn.time_s, math.isfinite(ffn.time_s)


def smart_fused_layer(gpu):
    """Epilogue-only fusion: F2 attention + SwiGLU folded into up_gate epilogue (F4, out_factor 0.5),
    keeping up_gate & down as weight-amortized grouped GEMMs. Fuses vector ops WITHOUT forfeiting
    weight amortization -> scales with batch (contrast to F6)."""
    B, Me = fte.BATCH, fte.TOKENS_PER_EXPERT
    try:
        attn = estimate_fused_gemm("mla_o+res+rms", B, HIDDEN, KV, 1,
                                   Epilogue(extra_hbm_once=B * HIDDEN * BPE + B * 4,
                                            aux_smem_per_tile=_residual_rms_aux), gpu).time_s
    except ValueError:
        attn = _g("mla_o", B, HIDDEN, KV, 1, gpu)
    router = _g("router", B, EXPERTS, HIDDEN, 1, gpu)
    try:
        upg = estimate_fused_gemm("up_gate+swiglu", Me, 2 * INT, HIDDEN, EXPERTS,
                                  Epilogue(out_factor=0.5), gpu).time_s   # write activated (half-width)
    except ValueError:
        upg = _g("up_gate", Me, 2 * INT, HIDDEN, EXPERTS, gpu)
    down = _g("down", Me, HIDDEN, INT, EXPERTS, gpu)   # stays a grouped GEMM (weight-amortized)
    return attn + router + upg + down


def sweep(gpu, batches):
    rows = []
    for B in batches:
        set_batch(B)
        Me = fte.TOKENS_PER_EXPERT
        u = unfused_layer(gpu)
        f, feas = fused_layer(gpu)
        sm = smart_fused_layer(gpu)
        rows.append({"B": B, "Me": Me, "unf_ms": u * 1e3, "fus_ms": f * 1e3 if math.isfinite(f) else None,
                     "smart_ms": sm * 1e3, "unf_tput": B / u, "fus_tput": (B / f) if math.isfinite(f) else None,
                     "smart_tput": B / sm, "ffn_feasible": feas,
                     "speedup": (u / f) if math.isfinite(f) else None, "smart_speedup": u / sm})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="h100-sxm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu = GPUS[args.gpu]
    # batches: multiples of 32 so tokens/expert = B/32 is integer; powers of 2
    batches = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    rows = sweep(gpu, batches)
    print(f"# GLM-5.2 MoE full decode layer — {gpu.name}  (throughput = tokens/s)")
    print(f"{'batch':>7} {'tok/exp':>7} {'unf Mtok/s':>11} {'F6 Mtok/s':>11} {'smart Mtok/s':>12} "
          f"{'F6/unf':>8} {'smart/unf':>9}")
    for r in rows:
        ft = f"{r['fus_tput']/1e6:.3f}" if r['fus_tput'] else "INFEAS"
        sp = f"{r['speedup']:.3f}x" if r['speedup'] else "--"
        print(f"{r['B']:>7} {r['Me']:>7} {r['unf_tput']/1e6:>11.3f} {ft:>11} {r['smart_tput']/1e6:>12.3f} "
              f"{sp:>8} {r['smart_speedup']:>8.3f}x")
    ubest = max(rows, key=lambda r: r['unf_tput'])
    ffeas = [r for r in rows if r['fus_tput']]
    fbest = max(ffeas, key=lambda r: r['fus_tput']) if ffeas else None
    sbest = max(rows, key=lambda r: r['smart_tput'])
    print(f"\n  UNFUSED     optimal: batch {ubest['B']} -> {ubest['unf_tput']/1e6:.3f} Mtok/s")
    if fbest:
        print(f"  F6 (on-chip)optimal: batch {fbest['B']} -> {fbest['fus_tput']/1e6:.3f} Mtok/s  "
              f"(ratio {fbest['fus_tput']/ubest['unf_tput']:.3f}x)")
    print(f"  SMART epi-fusedoptimal: batch {sbest['B']} -> {sbest['smart_tput']/1e6:.3f} Mtok/s  "
          f"(ratio {sbest['smart_tput']/ubest['unf_tput']:.3f}x)")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
