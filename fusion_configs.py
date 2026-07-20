"""Exhaustive fusion-configuration enumeration + throughput-vs-batch sweep (GLM-5.2 MoE decode, H100).

40 configs = ATTN(5) x FFN(4) x RES2(2). Throughput = tokens/s = B / layer_time. Estimation via
fusion_time_estimator's validated kernel/fused physics. Batch bounded <= 16384.
"""
import argparse
import itertools
import json
import math

import fusion_time_estimator as fte
from fusion_time_estimator import (
    Epilogue, _residual_aux, _residual_rms_aux, estimate_ffn_fused, estimate_fused_gemm,
    estimate_gemm_grouped, estimate_vector_kernel,
)
from gemm_time_estimator import GPUS

HIDDEN, INT, EXPERTS, TOPK, BPE = fte.HIDDEN, fte.INTERMEDIATE, fte.EXPERTS, fte.TOP_K, fte.BPE
KV = fte.N_HEADS * fte.V_HEAD_DIM

ATTN = ["S1", "S2", "S3", "S4", "S5"]   # residual1 + RMSNorm placement
FFN = ["N0", "N4", "N5", "N6"]           # SwiGLU + FFN GEMM structure
RES2 = [False, True]                      # post-FFN residual fused into down/F6 epilogue


def set_batch(B):
    fte.BATCH = B
    fte.TOKENS_PER_EXPERT = B * TOPK // EXPERTS
    fte.RESIDUAL_TRAFFIC = 3 * B * HIDDEN * BPE
    fte.RMSNORM_TRAFFIC = B * HIDDEN * BPE + B * 4
    fte.ACTIVATION_TRAFFIC = (B * TOPK) * (2 * INT + INT) * BPE


def _g(label, m, n, k, count, gpu):
    try:
        return estimate_gemm_grouped(label, m, n, k, count, gpu).time_s
    except ValueError:
        return float("inf")


def _fg(label, m, n, k, count, epi, gpu, fallback):
    try:
        return estimate_fused_gemm(label, m, n, k, count, epi, gpu).time_s
    except ValueError:
        return fallback()


def _norm_prologue_epi():
    return Epilogue(aux_smem_per_tile=lambda m0, n0: m0 * 4)   # F3: RMS stat, input already read


def layer_time(attn, ffn, res2, gpu):
    B, Me = fte.BATCH, fte.TOKENS_PER_EXPERT
    t = 0.0
    RES = fte.RESIDUAL_TRAFFIC
    RMS = fte.RMSNORM_TRAFFIC
    ACT = fte.ACTIVATION_TRAFFIC
    norm_in_ffn = attn in ("S4", "S5")

    # ---- attention ----
    if attn == "S3":                    # mla_o + res1 + norm (F2)
        t += _fg("mla_o+res+rms", B, HIDDEN, KV, 1,
                 Epilogue(extra_hbm_once=B * HIDDEN * BPE + B * 4, aux_smem_per_tile=_residual_rms_aux),
                 gpu, lambda: _g("mla_o", B, HIDDEN, KV, 1, gpu) + RES / gpu.bw_bytes_per_s + RMS / gpu.bw_bytes_per_s)
    elif attn in ("S2", "S5"):          # mla_o + res1 (F1); norm elsewhere
        t += _fg("mla_o+res", B, HIDDEN, KV, 1,
                 Epilogue(extra_hbm_once=B * HIDDEN * BPE, aux_smem_per_tile=_residual_aux),
                 gpu, lambda: _g("mla_o", B, HIDDEN, KV, 1, gpu) + RES / gpu.bw_bytes_per_s)
    else:                                # S1, S4: mla_o alone; res1 standalone
        t += _g("mla_o", B, HIDDEN, KV, 1, gpu)
        t += estimate_vector_kernel("res1", RES, gpu).time_s
    # RMSNorm standalone iff not folded into attn (S3) and not into ffn (S4/S5)
    if attn in ("S1", "S2"):
        t += estimate_vector_kernel("rmsnorm", RMS, gpu).time_s

    # ---- router (always standalone) ----
    t += _g("router", B, EXPERTS, HIDDEN, 1, gpu)

    # ---- FFN ----
    ug_epi_extra = {}
    if norm_in_ffn:
        ug_epi_extra = dict(aux_smem_per_tile=lambda m0, n0: m0 * 4)   # norm prologue aux
    feas = True
    def upg_plain():
        return (_fg("up_gate", Me, 2 * INT, HIDDEN, EXPERTS, Epilogue(**ug_epi_extra), gpu,
                    lambda: _g("up_gate", Me, 2 * INT, HIDDEN, EXPERTS, gpu))
                if norm_in_ffn else _g("up_gate", Me, 2 * INT, HIDDEN, EXPERTS, gpu))

    if ffn == "N6":                      # on-chip full FFN (SwiGLU inside; norm prologue if S4/S5)
        ffnk = estimate_ffn_fused("ffn", EXPERTS, gpu).time_s
        t += ffnk
        feas = math.isfinite(ffnk)
    elif ffn == "N4":                    # up_gate + SwiGLU (F4, half-width out) ; down separate
        swv = estimate_vector_kernel("swiglu", ACT, gpu).time_s
        t += _fg("up_gate+swiglu", Me, 2 * INT, HIDDEN, EXPERTS, Epilogue(out_factor=0.5, **ug_epi_extra),
                 gpu, lambda: _g("up_gate", Me, 2 * INT, HIDDEN, EXPERTS, gpu) + swv)  # fallback keeps swiglu cost
        t += _g("down", Me, HIDDEN, INT, EXPERTS, gpu)
    elif ffn == "N5":                    # up_gate ; SwiGLU + down (F5, 2x-wide in)
        swv = estimate_vector_kernel("swiglu", ACT, gpu).time_s
        t += upg_plain()
        t += _fg("swiglu+down", Me, HIDDEN, INT, EXPERTS, Epilogue(a_factor=2.0), gpu,
                 lambda: _g("down", Me, HIDDEN, INT, EXPERTS, gpu) + swv)
    else:                                # N0: up_gate + swiglu(vec) + down (all separate)
        t += upg_plain()
        t += estimate_vector_kernel("swiglu", ACT, gpu).time_s
        t += _g("down", Me, HIDDEN, INT, EXPERTS, gpu)

    # post-FFN residual2 on the [B,HIDDEN] combined output (once, not per-expert).
    # standalone: full add (read out + read res, write sum = 3x). fused into the expert-combine
    # epilogue: only the residual read folded in (~1x, output write shared with combine).
    t += estimate_vector_kernel("res2", (1 if res2 else 3) * B * HIDDEN * BPE, gpu).time_s
    return t, feas


def name(attn, ffn, res2):
    return f"{attn}-{ffn}-{'r2f' if res2 else 'r2s'}"


def sweep(gpu, batches):
    out = {}
    for attn, ffn, res2 in itertools.product(ATTN, FFN, RES2):
        cfg = name(attn, ffn, res2)
        curve = []
        for B in batches:
            set_batch(B)
            tt, feas = layer_time(attn, ffn, res2, gpu)
            curve.append({"B": B, "ms": tt * 1e3 if math.isfinite(tt) else None,
                          "tput": (B / tt) if math.isfinite(tt) and tt > 0 else None, "feasible": feas})
        out[cfg] = curve
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="h100-sxm")
    ap.add_argument("--out", default="fusion_configs.json")
    args = ap.parse_args()
    gpu = GPUS[args.gpu]
    # denser grid incl. 3*2^k points (better divisor tiles) so the jagged peak isn't grid-missed
    batches = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
               10240, 12288, 14336, 15360, 16384]
    data = sweep(gpu, batches)
    json.dump({"gpu": gpu.name, "batches": batches, "curves": data}, open(args.out, "w"), indent=1)
    # best config by peak throughput over the batch range
    best = None
    for cfg, curve in data.items():
        peak = max((p["tput"] for p in curve if p["tput"]), default=0)
        bestB = max((p for p in curve if p["tput"]), key=lambda p: p["tput"], default=None)
        if best is None or peak > best[1]:
            best = (cfg, peak, bestB["B"] if bestB else None)
    print(f"# {gpu.name}: {len(data)} configs, batch<=16384")
    ranked = sorted(((c, max((p["tput"] for p in cv if p["tput"]), default=0)) for c, cv in data.items()),
                    key=lambda x: -x[1])
    print(f"{'rank':>4} {'config':>14} {'peak Mtok/s':>12}")
    for i, (c, pk) in enumerate(ranked[:8], 1):
        print(f"{i:>4} {c:>14} {pk/1e6:>12.4f}")
    print(f"...\n{'':>4} {ranked[-1][0]:>14} {ranked[-1][1]/1e6:>12.4f}  (worst)")
    print(f"\nBEST: {best[0]} -> {best[1]/1e6:.4f} Mtok/s at batch {best[2]}")
    print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
