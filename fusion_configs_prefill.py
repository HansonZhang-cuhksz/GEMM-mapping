"""Exhaustive fusion-config enumeration for PREFILL (GLM-5.2 MoE, H100).

Same 40 configs as decode (fusion_configs.py), but the layer now includes the prefill attention
core (O(T^2) causal flash) as a FIXED prefix uniform across all configs. throughput = tokens/s =
T / layer_time. Sweep prefill token count T <= 131072.
"""
import argparse
import itertools
import json
import math

import fusion_configs as fc
import fusion_time_estimator as fte
from fusion_time_estimator import estimate_gemm_grouped, estimate_vector_kernel
from gemm_time_estimator import GPUS

HIDDEN, KV, BPE = fte.HIDDEN, fte.N_HEADS * fte.V_HEAD_DIM, fte.BPE
N_HEADS, QK_DIM, V_DIM = fte.N_HEADS, 192, fte.V_HEAD_DIM   # MLA head dims (qk=192, v=256; estimates)
KV_LATENT = 512                                            # MLA compresses K/V to a small latent (not KV=16384)


def flash_time(T, gpu):
    """Causal flash-attention core: ops = N_HEADS*T^2*(qk+v) flops (MAC*2 * causal 0.5 = 1)."""
    ops = N_HEADS * T * T * (QK_DIM + V_DIM)
    compute = ops / gpu.peak_tensor_flops
    mem = 4 * T * KV * BPE / gpu.bw_bytes_per_s      # read Q,K,V + write O (~O(T), hidden at large T)
    return max(compute, mem), ops


def attn_prefix(T, gpu):
    """Fixed prefill attention cost before mla_o: pre-norm + MLA q/kv projections + flash core.

    MLA projections (not one full KV=16384 GEMM): q_proj [T, N_HEADS*qk, HIDDEN] + kv down-proj
    [T, latent=512, HIDDEN] + kv up-proj [T, N_HEADS*v, latent].
    """
    pre = estimate_vector_kernel("pre_norm", T * HIDDEN * BPE + T * 4, gpu).time_s
    def g(m, n, k):
        try:
            return estimate_gemm_grouped("proj", m, n, k, 1, gpu).time_s
        except ValueError:
            return 0.0
    qkv = (g(T, N_HEADS * QK_DIM, HIDDEN)          # q projection
           + g(T, KV_LATENT, HIDDEN)               # kv down-projection to latent
           + g(T, N_HEADS * V_DIM, KV_LATENT))     # kv up-projection (K=512 -> cheap)
    fl, _ = flash_time(T, gpu)
    return pre + qkv + fl


def prefill_layer(attn, ffn, res2, T, gpu):
    fc.set_batch(T)
    lt, feas = fc.layer_time(attn, ffn, res2, gpu)
    return attn_prefix(T, gpu) + lt, feas


def sweep(gpu, tokens):
    out = {}
    for attn, ffn, res2 in itertools.product(fc.ATTN, fc.FFN, fc.RES2):
        cfg = fc.name(attn, ffn, res2)
        curve = []
        for T in tokens:
            tt, feas = prefill_layer(attn, ffn, res2, T, gpu)
            curve.append({"B": T, "ms": tt * 1e3 if math.isfinite(tt) else None,
                          "tput": (T / tt) if math.isfinite(tt) and tt > 0 else None, "feasible": feas})
        out[cfg] = curve
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="h100-sxm")
    ap.add_argument("--out", default="prefill_configs.json")
    args = ap.parse_args()
    gpu = GPUS[args.gpu]
    # incl. 3*2^k refinement points (10240/11264/14336) to resolve the jagged peak (per audit)
    tokens = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 11264, 12288, 14336, 16384,
              24576, 32768, 49152, 65536, 98304, 131072]
    data = sweep(gpu, tokens)
    json.dump({"gpu": gpu.name, "batches": tokens, "curves": data}, open(args.out, "w"), indent=1)
    ranked = sorted(((c, max((p["tput"] for p in cv if p["tput"]), default=0),
                      max((p for p in cv if p["tput"]), key=lambda p: p["tput"])["B"] if any(p["tput"] for p in cv) else None)
                     for c, cv in data.items()), key=lambda x: -x[1])
    print(f"# PREFILL {gpu.name}: {len(data)} configs, T<=131072")
    print(f"{'rank':>4} {'config':>14} {'peak Mtok/s':>12} {'@T':>8}")
    for i, (c, pk, tb) in enumerate(ranked[:6], 1):
        print(f"{i:>4} {c:>14} {pk/1e6:>12.4f} {tb:>8}")
    print(f"     {ranked[-1][0]:>14} {ranked[-1][1]/1e6:>12.4f}  (worst)")
    # attention-domination check + best-vs-unfused at each T
    best = ranked[0][0]; unf = "S1-N0-r2s"
    print(f"\nBEST {best} peak {ranked[0][1]/1e6:.4f} Mtok/s @ T={ranked[0][2]}")
    print(f"{'T':>8} {'best Mtok/s':>12} {'unf Mtok/s':>11} {'best/unf':>9} {'flash %':>8}")
    for i, T in enumerate(tokens):
        tb = data[best][i]["tput"]; tu = data[unf][i]["tput"]
        fl = flash_time(T, gpu)[0]; tot = data[best][i]["ms"] / 1e3
        if tb and tu:
            print(f"{T:>8} {tb/1e6:>12.4f} {tu/1e6:>11.4f} {tb/tu:>8.3f}x {100*fl/tot:>7.0f}%")
    print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
