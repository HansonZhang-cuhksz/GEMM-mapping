"""Sweep MODEL PARAMETERS and measure the fusion 'ceiling' on the physical C500 (fusion env).

Fusion ceiling = vec_ms / (gemm_ms + vec_ms) — the best-case benefit if the memory-bound vector op
is fully absorbed into the GEMM. Also classify the GEMM compute- vs memory-bound (achieved TF/s).
Tests whether GLM-5.2's ~0% fusion benefit is intrinsic to its params or shifts for other models.
"""
import json, statistics
import torch
import torch.nn.functional as F

DEV = "cuda:0"
PEAK = 226.0  # measured C500 bf16 TFLOP/s


def med(fn, iters=25, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


def bf(*s): return torch.randn(*s, device=DEV, dtype=torch.bfloat16)


def gemm_ms(m, n, k):
    a, b = bf(m, k), bf(k, n)
    t = med(lambda: a @ b, iters=15, warmup=6)
    del a, b; torch.cuda.empty_cache()
    return t, 2 * m * n * k / t / 1e12


def bmm_ms(e, m, n, k):
    a, b = bf(e, m, k), bf(e, k, n)
    t = med(lambda: torch.bmm(a, b), iters=15, warmup=6)
    del a, b; torch.cuda.empty_cache()
    return t, 2 * e * m * n * k / t / 1e12


def swiglu_ms(rows, inter):
    g, u = bf(rows, inter), bf(rows, inter)
    t = med(lambda: F.silu(g) * u, iters=30, warmup=10)
    del g, u; torch.cuda.empty_cache()
    return t


def add_ms(rows, cols):
    a, b = bf(rows, cols), bf(rows, cols)
    t = med(lambda: a + b, iters=30, warmup=10)
    del a, b; torch.cuda.empty_cache()
    return t


def bound(tf): return "compute" if tf > 0.7 * PEAK else "memory"


out = {"axis1_dim": [], "axis2_moe": [], "axis3_attn": []}

# Axis 1 — model DIM (dense FFN up_gate+SwiGLU, M=2048 tokens). intermediate=hidden.
print("# Axis 1: dense FFN, sweep hidden dim")
for H in [1024, 2048, 4096, 8192]:
    g, tf = gemm_ms(2048, 2 * H, H)          # up_gate: [2048, 2H, H]
    v = swiglu_ms(2048, H)                     # SwiGLU on [2048, H]
    ceil = v / (g + v) * 100
    out["axis1_dim"].append({"hidden": H, "gemm_ms": g, "tflops": tf, "vec_ms": v,
                             "ceiling_pct": ceil, "bound": bound(tf)})
    print(f"  H={H:>5}: gemm {g*1e3:7.3f}ms ({tf:5.0f}TF {bound(tf):>7}) + swiglu {v*1e3:6.3f}ms "
          f"-> fusion ceiling {ceil:4.1f}%")

# Axis 2 — MoE vs DENSE (fixed 16384 tokens, hidden 2048, intermediate 2048). experts sweep.
print("\n# Axis 2: MoE vs dense, sweep #experts (fixed 16384 tokens, hidden 2048)")
TOK, H = 16384, 2048
for E in [1, 8, 64, 256]:
    m = TOK // E
    g, tf = bmm_ms(E, m, 2 * H, H)            # grouped up_gate
    v = swiglu_ms(TOK, H)                      # swiglu over all tokens (same regardless of E)
    ceil = v / (g + v) * 100
    out["axis2_moe"].append({"experts": E, "tokens_per_expert": m, "gemm_ms": g, "tflops": tf,
                             "vec_ms": v, "ceiling_pct": ceil, "bound": bound(tf)})
    print(f"  E={E:>3} (m={m:>5}): gemm {g*1e3:7.3f}ms ({tf:5.0f}TF {bound(tf):>7}) "
          f"-> fusion ceiling {ceil:4.1f}%")

# Axis 3 — attention F1 (mla_o + residual), sweep batch.
print("\n# Axis 3: attention out-proj + residual, sweep batch")
for B in [256, 1024, 4096, 16384]:
    g, tf = gemm_ms(B, 6144, 16384)           # mla_o [B, 6144, 16384]
    v = add_ms(B, 6144)                        # residual add
    ceil = v / (g + v) * 100
    out["axis3_attn"].append({"batch": B, "gemm_ms": g, "tflops": tf, "vec_ms": v,
                              "ceiling_pct": ceil, "bound": bound(tf)})
    print(f"  B={B:>5}: gemm {g*1e3:7.3f}ms ({tf:5.0f}TF {bound(tf):>7}) + resid {v*1e3:6.3f}ms "
          f"-> fusion ceiling {ceil:4.1f}%")

with open("param_sweep.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n[wrote param_sweep.json]")
