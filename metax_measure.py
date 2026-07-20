"""Measure real GEMM + unfused-chain times on the physical MetaX C500 (fusion env, bf16).

Self-contained (no estimator import — runs in py3.10 'fusion'). Saves metax_measured.json for
metax_compare.py (area env) to compare against the snowcat-roofline estimator with the C500 model.
Dimension sets mirror the estimation studies. Timing: bf16, warmup, cuda Events, median.
"""
import json, statistics, itertools
import torch

DEV = "cuda:0"


def _iters(m, n, k):
    flop = 2 * m * n * k
    if flop > 5e11: return 8, 3
    if flop > 5e10: return 20, 5
    return 60, 15


def measure(m, n, k, dtype=torch.bfloat16):
    it, wu = _iters(m, n, k)
    a = torch.randn(m, k, device=DEV, dtype=dtype)
    b = torch.randn(k, n, device=DEV, dtype=dtype)
    for _ in range(wu):
        c = a @ b
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); c = a @ b; e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    t = statistics.median(ts)
    del a, b, c; torch.cuda.empty_cache()
    return {"m": m, "n": n, "k": k, "t_s": t, "tflops": 2 * m * n * k / t / 1e12}


def measure_chain(M, widths):
    """Unfused chain: Y = X @ W1 @ ... @ WL, intermediates materialized in HBM. Sum of matmuls."""
    L = len(widths) - 1
    it, wu = _iters(M, max(widths), max(widths))
    x0 = torch.randn(M, widths[0], device=DEV, dtype=torch.bfloat16)
    Ws = [torch.randn(widths[s], widths[s + 1], device=DEV, dtype=torch.bfloat16) for s in range(L)]
    def run():
        h = x0
        for W in Ws:
            h = h @ W
        return h
    for _ in range(wu):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    t = statistics.median(ts)
    del x0, Ws; torch.cuda.empty_cache()
    return {"M": M, "widths": widths, "L": L, "t_s": t}


def case_dims(a_a, a_b, a_d, MAXDIM=16384):
    A = {"square": 1.0, "tall": 0.25, "wide": 4.0}
    rels = [1.0]
    for a in (a_a, a_b, a_d):
        rels.append(rels[-1] * A[a])
    scale = MAXDIM / max(rels)
    return tuple(int(round(r * scale)) for r in rels)


def main():
    p = torch.cuda.get_device_properties(0)
    out = {"device": p.name, "sm": p.multi_processor_count, "l2_mib": p.L2_cache_size / 2**20,
           "gemms": {}, "chains": {}}
    seen = {}
    def add(group, m, n, k):
        key = (m, n, k)
        if key not in seen:
            seen[key] = measure(m, n, k)
        out["gemms"].setdefault(group, []).append(seen[key])

    # 1. validation squares + tall/wide
    for n in [1024, 2048, 4096, 8192]:
        add("square", n, n, n)
    for (m, n, k) in [(16384, 4096, 4096), (4096, 16384, 4096), (16384, 16384, 4096),
                      (4096, 4096, 16384), (2048, 8192, 8192)]:
        add("rect", m, n, k)

    # 2. GLM-5.2 decode GEMMs (batch 2048)
    for (m, n, k, tag) in [(2048, 6144, 16384, "mla_o"), (2048, 256, 6144, "router"),
                           (64, 4096, 6144, "up_gate"), (64, 6144, 2048, "down")]:
        add("glm_decode", m, n, k)

    # 3. focused flash-attn regime (M=8192): GEMM1 [M,N1,K1], GEMM2 [M,N2,N1]
    for (N1, N2, K1) in [(4096, 128, 512), (4096, 128, 4096), (4096, 256, 16384),
                         (8192, 128, 512), (8192, 256, 2048)]:
        add("focused", 8192, N1, K1)
        add("focused", 8192, N2, N1)

    # 4. multi-GEMM chain stages [131072, w, w]
    for w in [128, 256, 512, 1024, 2048]:
        add("multi_stage", 131072, w, w)

    # 5. a few 27-shape cases (GEMM1 + GEMM2)
    for aspects in [("square", "square", "square"), ("tall", "tall", "square"),
                    ("tall", "tall", "tall"), ("wide", "tall", "tall")]:
        M, K1, N1, N2 = case_dims(*aspects)
        add("shape27", M, N1, K1)
        add("shape27", M, N2, N1)

    # unfused chains
    # 2-GEMM focused winners (M=8192)
    out["chains"]["chain2_8192x4096x1024x128"] = measure_chain(8192, [512, 4096, 128])   # X[8192,512]@[512,4096]@[4096,128]
    # multi-GEMM uniform chains M=131072
    for w in [128, 256, 512]:
        for L in [3, 6]:
            out["chains"][f"multi_w{w}_L{L}"] = measure_chain(131072, [w] * (L + 1))
    # square chains
    for n in [1024, 2048]:
        out["chains"][f"square_n{n}_L3"] = measure_chain(n, [n] * 4)

    with open("metax_measured.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[wrote metax_measured.json] {sum(len(v) for v in out['gemms'].values())} gemm-slots, "
          f"{len(out['chains'])} chains, {len(seen)} unique GEMMs")
    print(f"device: {out['device']} {out['sm']} SM, L2 {out['l2_mib']:.0f} MiB")


if __name__ == "__main__":
    main()
