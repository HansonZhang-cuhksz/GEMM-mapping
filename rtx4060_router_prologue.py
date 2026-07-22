"""T6.B (RTX4060_SIM_REAL_TASK.md) - router PROLOGUE fusion (b1 residual+RMSNorm, b2 residual-only).

Router = dense GEMM  logits[M,256] = X[M,6144] @ Wrouter[6144,256]  (m=M=tokens, n=EXPERTS=256,
k=HIDDEN=6144). X is produced by  h = attn_out + residual_in  then  x = RMSNorm(h).  A PROLOGUE
folds these into the router GEMM's A-load:
  (b2) residual-only:      logits = (attn + res) @ Wr
  (b1) residual + RMSNorm: logits = RMSNorm(attn + res) @ Wr   (gamma folded into
       Wp = Wr * gamma[:,None], precomputed UNTIMED on the host; 1/rms is a per-row
       epilogue scale co-accumulated over the SAME K-loop -- dual-accumulator hand kernel)

Six paths per config (T6.0.4): unfused | addmm (DROP: beta-accumulate is an OUTPUT-side
[M,256] add, the prologue residual is INPUT-side [M,6144] -- structurally inexpressible) |
compiled | nocg (= best-UNFUSED realization, NOT a fused candidate here) | forced (the key
stock test for b2: does inductor prologue-fusion fold the input add into the Triton GEMM
template?) | triton (hand kernel: guaranteed-fused upper bound for b2, the ONLY fully-fused
path for b1). Hand kernel: grid (cdiv(M,BM), 1), BN=256 full-N, HARD no-split-K.

UNITS: rtx4060_common.med_time and estimator KTime.time_s are SECONDS; every *_ms JSON field
is milliseconds. gain = unfused/fused (>1 => fusion FASTER).

Run (from GEMM-mapping/):
    python rtx4060_router_prologue.py --out rtx4060_router_prologue.json --t2-json rtx4060_peak.json
    python rtx4060_router_prologue.py --smoke --out /tmp/.../router_smoke.json --t2-json rtx4060_peak.json
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback

import torch
import torch.nn.functional as F
import torch._dynamo as _dynamo
import torch._inductor.config as _ic

sys.path.insert(0, "/home/shuhan/snowcat-demo/GEMM-mapping")

# Reuse the PROVEN T4 infrastructure verbatim (importing also enables inductor max-autotune
# at module load, exactly as the T4 sweep ran).
from rtx4060_fusion_measure import (  # noqa: E402
    _HAVE_TRITON,
    _cudagraph_copy_us,
    _maxabs,
    _rel_max,
    build_adjusted_profile,
    compile_and_time,
    judge_fused,
    profile_kernels,
    vendor_fused_ok,
)
from rtx4060_common import (  # noqa: E402  (med_time returns SECONDS; auto clock-warmer)
    DEV,
    ClockSampler,
    bf,
    geomean,
    med_time,
    save_json,
    sleep_cooldown,
)
from gemm_time_estimator import GPUS  # noqa: E402
from fusion_time_estimator import (  # noqa: E402
    BPE,
    Epilogue,
    estimate_fused_gemm,
    estimate_gemm_grouped,
    estimate_vector_kernel,
)

if _HAVE_TRITON:
    import triton
    import triton.language as tl

# Many shapes recompile the SAME code object per compile variant (dynamo caches on the code
# object; 2 variants x up to 9 M values). Raise well past torch's default of 8 (task: >=256).
for _attr in ("cache_size_limit", "recompile_limit"):
    if getattr(_dynamo.config, _attr, 0) < 256:
        try:
            setattr(_dynamo.config, _attr, 256)
        except Exception:
            pass

# GLM-5.2 constants (T6.0.1). Module-level (not imported from fusion_time_estimator) because
# --smoke shrinks HIDDEN so the estimator helpers stay dim-consistent with the smoke tensors.
HIDDEN = 6144
EXPERTS = 256
EPS = 1e-6

# --------------------------------------------------------------------------- #
# Spec caveats + model-labeling notes (spec B.1 / B.5) -- quoted into the JSON   #
# --------------------------------------------------------------------------- #
CAVEAT_MINOR_COST_CENTER = (
    "ABSOLUTE-MAGNITUDE CAVEAT (spec B.1, stated prominently): the router is a MINOR cost "
    "center - n=EXPERTS=256 is tiny, its FLOPs 2*M*256*6144 are ~16x below one up_gate "
    "expert-layer and dwarfed by mla_o. A healthy RELATIVE prologue speedup here is a SMALL "
    "ABSOLUTE layer saving.")
CAVEAT_SHARED_H_X = (
    "SHARED-TENSOR CAVEAT (spec T6.0.2c / B.1): in the real layer h = attn_out+residual_in "
    "and x = RMSNorm(h) are shared downstream, so the router-ATTRIBUTABLE saving is only the "
    "avoided re-read of x - SMALLER than this standalone microbenchmark shows, which charges "
    "the full residual+rmsnorm vector kernels to the router alone.")
RMS_TRAFFIC_NOTE = (
    "MODEL-LABELING NOTE (spec B.5): the b1 fused estimate decomposes as residual -> "
    "a_factor=2.0 (the RESIDUAL's second [M,HIDDEN] operand read; NOT F5's gate+up 2x-wide "
    "contraction and NOT the RMSNorm) + rmsnorm -> aux_smem m0*4 fp32 per-row partials ONLY "
    "(RMSNorm adds NO extra A-read: it is a free K-reduction over the already-resident "
    "A-tile), so the A-read is counted ONCE for the norm. The STANDALONE unfused rmsnorm "
    "baseline is modeled as 2*M*HIDDEN*BPE + M*4 traffic (read h, write normalized x, "
    "per-row stat) because a standalone kernel physically must write x - a DOCUMENTED "
    "DEVIATION from the estimator's RMSNORM_TRAFFIC constant (M*HIDDEN*BPE + M*4), which "
    "assumes the norm output is itself fused forward (the F3 assumption). For a pure-RMSNorm "
    "no-residual prologue variant use a_factor=1.0 + aux m0*4 (repo F3).")
ADDMM_DROP_REASON = (
    "DROP (spec B.3): torch.addmm / cuBLASLt beta-accumulate adds an [M,256] tensor to the "
    "GEMM OUTPUT; the router prologue residual is applied to the [M,6144] INPUT at a "
    "different stage - structurally inexpressible as a beta-accumulate epilogue (clean "
    "contrast to F1, where addmm was the star fused path). Recorded, not measured: "
    "addmm_ms=null.")


# --------------------------------------------------------------------------- #
# B.5 estimator wiring -- verbatim from the spec snippet                        #
# --------------------------------------------------------------------------- #
# b2 residual-only prologue: a_factor=2.0 = residual second-operand read
def est_router_residual_ms(M, gpu):
    g = estimate_gemm_grouped("router", M, EXPERTS, HIDDEN, 1, gpu)
    r = estimate_vector_kernel("residual", 3*M*HIDDEN*BPE, gpu)       # read attn, read res, write h
    f = estimate_fused_gemm("router+res_prologue", M, EXPERTS, HIDDEN, 1,
                            Epilogue(a_factor=2.0), gpu)              # residual second-operand read
    return (g.time_s + r.time_s)*1e3, f.time_s*1e3


# b1 residual + RMSNorm prologue: a_factor=2.0 (residual read) + aux m0*4 (rmsnorm K-reduction)
def est_router_residual_rms_ms(M, gpu):
    g = estimate_gemm_grouped("router", M, EXPERTS, HIDDEN, 1, gpu)
    r = estimate_vector_kernel("residual", 3*M*HIDDEN*BPE, gpu)
    n = estimate_vector_kernel("rmsnorm",  2*M*HIDDEN*BPE + M*4, gpu)  # read h, write x, per-row stat
    f = estimate_fused_gemm("router+res+rms_prologue", M, EXPERTS, HIDDEN, 1,
                            Epilogue(a_factor=2.0,                    # residual second-operand read
                                     aux_smem_per_tile=lambda m0, n0: m0*4), gpu)  # rmsnorm K-reduction
    return (g.time_s + r.time_s + n.time_s)*1e3, f.time_s*1e3


# --------------------------------------------------------------------------- #
# B.4 hand Triton kernels (grid (cdiv(M,BM), 1), BN=full-N=256, NO split-K)     #
# --------------------------------------------------------------------------- #
if _HAVE_TRITON:

    @triton.jit
    def _router_res_kernel(          # b2: acc += dot(a + r, W-tile)
        A, R, W, O, M, K,
        sam, sak, srm, srk, swk, swn, som, son,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)                 # BN == N (full width, no N-tiling)
        offs_k = tl.arange(0, BK)
        m_mask = offs_m[:, None] < M
        a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :] * sak
        r_ptrs = R + offs_m[:, None] * srm + offs_k[None, :] * srk
        w_ptrs = W + offs_k[:, None] * swk + offs_n[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kmask = offs_k < (K - k0)
            a = tl.load(a_ptrs, mask=m_mask & kmask[None, :], other=0.0)
            r = tl.load(r_ptrs, mask=m_mask & kmask[None, :], other=0.0)
            # residual prologue (tile-local): h = a + r, rounded to the input dtype exactly
            # as the eager baseline's bf16 add before the vendor GEMM
            h = (a.to(tl.float32) + r.to(tl.float32)).to(A.dtype.element_ty)
            w = tl.load(w_ptrs, mask=kmask[:, None], other=0.0)
            acc += tl.dot(h, w)
            a_ptrs += BK * sak
            r_ptrs += BK * srk
            w_ptrs += BK * swk
        o_ptrs = O + offs_m[:, None] * som + offs_n[None, :] * son
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask)

    @triton.jit
    def _router_res_rms_kernel(      # b1: dual-accumulator single pass, gamma folded into Wp
        A, R, W, O, M, K, eps,
        sam, sak, srm, srk, swk, swn, som, son,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        m_mask = offs_m[:, None] < M
        a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :] * sak
        r_ptrs = R + offs_m[:, None] * srm + offs_k[None, :] * srk
        w_ptrs = W + offs_k[:, None] * swk + offs_n[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        sumsq = tl.zeros((BM,), dtype=tl.float32)         # fp32 per-row sum of squares
        for k0 in range(0, K, BK):
            kmask = offs_k < (K - k0)
            a = tl.load(a_ptrs, mask=m_mask & kmask[None, :], other=0.0)
            r = tl.load(r_ptrs, mask=m_mask & kmask[None, :], other=0.0)
            hf = a.to(tl.float32) + r.to(tl.float32)      # residual prologue (tile-local)
            sumsq += tl.sum(hf * hf, axis=1)              # reduction, free (h resident)
            w = tl.load(w_ptrs, mask=kmask[:, None], other=0.0)
            acc += tl.dot(hf.to(A.dtype.element_ty), w)   # gamma already folded into Wp
            a_ptrs += BK * sak
            r_ptrs += BK * srk
            w_ptrs += BK * swk
        rms = tl.sqrt(sumsq / K + eps)                    # rms free at K-loop end
        out = acc / rms[:, None]                          # 1/rms = per-row epilogue scale
        o_ptrs = O + offs_m[:, None] * som + offs_n[None, :] * son
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=m_mask)

    def router_triton_b2(attn, res, Wr, BM=64, BK=32, num_warps=4, num_stages=2):
        M, K = attn.shape
        N = Wr.shape[1]
        assert (N & (N - 1)) == 0, "BN=N must be a power of two (EXPERTS=256)"
        O = torch.empty((M, N), device=attn.device, dtype=attn.dtype)
        grid = (triton.cdiv(M, BM), 1)   # HARD no-split-K: one CTA owns the full K-loop
        _router_res_kernel[grid](
            attn, res, Wr, O, M, K,
            attn.stride(0), attn.stride(1), res.stride(0), res.stride(1),
            Wr.stride(0), Wr.stride(1), O.stride(0), O.stride(1),
            BM=BM, BN=N, BK=BK, num_warps=num_warps, num_stages=num_stages)
        return O

    def router_triton_b1(attn, res, Wp, BM=64, BK=32, num_warps=4, num_stages=2):
        # single-pass sumsq is only correct if one CTA owns the full K-loop for its rows;
        # grid has NO K dimension by construction (fp32 numerics would catch any violation)
        M, K = attn.shape
        N = Wp.shape[1]
        assert (N & (N - 1)) == 0, "BN=N must be a power of two (EXPERTS=256)"
        O = torch.empty((M, N), device=attn.device, dtype=attn.dtype)
        grid = (triton.cdiv(M, BM), 1)
        _router_res_rms_kernel[grid](
            attn, res, Wp, O, M, K, EPS,
            attn.stride(0), attn.stride(1), res.stride(0), res.stride(1),
            Wp.stride(0), Wp.stride(1), O.stride(0), O.stride(1),
            BM=BM, BN=N, BK=BK, num_warps=num_warps, num_stages=num_stages)
        return O

    # (BM, BK, num_warps, num_stages) mini-autotune candidates: BM {64,128}, BK {32,64},
    # warps 4-8, stages 2-3 (spec B.4). BN is pinned to full-N=256.
    ROUTER_CANDS = [
        (64, 32, 4, 2), (64, 32, 8, 3), (64, 64, 4, 2), (64, 64, 8, 3),
        (128, 32, 4, 2), (128, 32, 8, 3), (128, 64, 4, 3), (128, 64, 8, 2),
    ]

    def tune_router(call, cands):
        """Fastest (BM,BK,w,s) via a short timing pass (mirrors tune_triton_swiglu);
        compile/launch failures (e.g. register-starved BM=128 x BN=256 tiles) are rejected."""
        best = None
        for cfg in cands:
            BM, BK, w, s = cfg
            try:
                t = med_time(lambda: call(BM, BK, w, s), iters=5, warmup=3)
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        return best[1] if best else None


# --------------------------------------------------------------------------- #
# Compile targets: one textually distinct def per (variant x compile mode) --    #
# dynamo caches on the code object and inductor config is NOT in its cache key  #
# --------------------------------------------------------------------------- #
def router_b2(attn, res, Wr):
    return (attn + res) @ Wr


def router_b2_nocg(attn, res, Wr):
    return (attn + res) @ Wr


def router_b2_forced(attn, res, Wr):
    return (attn + res) @ Wr


def router_b1(attn, res, Wr, gamma):
    h = attn + res
    return F.rms_norm(h, (h.shape[-1],), gamma, EPS) @ Wr


def router_b1_nocg(attn, res, Wr, gamma):
    h = attn + res
    return F.rms_norm(h, (h.shape[-1],), gamma, EPS) @ Wr


def router_b1_forced(attn, res, Wr, gamma):
    h = attn + res
    return F.rms_norm(h, (h.shape[-1],), gamma, EPS) @ Wr


COMPILE_FNS = {"b2": (router_b2, router_b2_nocg, router_b2_forced),
               "b1": (router_b1, router_b1_nocg, router_b1_forced)}

# separate-kernel markers for judge_fused / vendor_fused_ok
B2_MARKERS = ("add", "elementwise")
B1_MARKERS = ("add", "elementwise", "rms", "norm", "mean", "pow", "rsqrt", "sqrt", "red_fused")

_GAMMA = {}


def get_gamma(k):
    """gamma drawn ONCE per hidden size (T6.0.1: randn(HIDDEN)*0.1 + 1), bf16, identical
    between baseline and every fused path (and across configs of the same k)."""
    if k not in _GAMMA:
        _GAMMA[k] = (torch.randn(k, device=DEV) * 0.1 + 1.0).to(torch.bfloat16)
    return _GAMMA[k]


def _ref32_chunked(attn, res, Wr, gamma, variant, chunk):
    """fp32 ground truth in row-chunks (never materializes the [M,6144] fp32 h -- 6 GiB at
    M=131072). ref32 itself is only [M,256] fp32."""
    M = attn.shape[0]
    W32 = Wr.float()
    g32 = gamma.float() if gamma is not None else None
    out = torch.empty((M, Wr.shape[1]), device=attn.device, dtype=torch.float32)
    for i in range(0, M, chunk):
        h32 = attn[i:i + chunk].float() + res[i:i + chunk].float()
        if variant == "b1":
            inv = torch.rsqrt(h32.pow(2).mean(dim=1, keepdim=True) + EPS)
            h32 = h32 * inv * g32
        out[i:i + chunk] = h32 @ W32
        del h32
    return out


def _finish_compiled(r, markers, ref32, rel_tol):
    """Numerics (T6.0.4 fp32 criterion) + fusion judgement for one compiled-path result."""
    o = r.pop("_cfn_out", None)
    if o is not None:
        rel = _rel_max(o, ref32)
        r["rel_max_vs_fp32"] = rel
        r["numerics_ok"] = bool(rel <= rel_tol)
        del o
    if r["kernels"]:
        r["fused"], r["evidence"] = judge_fused(r["kernels"], markers)
        vok, vsep = vendor_fused_ok(r["kernels"], markers)
        r["evidence"]["vendor_fused_ok"] = vok
        r["evidence"]["vendor_separate_kernels"] = vsep
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# One config                                                                    #
# --------------------------------------------------------------------------- #
def measure_router(name, variant, M, regime, iters, warmup, gpu, gpu_adj, smoke=False):
    n, k = EXPERTS, HIDDEN
    print(f"[router {variant}] {name}: M={M} n={n} k={k} regime={regime} "
          f"(autotune may take minutes) ...", flush=True)
    attn = bf(M, k)
    res = bf(M, k)
    Wr = bf(k, n)
    gamma = get_gamma(k) if variant == "b1" else None
    # spec B.4: Wp = Wrouter * gamma[:,None]  (diag(gamma) @ W), precomputed UNTIMED (host-
    # side setup, outside every timed region)
    Wp = (Wr.float() * gamma.float()[:, None]).to(torch.bfloat16) if variant == "b1" else None

    if variant == "b2":
        def eager_fn():                     # 2 kernels: residual add, vendor GEMM
            return (attn + res) @ Wr
    else:
        def eager_fn():                     # 3 kernels: residual add, rmsnorm, vendor GEMM
            h = attn + res
            return F.rms_norm(h, (k,), gamma, EPS) @ Wr

    markers = B2_MARKERS if variant == "b2" else B1_MARKERS
    fn_def, fn_nocg, fn_forced = COMPILE_FNS[variant]
    args_tuple = (attn, res, Wr) if variant == "b2" else (attn, res, Wr, gamma)

    triton_ms = None
    triton_info = {}
    t_ok = False
    with ClockSampler() as cs:
        # bare router GEMM, contemporaneous with everything below (fusion-tax denominator);
        # attn serves as the bare X (any [M,6144] bf16 operand; avoids a 3rd M x k buffer)
        gemm_only_s = med_time(lambda: attn @ Wr, iters, warmup)
        unfused_s = med_time(eager_fn, iters, warmup)
        eager_kernels = profile_kernels(eager_fn)

        # fp32 ground truth (chunked) + eager tolerance (T6.0.4: path OK iff
        # rel_max_vs_fp32 <= max(2*eager_rel_max, 5e-2))
        ref32 = _ref32_chunked(attn, res, Wr, gamma, variant, chunk=(128 if smoke else 8192))
        eager_out = eager_fn()
        eager_rel_max = _rel_max(eager_out, ref32)
        rel_tol = max(2.0 * eager_rel_max, 5e-2)
        del eager_out

        # --- addmm: DROP up front (spec B.3/B.7), reason recorded, nothing measured ---

        # --- compiled paths (def / nocg / forced), distinct code objects per variant ---
        r_def = compile_and_time(fn_def, args_tuple, iters, warmup, "max-autotune")
        _finish_compiled(r_def, markers, ref32, rel_tol)
        r_nocg = compile_and_time(fn_nocg, args_tuple, iters, warmup,
                                  "max-autotune-no-cudagraphs")
        _finish_compiled(r_nocg, markers, ref32, rel_tol)
        r_forced = compile_and_time(fn_forced, args_tuple, iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        _finish_compiled(r_forced, markers, ref32, rel_tol)

        # --- hand Triton kernel (B.4): mini-autotune, then time + profile + numerics ---
        if _HAVE_TRITON:
            try:
                cands = ROUTER_CANDS if iters >= 10 else ROUTER_CANDS[:2]
                if variant == "b2":
                    call = lambda BM, BK, w, s: router_triton_b2(attn, res, Wr, BM, BK, w, s)
                else:
                    call = lambda BM, BK, w, s: router_triton_b1(attn, res, Wp, BM, BK, w, s)
                tcfg = tune_router(call, cands)
                if tcfg is None:
                    triton_info = {"error": "all tile configs failed"}
                else:
                    BM, BK, w, s = tcfg
                    tout = call(BM, BK, w, s)
                    t_rel = _rel_max(tout, ref32)
                    t_ok = bool(t_rel <= rel_tol)
                    triton_s = med_time(lambda: call(BM, BK, w, s), iters, warmup)
                    triton_ms = triton_s * 1e3
                    t_kernels = profile_kernels(lambda: call(BM, BK, w, s))
                    triton_info = {
                        "numerics_ok": t_ok,
                        "rel_max_vs_fp32": t_rel,
                        "eager_rel_max_vs_fp32": eager_rel_max,
                        "rel_tol": rel_tol,
                        "max_abs_diff_vs_fp32": _maxabs(tout, ref32),
                        "config": f"BM{BM} BN{n} BK{BK} w{w} s{s} "
                                  f"(tuned over {len(cands)} candidates)",
                        "grid": f"(cdiv({M},{BM}), 1) = ({-(-M // BM)}, 1)",
                        "no_split_k": True,
                        "fused_by_construction": True,
                        "kernel_evidence": t_kernels,
                        "n_kernels": len(t_kernels),
                    }
                    del tout
            except Exception as exc:
                triton_info = {"error": f"{type(exc).__name__}: {exc}"}
                traceback.print_exc()
        else:
            triton_info = {"skipped": "triton unavailable"}

        # drift probe (spec B.2): bare X @ Wrouter re-measured at config END
        gemm_repeat_s = med_time(lambda: attn @ Wr, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    compiled_ms = r_def["ms"]
    compiled_nocg_ms = r_nocg["ms"]
    forced_ms = r_forced["ms"]

    drift_ratio = gemm_repeat_s * 1e3 / gemm_only_ms
    drift_clean = abs(drift_ratio - 1.0) <= 0.05
    # B.2: best_unfused = min(eager, nocg); nocg is a best-UNFUSED realization here, NOT a
    # fused candidate (spec B.3)
    unf_cand = [v for v in (unfused_ms, compiled_nocg_ms) if v is not None]
    best_unfused_ms = min(unf_cand)
    # B.7 mutually-inconsistent-baseline guard: the unfused chain CONTAINS the bare GEMM, so
    # best_unfused < gemm_only (beyond noise) marks a latency/occupancy-bound or drifting row
    baselines_consistent = best_unfused_ms >= 0.97 * gemm_only_ms
    exclusion_reasons = []
    if not drift_clean:
        exclusion_reasons.append(
            f"gemm_drift_ratio {drift_ratio:.4f} outside [0.95,1.05] (T6.0.4 drift rule)")
    if not baselines_consistent:
        exclusion_reasons.append(
            "best_unfused_ms < 0.97*gemm_only_ms: mutually-inconsistent baselines "
            "(latency/occupancy-bound small-M suspected, spec B.7)")
    excluded = bool(exclusion_reasons)

    # measured_gain (C500-convention, T6.0.4): includes UNVERIFIED fused paths
    fcand = [v for v in (compiled_ms, forced_ms, triton_ms) if v is not None]
    best_fused_ms = min(fcand) if fcand else None
    measured_gain = (best_unfused_ms / best_fused_ms) if best_fused_ms else None
    # verified-only: hand triton (fused by construction) iff numerics OK; forced/compiled
    # iff the kernel evidence confirms the fold AND numerics OK
    vcand = []
    if triton_ms is not None and t_ok:
        vcand.append(triton_ms)
    if forced_ms is not None and r_forced["fused"] and r_forced.get("numerics_ok"):
        vcand.append(forced_ms)
    if compiled_ms is not None and r_def["fused"] and r_def.get("numerics_ok"):
        vcand.append(compiled_ms)
    measured_gain_verified = (best_unfused_ms / min(vcand)) if vcand else None

    # forced-path documentation (spec B.3/B.7: record evidence either way)
    if forced_ms is None:
        forced_note = f"forced path errored: {r_forced['error']}"
    elif r_forced["fused"]:
        forced_note = ("forced Triton template FOLDED the input " +
                       ("add (b2 prologue fusion fired)" if variant == "b2" else
                        "chain incl. the RMSNorm reduction (unexpected for b1 - re-check evidence)"))
    else:
        sep = r_forced.get("evidence", {}).get("separate_elementwise_kernels", [])
        if variant == "b2":
            forced_note = ("stock/forced template did NOT fold the input add on this torch "
                           f"build; surviving separate kernels: {sep}")
        else:
            forced_note = ("expected for b1: at best folds the residual; the RMSNorm "
                           f"K-reduction survives as separate kernel(s): {sep}")

    est_fn = est_router_residual_ms if variant == "b2" else est_router_residual_rms_ms
    est_unf, est_fus = est_fn(M, gpu)

    row = {
        "name": name, "kind": "router_prologue", "variant": variant, "regime": regime,
        "dims": {"M": M, "n": n, "k": k},
        "gemm_only_ms": gemm_only_ms,
        "unfused_ms": unfused_ms,
        "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": drift_ratio,
        "drift_clean": drift_clean,
        "baselines_consistent": baselines_consistent,
        "excluded_from_aggregate": excluded,
        "exclusion_reasons": exclusion_reasons,
        "eager_rel_max_vs_fp32": eager_rel_max,
        "rel_tol": rel_tol,
        "eager_kernel_evidence": eager_kernels,
        "unfused_compiled_nocg_ms": compiled_nocg_ms,
        "best_unfused_ms": best_unfused_ms,
        "addmm_ms": None, "addmm_dropped": True, "addmm_drop_reason": ADDMM_DROP_REASON,
        "fused_paths": {"compiled_ms": compiled_ms,
                        "compiled_forced_triton_ms": forced_ms,
                        "triton_ms": triton_ms},
        "best_fused_ms": best_fused_ms,
        "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "est_model_note": "faithful F3-analog (residual tile-local; RMSNorm reduction K-local "
                          "to one CTA's loop) - NOT structure-blind, no INVALID flag needed "
                          "(contrast T6.A epilogue / T6.C)",
        "fused_verified": r_def["fused"],
        "fused_verified_forced": r_forced["fused"],
        "forced_prologue_folded": r_forced["fused"],
        "forced_note": forced_note,
        "kernel_evidence": r_def["kernels"],
        "fusion_evidence": r_def["evidence"],
        "nocg_kernel_evidence": r_nocg["kernels"],
        "nocg_fusion_evidence": r_nocg["evidence"],
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced["evidence"],
        "numerics_ok": r_def.get("numerics_ok"),
        "rel_max_vs_fp32": r_def.get("rel_max_vs_fp32"),
        "nocg_numerics_ok": r_nocg.get("numerics_ok"),
        "nocg_rel_max_vs_fp32": r_nocg.get("rel_max_vs_fp32"),
        "forced_numerics_ok": r_forced.get("numerics_ok"),
        "forced_rel_max_vs_fp32": r_forced.get("rel_max_vs_fp32"),
        "triton_numerics_ok": (t_ok if triton_ms is not None else None),
        "compiled_over_gemm_ratio": (compiled_ms / gemm_only_ms) if compiled_ms else None,
        "nocg_over_gemm_ratio": (compiled_nocg_ms / gemm_only_ms) if compiled_nocg_ms else None,
        "forced_over_gemm_ratio": (forced_ms / gemm_only_ms) if forced_ms else None,
        "triton_over_gemm_ratio": (triton_ms / gemm_only_ms) if triton_ms else None,
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "triton_info": triton_info,
        "compile_error": r_def["error"],
        "nocg_error": r_nocg["error"],
        "forced_error": r_forced["error"],
        "clocks": clocks,
    }
    if M <= 1024:
        row["small_m_latency_note"] = (
            "M<=1024: few row-tiles on 24 SM (grid=cdiv(M,BM)) - latency/occupancy-bound "
            "regime; BM=64 preferred to raise tile count; see baselines_consistent + drift "
            "(spec B.7)")
    if gpu_adj is not None:
        a_unf, a_fus = est_fn(M, gpu_adj)
        row["est_unfused_ms_adj"] = a_unf
        row["est_fused_ms_adj"] = a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    print(f"    gemm_only={gemm_only_ms:.4f}  unfused={unfused_ms:.4f}  "
          f"nocg={compiled_nocg_ms}  compiled={compiled_ms}  forced={forced_ms}  "
          f"triton={triton_ms}  meas_gain={measured_gain}  "
          f"gain_verified={measured_gain_verified}  est_gain={est_unf/est_fus:.4f}  "
          f"forced_folded={r_forced['fused']}  drift={drift_ratio:.4f}  "
          f"excluded={excluded}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# Sweep + aggregate                                                             #
# --------------------------------------------------------------------------- #
def router_configs(smoke):
    """(name, variant, M, regime). Router is DENSE -> M = tokens (NOT tokens/32).
    DECODE M in {512..16384}, PREFILL M in {8192,32768,131072}; M=8192 is the T6.0.3
    decode/prefill boundary - present in both regime lists, measured ONCE."""
    if smoke:
        return [("router_b2_smoke_M256", "b2", 256, "smoke"),
                ("router_b1_smoke_M256", "b1", 256, "smoke")]
    cfgs = []
    for variant in ("b2", "b1"):
        for M in (512, 1024, 2048, 4096, 8192, 16384, 32768, 131072):
            regime = ("decode/prefill-boundary" if M == 8192
                      else ("decode" if M <= 16384 else "prefill"))
            cfgs.append((f"router_{variant}_M{M}", variant, M, regime))
    return cfgs


def build_aggregate(rows):
    """Per-variant geomeans over drift-clean, baseline-consistent rows only (T6.0.4/B.7)."""
    agg = {}
    for variant in ("b2", "b1"):
        vrows = [r for r in rows if r.get("variant") == variant and "error" not in r]
        inc = [r for r in vrows if not r.get("excluded_from_aggregate")]

        def gm(key):
            vals = [r[key] for r in inc if r.get(key)]
            return geomean(vals) if vals else None

        agg[variant] = {
            "n_rows": len(vrows),
            "n_included": len(inc),
            "excluded_rows": [r["name"] for r in vrows if r.get("excluded_from_aggregate")],
            "measured_gain_geomean": gm("measured_gain"),
            "measured_gain_verified_geomean": gm("measured_gain_verified"),
            "estimated_gain_geomean": gm("estimated_gain"),
            "estimated_gain_adj_geomean": gm("estimated_gain_adj"),
            "forced_prologue_folded_rows": [r["name"] for r in vrows
                                            if r.get("forced_prologue_folded")],
        }
    return agg


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny dims (M=256, HIDDEN=512), iters=3/warmup=3, every path end-to-end")
    ap.add_argument("--t2-json", default=None,
                    help="T2 measured-peaks JSON (rtx4060_peak.json); adds adjusted-profile estimates")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    iters, warmup = (3, 3) if args.smoke else (30, 15)

    global HIDDEN
    if args.smoke:
        HIDDEN = 512          # estimator helpers read the module global -> smoke est dims
                              # match the smoke tensors (EXPERTS stays 256: BN=full-N path)

    gpu = GPUS["rtx4060-measured"]
    gpu_adj, adj_info = (None, None)
    if args.t2_json:
        gpu_adj, adj_info = build_adjusted_profile(args.t2_json, gpu)

    props = torch.cuda.get_device_properties(0)
    env = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": (triton.__version__ if _HAVE_TRITON else None),
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": props.multi_processor_count,
        "total_memory_gib": round(props.total_memory / 1024**3, 4),
        "smem_per_block_optin_bytes": getattr(props, "shared_memory_per_block_optin", None),
        "estimator_profile": gpu.name,
        "estimator_peak_tflops": gpu.peak_tensor_flops / 1e12,
        "estimator_bw_gbps": gpu.bw_bytes_per_s / 1e9,
        "clocks_locked": True,
        "clocks_note": "clocks LOCKED 1500 MHz core / 5501 MHz VRAM by the host; the 35 W "
                       "power cap can still throttle below the lock on sustained heavy "
                       "configs - ClockSampler + drift probes record it",
        "inductor_prologue_fusion_flag": getattr(_ic, "prologue_fusion", None),
        "smoke": args.smoke,
        "adjusted_profile": adj_info,
    }
    conventions = {
        "test": "T6.B router PROLOGUE fusion: b2 = residual-only (logits=(attn+res)@Wr), "
                "b1 = residual+RMSNorm (logits=RMSNorm(attn+res)@Wr, gamma folded into "
                "Wp=Wr*gamma[:,None] precomputed UNTIMED on host)",
        "caveat_router_minor_cost_center": CAVEAT_MINOR_COST_CENTER,
        "caveat_shared_h_x": CAVEAT_SHARED_H_X,
        "rmsnorm_baseline_traffic_note": RMS_TRAFFIC_NOTE,
        "addmm_drop_reason": ADDMM_DROP_REASON,
        "time_unit": "milliseconds for every field ending in _ms",
        "med_time_returns": "SECONDS (rtx4060_common.med_time); converted to ms here (*1e3)",
        "estimator_time_unit": "SECONDS (KTime.time_s); converted to ms here (*1e3)",
        "dtype": "bfloat16 for all measured tensors; fp32 accumulate",
        "gain_definition": "gain = unfused_time / fused_time (>1.0 => fusion FASTER)",
        "measured_gain": "best_unfused_ms / best_fused_ms (C500-convention, T6.0.4; best "
                         "fused over compiled/forced/triton INCLUDING unverified paths; "
                         "best_unfused = min(eager, nocg) per spec B.2)",
        "measured_gain_verified": "best_unfused_ms / best VERIFIED-fused path: hand triton "
                         "(fused by construction, grid (cdiv(M,BM),1), no split-K) iff "
                         "numerics_ok; forced/compiled iff judge_fused confirms the fold AND "
                         "numerics_ok. The verdict keys on THIS",
        "estimated_gain": "est_unfused_ms / est_fused_ms (B.5 helpers, rtx4060-measured "
                          "profile; *_adj = --t2-json adjusted profile via "
                          "build_adjusted_profile)",
        "est_model_validity": "the B.5 est model is a faithful F3-analog (residual is "
                          "tile-local; the RMSNorm reduction is K-local to one CTA's loop) - "
                          "NOT structure-blind; no est=INVALID flag applies here (contrast "
                          "T6.A epilogue and T6.C, per the T6.0.4 reporting guard)",
        "numerics": "fp32 ground truth per config (row-chunked; never materializes fp32 "
                    "[M,6144]); each path OK iff rel_max_vs_fp32 <= max(2*eager_rel_max, "
                    "5e-2). fp32-accumulated fused kernels are typically MORE accurate than "
                    "eager, so allclose-vs-eager would be the wrong test",
        "unfused_paths": "b2: eager residual + vendor GEMM (2 kernels); b1: eager residual + "
                    "eager F.rms_norm + vendor GEMM (3 kernels); nocg (max-autotune-no-"
                    "cudagraphs, unforced) is the BEST-UNFUSED realization on this 24-SM part "
                    "(vendor GEMM + fused pointwise add [+ separate rms reduction]), NOT a "
                    "fused candidate (spec B.3); best_unfused = min(eager, nocg)",
        "forced_path": "is_big_gpu patched + TRITON-only autotune backends + no-cudagraphs: "
                    "the key STOCK test for b2 - does inductor prologue-fusion fold the "
                    "input add into a single triton_tem_fused_* template? Evidence recorded "
                    "either way (forced_note). b1: even forced, inductor cannot co-accumulate "
                    "sum_k h^2 inside the matmul template - at best folds the residual, "
                    "leaves the RMSNorm as a separate reduction kernel",
        "hand_kernel": "B.4 exactly: BN=256 full-N, grid (cdiv(M,BM), 1), BM {64,128}, BK "
                    "{32,64}, warps 4-8, stages 2-3 mini-autotune; b2: acc += dot(a+r, "
                    "W-tile); b1: dual-accumulator single pass co-accumulating fp32 sumsq "
                    "over the SAME K-loop, rms applied at K-loop end, gamma pre-folded into "
                    "Wp. HARD no-split-K: single-pass sumsq is only correct if one CTA owns "
                    "the full K-loop for its rows (M/BM CTAs give the parallelism; fp32 "
                    "numerics would catch any violation)",
        "gemm_drift_ratio": "bare X@Wr GEMM re-measured at config END / at START (X=attn); "
                    "clean iff |drift-1| <= 0.05; drift-tainted rows kept in JSON but "
                    "excluded from the aggregate geomeans (T6.0.4)",
        "excluded_from_aggregate": "spec B.7: rows with |drift-1| > 0.05 OR mutually-"
                    "inconsistent baselines (best_unfused < 0.97*gemm_only - the unfused "
                    "chain contains the bare GEMM, so this marks a latency/occupancy-bound "
                    "small-M or drifting row) are flagged and excluded from geomeans",
        "regime_boundary": "M=8192 sits in both the decode and prefill token lists (T6.0.3 "
                    "boundary anchor); the router is dense (M=tokens) so it is measured ONCE "
                    "and labeled 'decode/prefill-boundary'",
        "oom_fallback": "spec B.7: prefill M=131072 b1 unfused holds attn,res,h,x (~6 GiB + "
                    "workspaces); if a config OOMs at M=131072 it is re-run at M=65536 and "
                    "documented in its oom_fallback field (the fused path uses fewer buffers)",
        "compiled_cudagraph_input_copy_us": "GPU time of multi_tensor_apply static-input-copy "
                    "kernels inside ONE compiled (cudagraphs) call - deployment overhead, "
                    "not fusion work",
        "timing": f"median of {iters} cuda-event samples, {warmup} warmup, per config "
                  "(rtx4060_common.med_time with busy-GEMM clock warmer)",
    }

    print(f"=== T6.B router prologue fusion  (smoke={args.smoke}) ===", flush=True)
    print(f"device={env['device_name']} torch={env['torch_version']} "
          f"triton={env['triton_version']} prologue_fusion={env['inductor_prologue_fusion_flag']}",
          flush=True)

    results = []
    out = {"conventions": conventions, "env": env, "configs": results, "aggregate": None}

    def _cleanup():
        # cudagraph static pools + compiled artifacts accumulate per (shape, variant) on the
        # code-object caches -- at M=131072 that is GiB-scale; drop them between configs
        _dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()

    def run_config(name, variant, M, regime):
        try:
            return measure_router(name, variant, M, regime, iters, warmup, gpu, gpu_adj,
                                  smoke=args.smoke)
        except torch.cuda.OutOfMemoryError as exc:
            traceback.print_exc()
            _cleanup()
            if M >= 131072:
                fb = 65536
                print(f"[oom] {name}: OOM at M={M}; dropping top prefill point to M={fb} "
                      f"(spec B.7)", flush=True)
                try:
                    row = measure_router(f"{name}_oomfb_M{fb}", variant, fb, regime, iters,
                                         warmup, gpu, gpu_adj, smoke=args.smoke)
                    row["oom_fallback"] = {
                        "requested_M": M, "fallback_M": fb,
                        "reason": f"OOM at M={M} ({type(exc).__name__}); spec B.7: b1 "
                                  "unfused holds attn,res,h,x (~6 GiB + workspaces) at "
                                  "131072 - top prefill point dropped to 65536, documented",
                    }
                    return row
                except Exception as exc2:
                    traceback.print_exc()
                    return {"name": f"{name}_oomfb_M{fb}", "kind": "router_prologue",
                            "variant": variant, "regime": regime,
                            "error": f"{type(exc2).__name__}: {exc2}",
                            "oom_fallback": {"requested_M": M, "fallback_M": fb}}
            return {"name": name, "kind": "router_prologue", "variant": variant,
                    "regime": regime, "error": f"OutOfMemoryError: {exc}"}
        except Exception as exc:
            traceback.print_exc()
            return {"name": name, "kind": "router_prologue", "variant": variant,
                    "regime": regime, "error": f"{type(exc).__name__}: {exc}"}

    for name, variant, M, regime in router_configs(args.smoke):
        row = run_config(name, variant, M, regime)
        results.append(row)
        out["aggregate"] = build_aggregate(results)
        save_json(args.out, out)          # incremental save after EVERY config
        _cleanup()

    # -------- human-readable summary --------
    ok_rows = [r for r in results if "error" not in r]
    err_rows = [r for r in results if "error" in r]
    print("\n" + "=" * 132)
    print(f"{'config':<26}{'var':<4}{'M':>7}{'unfused':>9}{'best_unf':>9}{'best_fus':>9}"
          f"{'gain':>8}{'gain_ver':>9}{'est_gain':>9}{'forced?':>8}{'drift':>7}{'excl':>5}")
    print("-" * 132)
    for r in ok_rows:
        bfu = r["best_fused_ms"]
        mg = r["measured_gain"]
        mgv = r.get("measured_gain_verified")
        print(f"{r['name']:<26}{r['variant']:<4}{r['dims']['M']:>7}{r['unfused_ms']:>9.4f}"
              f"{r['best_unfused_ms']:>9.4f}"
              f"{(bfu if bfu is not None else float('nan')):>9.4f}"
              f"{(mg if mg is not None else float('nan')):>8.4f}"
              f"{(mgv if mgv is not None else float('nan')):>9.4f}"
              f"{r['estimated_gain']:>9.4f}"
              f"{str(r.get('forced_prologue_folded')):>8}"
              f"{r['gemm_drift_ratio']:>7.3f}"
              f"{('Y' if r.get('excluded_from_aggregate') else '-'):>5}")
    for r in err_rows:
        print(f"{r['name']:<26}{r.get('variant', '?'):<4}  ERROR: {r['error']}")
    print("-" * 132)
    agg = out["aggregate"]
    for variant in ("b2", "b1"):
        a = agg[variant]
        print(f"[{variant}] included {a['n_included']}/{a['n_rows']} rows | "
              f"gain geomean={a['measured_gain_geomean']} | "
              f"verified={a['measured_gain_verified_geomean']} | "
              f"est={a['estimated_gain_geomean']} | est_adj={a['estimated_gain_adj_geomean']} | "
              f"forced-folded: {a['forced_prologue_folded_rows']}")
    if err_rows:
        print(f"WARNING: {len(err_rows)} config(s) errored (rows kept in JSON with 'error')")
    print("=" * 132)


if __name__ == "__main__":
    main()
