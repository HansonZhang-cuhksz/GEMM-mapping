"""T4 (RTX4060_SIM_REAL_TASK.md) - fusion realizability: the KEY experiment.

Measures, on working NVIDIA/CUDA paths, whether the estimator's predicted ~1-2.6% fusion
gain for the GLM-5.2 decode fusions materialises on real silicon -- disambiguating the C500
null result (A: real-but-un-capturable-on-C500 tooling problem, vs B: estimator over-predicts).

Two fusion primitives, each on scaled dense configs that fit the 8 GiB 4060:
  PRIMITIVE 1  SwiGLU folded into the up_gate epilogue (F4-analog):
      swiglu_ffn(x, Wug): gu = x@Wug ; g,u = gu[..,:inter],gu[..,inter:] ; silu(g)*u
      paths: eager unfused | torch.compile(max-autotune) | (optional) hand Triton GEMM+SwiGLU
  PRIMITIVE 2  residual folded into the mla_o epilogue (F1-analog):
      (x@Wo)+res  vs  torch.addmm(res,x,Wo) (cuBLAS beta-accumulate)  vs  compile(max-autotune)

For every config the estimator's predicted gain (fusion_time_estimator, rtx4060-measured
profile, same dims) is computed CPU-side and stored next to the measured gain.

UNITS: rtx4060_common.med_time and the estimator KTime.time_s are both SECONDS; every field
ending _ms in the output JSON is milliseconds. gain = unfused/fused (>1 => fusion FASTER),
matching the estimator's "1.020x" convention.

Run (from GEMM-mapping/):
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json
    python rtx4060_fusion_measure.py --smoke --out /tmp/.../fusion_smoke.json
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json --moe          # + grouped-bmm MoE
    python rtx4060_fusion_measure.py --out rtx4060_fusion.json --t2-json rtx4060_measured.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F
import torch._dynamo as _dynamo
import torch._inductor.config as _ic
import torch._inductor.utils as _iu

# max-autotune lets inductor *try* to emit a Triton GEMM template with a fused epilogue
# (on <=24 SM parts it may decline -- we record whichever it chose).
_ic.max_autotune = True
_ic.max_autotune_gemm = True


class force_triton_templates:
    """Make inductor emit a Triton GEMM template on this 24-SM part.

    is_big_gpu() hard-requires >=68 SMs (torch/_inductor/utils.py), so max-autotune on the
    4060 keeps the vendor GEMM and only fuses the pointwise epilogue into a SEPARATE kernel.
    Restricting the autotune backends to TRITON (instead of letting cuBLAS win) is what
    yields the actual object under test: a Triton GEMM template with the epilogue folded in.
    The backend change also changes the inductor cache key, so forced and unforced variants
    can never reuse each other's compiled artifacts. Must wrap COMPILE + FIRST CALLS
    (lowering happens on first call, not at torch.compile())."""

    def __enter__(self):
        self._orig_big = _iu.is_big_gpu
        self._orig_backends = _ic.max_autotune_gemm_backends
        _iu.is_big_gpu = lambda *a, **k: True
        _ic.max_autotune_gemm_backends = "TRITON"
        return self

    def __exit__(self, *exc):
        _iu.is_big_gpu = self._orig_big
        _ic.max_autotune_gemm_backends = self._orig_backends

# The full sweep compiles up to 8 distinct shapes of the SAME swiglu_ffn code object
# (6 dense + 2 MoE) -- exactly torch's default recompile limit of 8, past which dynamo
# SILENTLY falls back to eager (compiled_ms would then time the eager path). Raise it.
for _attr in ("cache_size_limit", "recompile_limit"):
    if getattr(_dynamo.config, _attr, 64) < 64:
        setattr(_dynamo.config, _attr, 64)

sys.path.insert(0, "/home/shuhan/snowcat-demo/GEMM-mapping")

from rtx4060_common import (  # noqa: E402  (mandatory shared timing module; med_time returns SECONDS)
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
    _residual_aux,
    estimate_fused_gemm,
    estimate_gemm_grouped,
    estimate_vector_kernel,
)

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


# --------------------------------------------------------------------------- #
# Hand Triton GEMM + SwiGLU epilogue (the "upper bound" / fusion-tax datapoint)  #
# --------------------------------------------------------------------------- #
if _HAVE_TRITON:

    @triton.jit
    def _swiglu_gemm_kernel(
        X, W, O,
        M, INTER, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        # One output tile O[BM, BN] over the INTER (half-width) dimension needs the GEMM's
        # gate columns [n : n+BN] AND up columns [n+INTER : n+INTER+BN]; accumulate both.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        x_ptrs = X + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        wg_ptrs = W + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        wu_ptrs = W + (offs_k[:, None] * stride_wk + (offs_n[None, :] + INTER) * stride_wn)
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < INTER
        for k0 in range(0, K, BK):
            kmask = offs_k < (K - k0)
            a = tl.load(x_ptrs, mask=m_mask & kmask[None, :], other=0.0)
            wg = tl.load(wg_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
            wu = tl.load(wu_ptrs, mask=kmask[:, None] & n_mask, other=0.0)
            acc_g += tl.dot(a, wg)
            acc_u += tl.dot(a, wu)
            x_ptrs += BK * stride_xk
            wg_ptrs += BK * stride_wk
            wu_ptrs += BK * stride_wk
        out = (acc_g * tl.sigmoid(acc_g)) * acc_u   # silu(gate) * up
        o_ptrs = O + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=m_mask & n_mask)

    def triton_swiglu(x, Wug, inter, BM=64, BN=64, BK=32, num_warps=4, num_stages=2):
        M, K = x.shape
        O = torch.empty((M, inter), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BM), triton.cdiv(inter, BN))
        _swiglu_gemm_kernel[grid](
            x, Wug, O, M, inter, K,
            x.stride(0), x.stride(1),
            Wug.stride(0), Wug.stride(1),
            O.stride(0), O.stride(1),
            BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages,
        )
        return O

    # (BM, BN, BK, num_warps, num_stages) candidates for a quick hand-rolled autotune.
    TRITON_CANDS = [
        (64, 64, 32, 4, 2), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
        (128, 128, 32, 8, 3), (128, 128, 64, 8, 4), (64, 64, 64, 4, 3),
    ]

    def tune_triton_swiglu(x, Wug, inter, cands):
        """Pick the fastest (BM,BN,BK,w,s) with a short timing pass; returns (cfg, none-if-all-fail)."""
        best = None
        for cfg in cands:
            BM, BN, BK, w, s = cfg
            try:
                t = med_time(lambda: triton_swiglu(x, Wug, inter, BM, BN, BK, w, s),
                             iters=5, warmup=3)
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        return best[1] if best else None


# --------------------------------------------------------------------------- #
# Fusion verification helpers (profiler + numerics)                             #
# --------------------------------------------------------------------------- #
def profile_kernels(call):
    """Return the CUDA kernels (name, us) launched by one call() invocation.

    activities=[ProfilerActivity.CUDA] per task spec; CPU-side ops carry dt=0 and are
    filtered out via device_type==CUDA.
    """
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    kernels = []
    for ev in prof.key_averages():
        if ev.device_type != DeviceType.CUDA:
            continue
        dt = getattr(ev, "device_time_total", None)
        if dt is None:
            dt = getattr(ev, "cuda_time_total", 0.0)
        dt = float(dt or 0.0)
        if dt <= 0:
            continue
        cnt = int(getattr(ev, "count", 1) or 1)
        kernels.append({
            "name": ev.key,
            "count": cnt,
            "total_us": round(dt, 3),
            "avg_us": round(dt / max(cnt, 1), 3),
        })
    kernels.sort(key=lambda d: -d["total_us"])
    return kernels


def _is_gemm_template(name_lower):
    # An inductor Triton GEMM *template* fuses the epilogue into the GEMM itself.
    return "triton" in name_lower and any(t in name_lower for t in ("mm", "gemm", "tem"))


def judge_fused(kernels, act_markers):
    """fused_verified iff a Triton GEMM template exists AND no separate activation kernel
    (silu/mul/add elementwise) consuming the intermediate remains. A tiny copy/reduction is
    tolerable (it just won't be the dominant kernel). Raw list is stored for human re-judge.
    """
    lows = [k["name"].lower() for k in kernels]
    total = sum(k["total_us"] for k in kernels) or 1.0
    has_template = any(_is_gemm_template(l) for l in lows)
    dominant = max(kernels, key=lambda k: k["total_us"]) if kernels else None
    dom_is_template = dominant is not None and _is_gemm_template(dominant["name"].lower())
    # separate elementwise activation/epilogue kernels that are NOT a GEMM template
    separate = [
        k["name"] for k, l in zip(kernels, lows)
        if not _is_gemm_template(l) and any(m in l for m in act_markers)
    ]
    fused = has_template and dom_is_template and len(separate) == 0
    evidence = {
        "n_kernels": len(kernels),
        "dominant_kernel": dominant["name"] if dominant else None,
        "dominant_frac_of_gpu_time": round(dominant["total_us"] / total, 4) if dominant else None,
        "triton_gemm_template_present": has_template,
        "separate_elementwise_kernels": separate,
    }
    return fused, evidence


def _maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _rel_max(t, ref32):
    """Max elementwise relative deviation from an fp32 reference (1.0 absolute floor)."""
    return ((t.float() - ref32).abs() / (ref32.abs() + 1.0)).max().item()


def vendor_fused_ok(kernels, markers):
    """Vendor-path fusion judge (no Triton template required): fused iff NO separate
    non-GEMM elementwise kernel matching `markers` remains. Memcpy DtoD is tolerated
    (for addmm it is the algorithmic beta-accumulate copy of `res`, not epilogue work)."""
    lows = [k["name"].lower() for k in kernels]
    separate = [
        k["name"] for k, l in zip(kernels, lows)
        if any(m in l for m in markers)
        and not _is_gemm_template(l) and "gemm" not in l and "cutlass" not in l
    ]
    return len(separate) == 0, separate


def _cudagraph_copy_us(kernels):
    """GPU time of the cudagraph static-input copy kernels (multi_tensor_apply) inside one
    compiled call. This is deployment overhead of the cudagraph path, NOT fusion work --
    at full dims it is ~2-3% of compiled_ms, the same order as the 1-2.6% gain under test,
    so it is stored separately for the writeup. (Memcpy DtoD is deliberately NOT counted:
    in the residual case it is the algorithmic beta-accumulate copy of `res`.)"""
    return sum(k["total_us"] for k in kernels if "multi_tensor_apply" in k["name"].lower())


# --------------------------------------------------------------------------- #
# Estimator side (CPU only -- no GPU)                                            #
# --------------------------------------------------------------------------- #
def est_swiglu_ms(M, hidden, inter, count, gpu):
    """F4-analog predicted (unfused_ms, fused_ms) for dense/grouped SwiGLU-into-up_gate.

    up_gate GEMM is (m=M, n=2*inter, k=hidden). count=1 dense; count=E grouped MoE.
    unfused = grouped GEMM + separate activation vector kernel (read gate+up 2*inter, write
    activated inter -> 3*rows*inter*BPE); fused = same GEMM with out_factor=0.5 epilogue.
    """
    rows = count * M
    unf_g = estimate_gemm_grouped("up_gate", M, 2 * inter, hidden, count, gpu)
    unf_a = estimate_vector_kernel("activation", 3 * rows * inter * BPE, gpu)
    fus = estimate_fused_gemm("up_gate+swiglu", M, 2 * inter, hidden, count,
                              Epilogue(out_factor=0.5), gpu)
    return (unf_g.time_s + unf_a.time_s) * 1e3, fus.time_s * 1e3


def est_residual_ms(M, n, k, gpu):
    """F1-analog predicted (unfused_ms, fused_ms) for residual-into-mla_o (dense, count=1).

    GEMM is (m=M, n, k). unfused = GEMM + residual vector kernel (read y, read x, write sum =
    3*M*n*BPE); fused = GEMM whose epilogue adds one residual read (extra_hbm_once=M*n*BPE) and
    holds the residual output tile on chip (aux SMEM = _residual_aux).
    """
    unf_g = estimate_gemm_grouped("gemm", M, n, k, 1, gpu)
    unf_r = estimate_vector_kernel("residual", 3 * M * n * BPE, gpu)
    fus = estimate_fused_gemm("gemm+residual", M, n, k, 1,
                              Epilogue(extra_hbm_once=M * n * BPE,
                                       aux_smem_per_tile=_residual_aux), gpu)
    return (unf_g.time_s + unf_r.time_s) * 1e3, fus.time_s * 1e3


# --------------------------------------------------------------------------- #
# GPU measurement side                                                          #
# --------------------------------------------------------------------------- #
def make_swiglu(inter, batched=False):
    def swiglu_ffn(x, W):
        gu = torch.bmm(x, W) if batched else x @ W
        g = gu[..., :inter]
        u = gu[..., inter:]
        return F.silu(g) * u
    return swiglu_ffn


# Dynamo's compiled-code cache lives ON the function's code object, and inductor config
# changes (mode / backends / is_big_gpu patch) are NOT part of its cache key -- compiling
# the same code object under a different variant could silently reuse the previous
# variant's artifact. Each compile variant therefore gets a textually distinct def.
def make_swiglu_nocg(inter, batched=False):
    def swiglu_ffn_nocg(x, W):
        gu = torch.bmm(x, W) if batched else x @ W
        g = gu[..., :inter]
        u = gu[..., inter:]
        return F.silu(g) * u
    return swiglu_ffn_nocg


def make_swiglu_forced(inter, batched=False):
    def swiglu_ffn_forced(x, W):
        gu = torch.bmm(x, W) if batched else x @ W
        g = gu[..., :inter]
        u = gu[..., inter:]
        return F.silu(g) * u
    return swiglu_ffn_forced


def residual_fn(x, Wo, res):
    return x @ Wo + res


def residual_fn_nocg(x, Wo, res):
    return x @ Wo + res


def residual_fn_forced(x, Wo, res):
    return x @ Wo + res


def compile_and_time(fn, args_tuple, iters, warmup, mode, forced=False):
    """Compile fn under `mode` (optionally with forced Triton GEMM templates), finish
    autotune with a few warm calls INSIDE the forced context (lowering happens on first
    call), then med_time + profile + numerics evidence. Returns a dict; never raises."""
    out = {"ms": None, "kernels": [], "fused": False, "evidence": {},
           "numerics_ok": None, "max_abs": None, "error": None}
    try:
        ctx = force_triton_templates() if forced else None
        if ctx:
            ctx.__enter__()
        try:
            cfn = torch.compile(fn, mode=mode, dynamic=False)
            for _ in range(4):
                cfn(*args_tuple)
            torch.cuda.synchronize()
            out["ms"] = med_time(lambda: cfn(*args_tuple), iters, warmup) * 1e3
            out["kernels"] = profile_kernels(lambda: cfn(*args_tuple))
            out["_cfn_out"] = cfn(*args_tuple)
        finally:
            if ctx:
                ctx.__exit__()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return out


def measure_swiglu(name, M, hidden, inter, iters, warmup, gpu, gpu_adj,
                   count=1, batched=False, do_triton=True):
    """Measure + estimate one SwiGLU-into-up_gate config (dense count=1, or grouped MoE)."""
    print(f"[swiglu] {name}: M={M} hidden={hidden} inter={inter} "
          f"count={count} batched={batched} (autotune may take minutes) ...", flush=True)
    if batched:
        x = bf(count, M, hidden)
        Wug = bf(count, hidden, 2 * inter)
        dims = {"experts": count, "tokens_per_expert": M, "hidden": hidden, "inter": inter}
    else:
        x = bf(M, hidden)
        Wug = bf(hidden, 2 * inter)
        dims = {"M": M, "hidden": hidden, "inter": inter}
    swiglu_ffn = make_swiglu(inter, batched=batched)

    triton_ms = None
    triton_info = {}
    act_markers = ("silu", "sigmoid", "elementwise")
    with ClockSampler() as cs:
        gemm_only_s = med_time(lambda: (torch.bmm(x, Wug) if batched else x @ Wug), iters, warmup)
        unfused_s = med_time(lambda: swiglu_ffn(x, Wug), iters, warmup)
        ref = swiglu_ffn(x, Wug)
        # fp32 ground truth. The bf16 EAGER reference itself deviates from it (gu is rounded
        # to bf16 before silu); a path is numerically OK if it deviates no more than eager
        # does (2x margin, 5e-2 floor). The hand kernel's fp32 accumulators are MORE accurate
        # than eager, which a naive allclose-vs-eager wrongly fails at large k.
        gu32 = torch.bmm(x.float(), Wug.float()) if batched else x.float() @ Wug.float()
        ref32 = F.silu(gu32[..., :inter]) * gu32[..., inter:]
        del gu32
        eager_rel_max = _rel_max(ref, ref32)
        rel_tol = max(2.0 * eager_rel_max, 5e-2)

        # --- compiled paths ---
        # def   : plain max-autotune (what a user gets; cudagraphs ON -> the static-input
        #         copy kernel runs inside every timed call, tracked separately)
        # nocg  : max-autotune-no-cudagraphs, unforced -- on this 24-SM part inductor keeps
        #         the vendor GEMM + ONE separate fused pointwise epilogue kernel = the best
        #         UNFUSED realization (matches the estimator's unfused model: GEMM + a
        #         single 3x-inter-traffic vector kernel)
        # forced: Triton GEMM template forced (is_big_gpu patch + TRITON-only backends),
        #         no-cudagraphs -- the true fused-epilogue GEMM the task asks about
        r_def = compile_and_time(make_swiglu(inter, batched), (x, Wug), iters, warmup,
                                 "max-autotune")
        r_nocg = compile_and_time(make_swiglu_nocg(inter, batched), (x, Wug), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        r_forced = compile_and_time(make_swiglu_forced(inter, batched), (x, Wug), iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        for r in (r_def, r_nocg, r_forced):
            o = r.pop("_cfn_out", None)
            if o is not None:
                r["numerics_ok"] = bool(torch.allclose(o, ref, rtol=2e-2, atol=1e-2))
                r["max_abs"] = _maxabs(o, ref)
            if r["kernels"]:
                r["fused"], r["evidence"] = judge_fused(r["kernels"], act_markers)

        # --- hand Triton GEMM+SwiGLU with a quick tile autotune (dense only) ---
        t_ok = False
        if do_triton and _HAVE_TRITON and not batched:
            try:
                cands = TRITON_CANDS if iters >= 10 else TRITON_CANDS[:2]
                tcfg = tune_triton_swiglu(x, Wug, inter, cands)
                if tcfg is None:
                    triton_info = {"error": "all tile configs failed"}
                else:
                    BM, BN, BK, w, s = tcfg
                    tout = triton_swiglu(x, Wug, inter, BM, BN, BK, w, s)
                    t_rel = _rel_max(tout, ref32)
                    t_ok = bool(t_rel <= rel_tol)
                    # time it regardless; the verdict metric only uses it when numerics_ok
                    triton_s = med_time(
                        lambda: triton_swiglu(x, Wug, inter, BM, BN, BK, w, s),
                        iters, warmup)
                    triton_ms = triton_s * 1e3
                    triton_info = {"numerics_ok": t_ok,
                                   "rel_max_vs_fp32": t_rel,
                                   "eager_rel_max_vs_fp32": eager_rel_max,
                                   "rel_tol": rel_tol,
                                   "max_abs_diff_vs_eager": _maxabs(tout, ref),
                                   "config": f"BM{BM} BN{BN} BK{BK} w{w} s{s} "
                                             f"(tuned over {len(cands)} candidates)"}
            except Exception as exc:
                triton_info = {"error": f"{type(exc).__name__}: {exc}"}
        elif do_triton and batched:
            triton_info = {"skipped": "hand triton kernel is dense-only (bmm not adapted)"}

        # drift probe: re-measure the bare GEMM at the END of the config -- the spread vs the
        # first measurement bounds DVFS/thermal drift across this config's timed region
        gemm_repeat_s = med_time(lambda: (torch.bmm(x, Wug) if batched else x @ Wug),
                                 iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    compiled_ms = r_def["ms"]
    compiled_nocg_ms = r_nocg["ms"]
    forced_ms = r_forced["ms"]
    fused_paths = {"compiled_ms": compiled_ms, "compiled_forced_triton_ms": forced_ms}
    if triton_ms is not None:
        fused_paths["triton_ms"] = triton_ms
    cand = [v for v in (compiled_ms, forced_ms, triton_ms) if v is not None]
    best_fused_ms = min(cand) if cand else None
    measured_gain = (unfused_ms / best_fused_ms) if best_fused_ms else None
    # verified-fused-only gain, against the BEST unfused realization: hand triton is fused
    # by construction; the forced template counts only if its kernel evidence confirms the fold
    unf_cand = [v for v in (unfused_ms, compiled_nocg_ms) if v is not None]
    best_unfused_ms = min(unf_cand)
    vcand = ([triton_ms] if (triton_ms is not None and t_ok) else []) + \
            ([forced_ms] if (forced_ms is not None and r_forced["fused"]) else [])
    measured_gain_verified = (best_unfused_ms / min(vcand)) if vcand else None
    compiled_over_gemm = (compiled_ms / gemm_only_ms) if compiled_ms else None
    nocg_over_gemm = (compiled_nocg_ms / gemm_only_ms) if compiled_nocg_ms else None
    forced_over_gemm = (forced_ms / gemm_only_ms) if forced_ms else None
    triton_over_gemm = (triton_ms / gemm_only_ms) if triton_ms else None

    est_unf, est_fus = est_swiglu_ms(M, hidden, inter, count, gpu)
    row = {
        "name": name, "kind": "swiglu", "dims": dims,
        "gemm_only_ms": gemm_only_ms, "unfused_ms": unfused_ms,
        "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": gemm_repeat_s * 1e3 / gemm_only_ms,
        "eager_rel_max_vs_fp32": eager_rel_max,
        "unfused_compiled_nocg_ms": compiled_nocg_ms,
        "best_unfused_ms": best_unfused_ms,
        "fused_paths": fused_paths,
        "best_fused_ms": best_fused_ms, "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "fused_verified": r_def["fused"],
        "fused_verified_forced": r_forced["fused"],
        "kernel_evidence": r_def["kernels"],
        "fusion_evidence": r_def["evidence"],
        "nocg_kernel_evidence": r_nocg["kernels"],
        "nocg_fusion_evidence": r_nocg["evidence"],
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced["evidence"],
        "numerics_ok": r_def["numerics_ok"], "max_abs_diff": r_def["max_abs"],
        "nocg_numerics_ok": r_nocg["numerics_ok"], "nocg_max_abs_diff": r_nocg["max_abs"],
        "forced_numerics_ok": r_forced["numerics_ok"], "forced_max_abs_diff": r_forced["max_abs"],
        "compiled_over_gemm_ratio": compiled_over_gemm,
        "nocg_over_gemm_ratio": nocg_over_gemm,
        "forced_over_gemm_ratio": forced_over_gemm,
        "triton_over_gemm_ratio": triton_over_gemm,
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "triton_info": triton_info,
        "compile_error": r_def["error"],
        "nocg_error": r_nocg["error"],
        "forced_error": r_forced["error"],
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a_unf, a_fus = est_swiglu_ms(M, hidden, inter, count, gpu_adj)
        row["est_unfused_ms_adj"] = a_unf
        row["est_fused_ms_adj"] = a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    print(f"    gemm_only={gemm_only_ms:.4f}  unfused={unfused_ms:.4f}  "
          f"nocg={compiled_nocg_ms}  compiled={compiled_ms}  forced={forced_ms}  "
          f"triton={triton_ms}  meas_gain={measured_gain}  "
          f"gain_verified={measured_gain_verified}  est_gain={est_unf/est_fus:.4f}  "
          f"fused(def/forced)={r_def['fused']}/{r_forced['fused']}", flush=True)
    return row


def measure_residual(name, M, n, k, iters, warmup, gpu, gpu_adj):
    """Measure + estimate one residual-into-mla_o config (F1-analog, dense)."""
    print(f"[residual] {name}: M={M} n={n} k={k} (autotune may take minutes) ...", flush=True)
    x = bf(M, k)
    Wo = bf(k, n)
    res = bf(M, n)

    act_markers = ("add", "elementwise")
    with ClockSampler() as cs:
        gemm_only_s = med_time(lambda: x @ Wo, iters, warmup)
        unfused_s = med_time(lambda: x @ Wo + res, iters, warmup)
        addmm_s = med_time(lambda: torch.addmm(res, x, Wo), iters, warmup)
        ref = x @ Wo + res
        addmm_out = torch.addmm(res, x, Wo)
        addmm_ok = bool(torch.allclose(addmm_out, ref, rtol=2e-2, atol=1e-2))
        addmm_max_abs = _maxabs(addmm_out, ref)
        addmm_kernels = profile_kernels(lambda: torch.addmm(res, x, Wo))
        addmm_fused, addmm_separate = vendor_fused_ok(addmm_kernels, act_markers)

        # def/nocg/forced compiled variants (see measure_swiglu). For the residual epilogue
        # the unforced compile may legitimately fuse VENDOR-side (inductor emits addmm /
        # cublasLt beta-accumulate) -- judged via vendor_fused_ok on the kernel evidence.
        r_def = compile_and_time(residual_fn, (x, Wo, res), iters, warmup, "max-autotune")
        r_nocg = compile_and_time(residual_fn_nocg, (x, Wo, res), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        r_forced = compile_and_time(residual_fn_forced, (x, Wo, res), iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        for r in (r_def, r_nocg, r_forced):
            o = r.pop("_cfn_out", None)
            if o is not None:
                r["numerics_ok"] = bool(torch.allclose(o, ref, rtol=2e-2, atol=1e-2))
                r["max_abs"] = _maxabs(o, ref)
            if r["kernels"]:
                r["fused"], r["evidence"] = judge_fused(r["kernels"], act_markers)
                vok, vsep = vendor_fused_ok(r["kernels"], act_markers)
                r["evidence"]["vendor_fused_ok"] = vok
                r["evidence"]["vendor_separate_kernels"] = vsep

        # drift probe (see measure_swiglu)
        gemm_repeat_s = med_time(lambda: x @ Wo, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    addmm_ms = addmm_s * 1e3
    compiled_ms = r_def["ms"]
    compiled_nocg_ms = r_nocg["ms"]
    forced_ms = r_forced["ms"]
    fused_paths = {"addmm_ms": addmm_ms, "compiled_ms": compiled_ms,
                   "compiled_nocg_ms": compiled_nocg_ms,
                   "compiled_forced_triton_ms": forced_ms}
    cand = [v for v in (addmm_ms, compiled_ms, compiled_nocg_ms, forced_ms) if v is not None]
    best_fused_ms = min(cand) if cand else None
    measured_gain = (unfused_ms / best_fused_ms) if best_fused_ms else None
    # verified-fused-only gain: addmm counts if its profile shows no separate add kernel;
    # compiled paths count if Triton-template-fused OR vendor-fused per evidence
    vcand = ([addmm_ms] if addmm_fused else [])
    for r, ms in ((r_def, compiled_ms), (r_nocg, compiled_nocg_ms), (r_forced, forced_ms)):
        if ms is not None and (r["fused"] or r.get("evidence", {}).get("vendor_fused_ok")):
            vcand.append(ms)
    measured_gain_verified = (unfused_ms / min(vcand)) if vcand else None
    compiled_over_gemm = (compiled_ms / gemm_only_ms) if compiled_ms else None
    nocg_over_gemm = (compiled_nocg_ms / gemm_only_ms) if compiled_nocg_ms else None
    forced_over_gemm = (forced_ms / gemm_only_ms) if forced_ms else None
    addmm_over_gemm = addmm_ms / gemm_only_ms

    est_unf, est_fus = est_residual_ms(M, n, k, gpu)
    row = {
        "name": name, "kind": "residual", "dims": {"M": M, "n": n, "k": k},
        "gemm_only_ms": gemm_only_ms, "unfused_ms": unfused_ms,
        "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": gemm_repeat_s * 1e3 / gemm_only_ms,
        "best_unfused_ms": unfused_ms,
        "fused_paths": fused_paths,
        "best_fused_ms": best_fused_ms, "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "fused_verified": r_def["fused"],
        "fused_verified_forced": r_forced["fused"],
        "addmm_fused_verified": addmm_fused,
        "addmm_kernel_evidence": addmm_kernels,
        "addmm_separate_kernels": addmm_separate,
        "kernel_evidence": r_def["kernels"],
        "fusion_evidence": r_def["evidence"],
        "nocg_kernel_evidence": r_nocg["kernels"],
        "nocg_fusion_evidence": r_nocg["evidence"],
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced["evidence"],
        "numerics_ok": r_def["numerics_ok"], "max_abs_diff": r_def["max_abs"],
        "nocg_numerics_ok": r_nocg["numerics_ok"], "nocg_max_abs_diff": r_nocg["max_abs"],
        "forced_numerics_ok": r_forced["numerics_ok"], "forced_max_abs_diff": r_forced["max_abs"],
        "addmm_numerics_ok": addmm_ok, "addmm_max_abs_diff": addmm_max_abs,
        "compiled_over_gemm_ratio": compiled_over_gemm,
        "nocg_over_gemm_ratio": nocg_over_gemm,
        "forced_over_gemm_ratio": forced_over_gemm,
        "addmm_over_gemm_ratio": addmm_over_gemm,
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "compile_error": r_def["error"],
        "nocg_error": r_nocg["error"],
        "forced_error": r_forced["error"],
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a_unf, a_fus = est_residual_ms(M, n, k, gpu_adj)
        row["est_unfused_ms_adj"] = a_unf
        row["est_fused_ms_adj"] = a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    print(f"    gemm_only={gemm_only_ms:.4f}  unfused={unfused_ms:.4f}  addmm={addmm_ms:.4f}  "
          f"nocg={compiled_nocg_ms}  compiled={compiled_ms}  forced={forced_ms}  "
          f"meas_gain={measured_gain}  gain_verified={measured_gain_verified}  "
          f"est_gain={est_unf/est_fus:.4f}  addmm_fused={addmm_fused}  "
          f"forced_fused={r_forced['fused']}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# --t2-json adjusted estimator profile                                          #
# --------------------------------------------------------------------------- #
def _find_numeric(obj, patterns):
    """PRIORITY-ordered recursive key search: an earlier pattern beats a later one no
    matter where its key sits in the JSON (dict-walk order only breaks ties within one
    pattern). A walk-order-first search would return e.g. peak_gemm.shapes[0].tflops
    (the FIRST square GEMM, not the peak) from rtx4060_peak.py output."""
    pats = [p.lower() for p in patterns]
    per_pattern = [[] for _ in pats]

    def walk(o):
        if isinstance(o, dict):
            for kk, vv in o.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    kl = str(kk).lower()
                    for i, p in enumerate(pats):
                        if p in kl:
                            per_pattern[i].append(float(vv))
                            break
                walk(vv)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(obj)
    for hits in per_pattern:
        if hits:
            return hits[0]
    return None


def build_adjusted_profile(t2_path, gpu):
    """Adjusted GpuModel from T2's measured peaks.

    Preferred: rtx4060_peak.py writes an `adjusted_profile` block with the exact
    dataclasses.replace overrides (clock_hz, bw_bytes_per_s) it documents for
    downstream consumption -- use it verbatim. Fallback for unknown T2 schemas:
    priority-ordered recursive key search (T2's real key names first: T2 reports
    bandwidth as *_gbs, and its peak lives in measured_peak_tflops, NOT the
    per-shape tflops entries).
    """
    try:
        with open(t2_path) as f:
            data = json.load(f)
    except Exception as exc:
        return None, {"error": f"could not read {t2_path}: {exc}"}

    blk = data.get("adjusted_profile") if isinstance(data, dict) else None
    if (isinstance(blk, dict)
            and isinstance(blk.get("clock_hz"), (int, float))
            and isinstance(blk.get("bw_bytes_per_s"), (int, float))):
        kwargs = {"clock_hz": float(blk["clock_hz"]),
                  "bw_bytes_per_s": float(blk["bw_bytes_per_s"])}
        adj = dataclasses.replace(gpu, name=gpu.name + " +T2adj", **kwargs)
        return adj, {"applied": kwargs, "source": "t2 adjusted_profile block",
                     "check_peak_tflops": adj.peak_tensor_flops / 1e12,
                     "check_bw_gbs": adj.bw_bytes_per_s / 1e9}

    tflops = _find_numeric(data, ("measured_peak_tflops", "peak_bf16_tflops", "peak_tflops",
                                  "bf16_tflops", "measured_tflops", "tflops"))
    gbps = _find_numeric(data, ("measured_bw_gbs", "measured_bw_gbps", "hbm_gbps", "bw_gbps",
                                "bandwidth_gbps", "measured_gbps", "gbps", "gbs"))
    if tflops is None and gbps is None:
        return None, {"note": "no recognizable tflops/bandwidth keys in t2 json; adjusted profile skipped"}
    kwargs = {}
    if gbps is not None:
        kwargs["bw_bytes_per_s"] = gbps * 1e9
    if tflops is not None:
        # peak_tensor_flops is derived: tensor_cores * flops_per_core_per_clock * clock_hz
        kwargs["tensor_flops_per_core_per_clock"] = tflops * 1e12 / (gpu.tensor_cores * gpu.clock_hz)
    adj = dataclasses.replace(gpu, name=gpu.name + " +T2adj", **kwargs)
    return adj, {"applied": {k: kwargs[k] for k in kwargs}, "src_tflops": tflops,
                 "src_gbps": gbps, "source": "pattern search fallback"}


# --------------------------------------------------------------------------- #
# Config sweeps                                                                  #
# --------------------------------------------------------------------------- #
def swiglu_configs(smoke):
    if smoke:
        return [("swiglu_smoke", 512, 512, 512, 1, False)]
    cfgs = []
    for M in (2048, 8192):
        for hidden in (1024, 2048, 4096):
            cfgs.append((f"swiglu_M{M}_h{hidden}", M, hidden, hidden, 1, False))
    return cfgs


def residual_configs(smoke):
    if smoke:
        return [("residual_smoke", 512, 512, 512)]      # (name, M, n, k) tiny
    # task ordering: x[M,6144]@Wo[6144,16384]+res[M,16384]  (n=16384, k=6144)
    # GLM/C500 ordering: x[M,16384]@Wo[16384,6144]+res[M,6144]  (n=6144, k=16384) --
    # the actual mla_o+residual layer the C500 measured; both fit 8 GiB, measure both.
    return ([(f"residual_M{M}", M, 16384, 6144) for M in (2048, 8192)]
            + [(f"residual_glm_M{M}", M, 6144, 16384) for M in (2048, 8192)])


def moe_configs(smoke):
    # grouped-bmm MoE: (name, tokens_per_expert, hidden, inter, experts)
    if smoke:
        return [("moe_smoke", 64, 512, 512, 4)]
    cfgs = []
    for E in (8, 32):
        cfgs.append((f"moe_E{E}", 128, 2048, 2048, E))
    return cfgs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny dims, iters=3/warmup=3, whole path end-to-end")
    ap.add_argument("--moe", action="store_true", help="add grouped-bmm MoE swiglu configs")
    ap.add_argument("--t2-json", default=None,
                    help="T2 measured-peaks JSON; also report estimator gains under an adjusted profile")
    args = ap.parse_args()

    # Fail fast on an unwritable --out BEFORE spending GPU time on compiles.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    iters, warmup = (3, 3) if args.smoke else (30, 15)

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
        "clocks_locked": False,
        "clocks_note": "WSL2 / no root: clocks NOT locked -> ClockSampler recorded per config",
        "smoke": args.smoke,
        "adjusted_profile": adj_info,
    }
    conventions = {
        "time_unit": "milliseconds for every field ending in _ms",
        "med_time_returns": "SECONDS (rtx4060_common.med_time); converted to ms here (*1e3)",
        "estimator_time_unit": "SECONDS (KTime.time_s); converted to ms here (*1e3)",
        "gain_definition": "gain = unfused_time / fused_time  (>1.0 => fusion FASTER), matches estimator '1.020x'",
        "measured_gain": "unfused_ms / best_fused_ms  (best over compiled/addmm/triton paths)",
        "estimated_gain": "est_unfused_ms / est_fused_ms  (fusion_time_estimator, rtx4060-measured profile)",
        "dtype": "bfloat16 for all measured tensors; fp32 accumulate",
        "hand_triton_numerics": "judged vs an fp32 reference: ok iff rel_max_vs_fp32 <= "
                          "max(2 x eager path's own rel_max_vs_fp32, 5e-2). The hand kernel keeps "
                          "gate/up in fp32 accumulators (MORE accurate than eager, which rounds gu "
                          "to bf16 pre-silu), so allclose-vs-eager is the wrong criterion at large k",
        "gemm_drift_ratio": "bare-GEMM time re-measured at the END of the config / at the START; "
                          "bounds DVFS+thermal drift across the config's timed region (~1.0 = clean)",
        "fused_verified": "True iff a Triton GEMM template folds the epilogue in AND no separate "
                          "silu/mul/add elementwise kernel remains (raw kernel_evidence stored for re-judge)",
        "compiled_paths": "compiled_ms = plain max-autotune (cudagraphs ON; is_big_gpu()<68 SMs so "
                          "inductor keeps the vendor GEMM + separate epilogue kernel on this GPU); "
                          "unfused_compiled_nocg_ms / compiled_nocg_ms = max-autotune-no-cudagraphs "
                          "unforced (for swiglu this is the BEST UNFUSED realization: vendor GEMM + one "
                          "fused pointwise kernel, exactly the estimator's unfused model); "
                          "compiled_forced_triton_ms = is_big_gpu patched + TRITON-only autotune "
                          "backends + no-cudagraphs -> a genuine Triton GEMM template with the epilogue "
                          "folded in (the object the task asks about); fused_verified_forced judges it",
        "measured_gain_verified": "best_unfused_ms / best VERIFIED-fused path (hand triton by "
                          "construction; forced template iff fused_verified_forced; addmm iff "
                          "addmm_fused_verified; compiled paths iff template- or vendor-fused per "
                          "evidence). The A-vs-B verdict keys on THIS, not measured_gain",
        "addmm_fused_verified": "residual only: torch.addmm profile shows no separate add/elementwise "
                          "kernel (Memcpy DtoD tolerated: beta-accumulate copy)",
        "compiled_cudagraph_input_copy_us": "GPU time of multi_tensor_apply static-input-copy kernels "
                          "inside ONE compiled call (cudagraph deployment overhead, not fusion work); "
                          "subtract from compiled_ms for a kernels-only fused-vs-unfused comparison",
        "timing": f"median of {iters} cuda-event samples, {warmup} warmup, per config",
    }

    print(f"=== T4 fusion realizability  (smoke={args.smoke}, moe={args.moe}) ===", flush=True)
    print(f"device={env['device_name']} torch={env['torch_version']} triton={env['triton_version']}",
          flush=True)

    results = []
    out = {"conventions": conventions, "env": env, "configs": results}

    def run_config(kind, name, thunk):
        """Isolate one config: an exception records an error row instead of killing the
        sweep, and the JSON is re-saved after EVERY config (partial-result safety)."""
        try:
            row = thunk()
        except Exception as exc:
            traceback.print_exc()
            row = {"name": name, "kind": kind, "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)
        save_json(args.out, out)

    for name, M, hidden, inter, count, batched in swiglu_configs(args.smoke):
        run_config("swiglu", name,
                   lambda: measure_swiglu(name, M, hidden, inter, iters, warmup, gpu, gpu_adj,
                                          count=count, batched=batched))
    for name, M, n, k in residual_configs(args.smoke):
        run_config("residual", name,
                   lambda: measure_residual(name, M, n, k, iters, warmup, gpu, gpu_adj))
    if args.moe:
        for name, tpe, hidden, inter, E in moe_configs(args.smoke):
            run_config("swiglu", name,
                       lambda: measure_swiglu(name, tpe, hidden, inter, iters, warmup, gpu,
                                              gpu_adj, count=E, batched=True, do_triton=False))

    # -------- human-readable summary --------
    ok_rows = [r for r in results if "error" not in r]
    err_rows = [r for r in results if "error" in r]
    print("\n" + "=" * 130)
    print(f"{'config':<20}{'kind':<10}{'unfused':>9}{'best_fus':>10}{'meas_gain':>10}"
          f"{'gain_ver':>9}{'est_gain':>9}{'forced?':>8}{'num_ok':>7}{'forc/gemm':>10}")
    print("-" * 130)
    for r in ok_rows:
        bf_ms = r["best_fused_ms"]
        mg = r["measured_gain"]
        mgv = r.get("measured_gain_verified")
        fog = r.get("forced_over_gemm_ratio")
        print(f"{r['name']:<20}{r['kind']:<10}{r['unfused_ms']:>9.4f}"
              f"{(bf_ms if bf_ms is not None else float('nan')):>10.4f}"
              f"{(mg if mg is not None else float('nan')):>10.4f}"
              f"{(mgv if mgv is not None else float('nan')):>9.4f}"
              f"{r['estimated_gain']:>9.4f}{str(r.get('fused_verified_forced')):>8}"
              f"{str(r['numerics_ok']):>7}"
              f"{(fog if fog is not None else float('nan')):>10.4f}")
    for r in err_rows:
        print(f"{r['name']:<20}{r['kind']:<10}  ERROR: {r['error']}")
    print("-" * 130)
    meas_gains = [r["measured_gain"] for r in ok_rows if r["measured_gain"]]
    ver_gains = [r["measured_gain_verified"] for r in ok_rows if r.get("measured_gain_verified")]
    est_gains = [r["estimated_gain"] for r in ok_rows if r["estimated_gain"]]
    if meas_gains:
        ver_part = (f"verified-fused gain geomean: {geomean(ver_gains):.4f}   "
                    if ver_gains else "verified-fused gain geomean: n/a   ")
        print(f"measured gain geomean: {geomean(meas_gains):.4f}   " + ver_part
              + f"estimated gain geomean: {geomean(est_gains):.4f}")
    n_def = sum(1 for r in ok_rows if r["fused_verified"])
    n_forced = sum(1 for r in ok_rows if r.get("fused_verified_forced"))
    print(f"fused (Triton template folded epilogue): default-compile {n_def}/{len(results)}, "
          f"forced-template {n_forced}/{len(results)} configs")
    if err_rows:
        print(f"WARNING: {len(err_rows)} config(s) errored (rows kept in JSON with an 'error' field)")
    print("=" * 130)


if __name__ == "__main__":
    main()
