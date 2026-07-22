"""T6.A (RTX4060_SIM_REAL_TASK.md) - RMSNorm fused into mla_o (epilogue, structurally
infeasible) + prologue fallback (up_gate / router hosts).

HOST DECISIONS (asked and answered, 2026-07-22):
  * epilogue prefill capped at tokens <= 32768 (131072 omitted; recorded in env);
  * the epilogue input INCLUDES a pre-materialized residual: the op under test is
      y = RMSNorm(x @ Wo + res, gamma)          (layer-faithful S3 composition)
    The residual-add is tile-local (F1); the cross-tile reduction over N remains the crux.

Placements:
  EPILOGUE (primary): fold the norm into mla_o [M, N=HIDDEN=6144, K=KV=16384].
    The norm reduces over the OUTPUT dim N, split across CTAs -> cross-tile, strictly
    harder than SwiGLU. Expected: no stock path fuses; the hand wide-tile kernel fails the
    99 KiB SMEM budget (fp32 stage BM*6144*4 = 384 KiB at BM=16) -> documented infeasible.
  PROLOGUE (fallback): fold the norm into the NEXT GEMM's A-load (reduction over K,
    tile-local to one CTA's K-loop). Hosts: up_gate [tpe, 4096, 6144] (primary, per-expert)
    and router [tokens, 256, 6144] (secondary, dense). Hand kernels:
      P2 (recommended, F3-analog): kernel1 rowwise sumsq -> inv_rms[M] (fp32);
         kernel2 tiled GEMM whose prologue scales each A-tile by inv_rms[row]*gamma[k].
      P1 (secondary): single kernel, two passes over K (reads A twice, a_factor=2.0).

Estimator: est_rms_epilogue_ms / est_rms_prologue_ms exactly per spec A.5. The EPILOGUE
est cell is STRUCTURE-BLIND (estimate_fused_gemm has no cross-N-reduction term) -> rows
carry est_invalid_structure_blind=True and are excluded from aggregate est-vs-measured
stats. The PROLOGUE est is the valid F3-analog yardstick.

UNITS: med_time/KTime.time_s are SECONDS; every *_ms JSON field is milliseconds.
gain = unfused/fused (>1 => fusion faster).

Run (from GEMM-mapping/):
    python rtx4060_rmsnorm_fuse.py --out rtx4060_rmsnorm.json --t2-json rtx4060_peak.json
    python rtx4060_rmsnorm_fuse.py --smoke --out /tmp/.../rms_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# T4 infrastructure by IMPORT (also applies the max-autotune inductor config + dynamo limits)
from rtx4060_fusion_measure import (  # noqa: E402
    _cudagraph_copy_us,
    _maxabs,
    _rel_max,
    build_adjusted_profile,
    compile_and_time,
    judge_fused,
    profile_kernels,
)
from rtx4060_common import (  # noqa: E402
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
    _residual_rms_aux,
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

HIDDEN, KV, TWOI, EXPERTS = 6144, 16384, 4096, 256
EPS = 1e-6
RMS_STAT = 4  # fp32 per-row stat

torch.manual_seed(0)
_GAMMA = {}


def gamma_for(h):
    """gamma drawn once per width (T6.0.1: randn*0.1+1), identical baseline vs fused."""
    if h not in _GAMMA:
        g = torch.Generator(device="cpu").manual_seed(1234 + h)
        _GAMMA[h] = (torch.randn(h, generator=g) * 0.1 + 1).to("cuda:0", torch.bfloat16)
    return _GAMMA[h]


def rmsnorm_ref32(h32, gamma):
    return h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()


# --------------------------------------------------------------------------- #
# Distinct code objects per compile variant (dynamo caches on the code object)  #
# --------------------------------------------------------------------------- #
def make_epi_full(gamma):
    def rms_epi_full(x, Wo, res):
        return F.rms_norm((x @ Wo + res).to(torch.bfloat16), (gamma.shape[0],), gamma, EPS)
    return rms_epi_full


def make_epi_full_nocg(gamma):
    def rms_epi_full_nocg(x, Wo, res):
        return F.rms_norm((x @ Wo + res).to(torch.bfloat16), (gamma.shape[0],), gamma, EPS)
    return rms_epi_full_nocg


def make_epi_full_forced(gamma):
    def rms_epi_full_forced(x, Wo, res):
        return F.rms_norm((x @ Wo + res).to(torch.bfloat16), (gamma.shape[0],), gamma, EPS)
    return rms_epi_full_forced


def make_epi_tail(gamma):
    """add+norm only (the vendor-GEMM-plus-ONE-fused-kernel unfused realization)."""
    def rms_epi_tail(out, res):
        return F.rms_norm((out + res).to(torch.bfloat16), (gamma.shape[0],), gamma, EPS)
    return rms_epi_tail


def make_pro_full(gamma):
    def rms_pro_full(h, W):
        return F.rms_norm(h, (gamma.shape[0],), gamma, EPS) @ W
    return rms_pro_full


def make_pro_full_nocg(gamma):
    def rms_pro_full_nocg(h, W):
        return F.rms_norm(h, (gamma.shape[0],), gamma, EPS) @ W
    return rms_pro_full_nocg


def make_pro_full_forced(gamma):
    def rms_pro_full_forced(h, W):
        return F.rms_norm(h, (gamma.shape[0],), gamma, EPS) @ W
    return rms_pro_full_forced


# --------------------------------------------------------------------------- #
# Hand kernels                                                                  #
# --------------------------------------------------------------------------- #
if _HAVE_TRITON:

    @triton.jit
    def _wide_tile_rms_epi_kernel(X, W, RES, GAMMA, Y, M,
                                  sxm, sxk, swk, swn, srm, srn, sym, syn,
                                  K: tl.constexpr, N: tl.constexpr,
                                  BM: tl.constexpr, BK: tl.constexpr):
        # A.4 option 1: one CTA owns the FULL [BM, N] output row-block so the N-reduction
        # is CTA-local. fp32 accumulator [BM, N] = BM*N*4 bytes of on-chip state -- at
        # BM=16, N=6144 that is 384 KiB >> 99 KiB: expected OutOfResources at full dims.
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, N)
        offs_k = tl.arange(0, BK)
        m_mask = offs_m[:, None] < M
        acc = tl.zeros((BM, N), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(X + offs_m[:, None] * sxm + (k0 + offs_k)[None, :] * sxk,
                        mask=m_mask, other=0.0)
            w = tl.load(W + (k0 + offs_k)[:, None] * swk + offs_n[None, :] * swn)
            acc += tl.dot(a, w)
        r = tl.load(RES + offs_m[:, None] * srm + offs_n[None, :] * srn,
                    mask=m_mask, other=0.0)
        h = acc + r.to(tl.float32)
        ms = tl.sum(h * h, axis=1) / N
        inv = 1.0 / tl.sqrt(ms + 1e-6)
        g = tl.load(GAMMA + offs_n)
        y = h * inv[:, None] * g.to(tl.float32)[None, :]
        tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn,
                 y.to(Y.dtype.element_ty), mask=m_mask)

    def wide_tile_rms_epi(x, Wo, res, gamma, BM=16, BK=32, num_warps=8, num_stages=1):
        M, K = x.shape
        N = Wo.shape[1]
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        _wide_tile_rms_epi_kernel[(triton.cdiv(M, BM),)](
            x, Wo, res, gamma, y, M,
            x.stride(0), x.stride(1), Wo.stride(0), Wo.stride(1),
            res.stride(0), res.stride(1), y.stride(0), y.stride(1),
            K=K, N=N, BM=BM, BK=BK, num_warps=num_warps, num_stages=num_stages)
        return y

    @triton.jit
    def _row_sumsq_kernel(H, INV, M, shm, shk,
                          K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        # P2 kernel 1: inv_rms[i] = 1/sqrt(mean_k h[i,k]^2 + eps), fp32.
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        offs_k = tl.arange(0, BK)
        m_mask = offs_m < M
        acc = tl.zeros((BM,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(H + offs_m[:, None] * shm + (k0 + offs_k)[None, :] * shk,
                        mask=m_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(a * a, axis=1)
        inv = 1.0 / tl.sqrt(acc / K + 1e-6)
        tl.store(INV + offs_m, inv, mask=m_mask)

    @triton.jit
    def _rms_pro_gemm_kernel(H, W, INV, GAMMA, Y, M, N,
                             shm, shk, swk, swn, sym, syn,
                             K: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # P2 kernel 2: tiled GEMM; prologue scales each A-tile by inv_rms[row]*gamma[k]
        # (rounded to bf16 like the eager normalized activation) before the MMA.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        inv = tl.load(INV + offs_m, mask=offs_m < M, other=0.0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(H + offs_m[:, None] * shm + (k0 + offs_k)[None, :] * shk,
                        mask=m_mask, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + k0 + offs_k).to(tl.float32)
            an = (a * inv[:, None] * g[None, :]).to(tl.bfloat16)
            w = tl.load(W + (k0 + offs_k)[:, None] * swk + offs_n[None, :] * swn,
                        mask=n_mask, other=0.0)
            acc += tl.dot(an, w)
        tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn,
                 acc.to(Y.dtype.element_ty), mask=m_mask & n_mask)

    @triton.jit
    def _rms_pro_gemm_2pass_kernel(H, W, GAMMA, Y, M, N,
                                   shm, shk, swk, swn, sym, syn,
                                   K: tl.constexpr,
                                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # P1: single kernel, pass 1 computes the row sumsq (reads A once), pass 2 re-reads
        # A scaled for the MMA (a_factor ~ 2.0). No inv_rms round-trip at all.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        acc_ss = tl.zeros((BM,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(H + offs_m[:, None] * shm + (k0 + offs_k)[None, :] * shk,
                        mask=m_mask, other=0.0).to(tl.float32)
            acc_ss += tl.sum(a * a, axis=1)
        inv = 1.0 / tl.sqrt(acc_ss / K + 1e-6)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(H + offs_m[:, None] * shm + (k0 + offs_k)[None, :] * shk,
                        mask=m_mask, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + k0 + offs_k).to(tl.float32)
            an = (a * inv[:, None] * g[None, :]).to(tl.bfloat16)
            w = tl.load(W + (k0 + offs_k)[:, None] * swk + offs_n[None, :] * swn,
                        mask=n_mask, other=0.0)
            acc += tl.dot(an, w)
        tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn,
                 acc.to(Y.dtype.element_ty), mask=m_mask & n_mask)

    PRO_CANDS = [(64, 64, 32, 4, 2), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
                 (128, 128, 32, 8, 3), (64, 64, 64, 4, 3), (32, 64, 32, 4, 2)]

    def p2_rms_pro(h, W, gamma, inv_buf, BM=64, BN=64, BK=32, num_warps=4, num_stages=2):
        M, K = h.shape
        N = W.shape[1]
        y = torch.empty((M, N), device=h.device, dtype=h.dtype)
        _row_sumsq_kernel[(triton.cdiv(M, 64),)](
            h, inv_buf, M, h.stride(0), h.stride(1), K=K, BM=64, BK=256, num_warps=8)
        _rms_pro_gemm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            h, W, inv_buf, gamma, y, M, N,
            h.stride(0), h.stride(1), W.stride(0), W.stride(1), y.stride(0), y.stride(1),
            K=K, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages)
        return y

    def p1_rms_pro(h, W, gamma, BM=64, BN=64, BK=32, num_warps=4, num_stages=2):
        M, K = h.shape
        N = W.shape[1]
        y = torch.empty((M, N), device=h.device, dtype=h.dtype)
        _rms_pro_gemm_2pass_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            h, W, gamma, y, M, N,
            h.stride(0), h.stride(1), W.stride(0), W.stride(1), y.stride(0), y.stride(1),
            K=K, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages)
        return y

    def tune_pro(make_call, cands):
        best = None
        for cfg in cands:
            try:
                t = med_time(make_call(cfg), iters=5, warmup=3)
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        return best[1] if best else None


# --------------------------------------------------------------------------- #
# Estimator side (spec A.5, verbatim formulas)                                  #
# --------------------------------------------------------------------------- #
def est_rms_epilogue_ms(M, hidden, kv, gpu):
    """STRUCTURE-BLIND (no cross-N-reduction term in Epilogue) - flagged, not a prediction.
    Residual variant (host decision): F2-analog -- unfused adds the residual vector kernel;
    fused adds the residual read + residual/gamma/stat aux SMEM (fusion_time_estimator F2)."""
    unf_g = estimate_gemm_grouped("mla_o", M, hidden, kv, 1, gpu)
    unf_res = estimate_vector_kernel("residual", 3 * M * hidden * BPE, gpu)
    unf_r = estimate_vector_kernel("rmsnorm", 2 * M * hidden * BPE, gpu)
    fus = estimate_fused_gemm("mla_o+res+rms_epilogue", M, hidden, kv, 1,
                              Epilogue(extra_hbm_once=M * hidden * BPE + M * RMS_STAT,
                                       aux_smem_per_tile=_residual_rms_aux), gpu)
    return (unf_g.time_s + unf_res.time_s + unf_r.time_s) * 1e3, fus.time_s * 1e3


def est_rms_prologue_ms(M, twoI, hidden, count, gpu, a_factor=1.0):
    """F3-analog, faithful (reduction legitimately K-local). a_factor 1.0=P2, 2.0=P1."""
    unf_g = estimate_gemm_grouped("up_gate", M, twoI, hidden, count, gpu)
    unf_r = estimate_vector_kernel("rmsnorm", M * hidden * BPE + M * RMS_STAT, gpu)
    fus = estimate_fused_gemm("up_gate+rms_prologue", M, twoI, hidden, count,
                              Epilogue(a_factor=a_factor,
                                       aux_smem_per_tile=lambda m0, n0: m0 * RMS_STAT), gpu)
    return (unf_g.time_s + unf_r.time_s) * 1e3, fus.time_s * 1e3


def est_p2_sumsq_ms(M, hidden, gpu):
    """P2's kernel-1 cost (extra A read + stat write) -- NOT in the spec's est formula;
    reported as an auxiliary field so the honest 2-kernel P2 prediction is available."""
    return estimate_vector_kernel("sumsq", M * hidden * BPE + M * RMS_STAT, gpu).time_s * 1e3


# --------------------------------------------------------------------------- #
# Measurement: EPILOGUE                                                         #
# --------------------------------------------------------------------------- #
EPI_MARKERS = ("rms", "norm", "sqrt", "add", "reduce", "elementwise", "red_fused")


def measure_rms_epilogue(name, M, hidden, kv, regime, iters, warmup, gpu, gpu_adj,
                         smoke=False):
    print(f"[rms_epilogue] {name}: M={M} N={hidden} K={kv} ({regime}) ...", flush=True)
    gamma = gamma_for(hidden)
    x, Wo, res = bf(M, kv), bf(kv, hidden), bf(M, hidden)
    epi_full = make_epi_full(gamma)

    # fp32 truth on a row-slice for big M (memory guard; reduction is rowwise so a slice
    # is exact for the rows it covers)
    ns = min(M, 4096)
    h32 = x[:ns].float() @ Wo.float() + res[:ns].float()
    ref32 = rmsnorm_ref32(h32, gamma)
    del h32
    ref = epi_full(x, Wo, res)
    eager_rel = _rel_max(ref[:ns], ref32)
    rel_tol = max(2.0 * eager_rel, 5e-2)

    skip_cudagraph = M >= 32768  # graph static copies of x (1 GiB) would double footprint
    with ClockSampler() as cs:
        gemm_only_s = med_time(lambda: x @ Wo, iters, warmup)
        # eager unfused: mm + add + rms_norm (3 kernels)
        unfused_s = med_time(lambda: epi_full(x, Wo, res), iters, warmup)
        # vendor GEMM + ONE compiled fused add+norm tail kernel (2 kernels)
        tail = compile_and_time(make_epi_tail(gamma), (x @ Wo, res), iters, warmup,
                                "max-autotune-no-cudagraphs")
        tail_full_s = None
        if tail["ms"] is not None:
            tail_cfn = torch.compile(make_epi_tail(gamma), mode="max-autotune-no-cudagraphs",
                                     dynamic=False)
            tail_cfn(x @ Wo, res)
            tail_full_s = med_time(lambda: tail_cfn(x @ Wo, res), iters, warmup)

        r_def = (compile_and_time(make_epi_full(gamma), (x, Wo, res), iters, warmup,
                                  "max-autotune")
                 if not skip_cudagraph else {"ms": None, "kernels": [], "fused": False,
                                             "evidence": {}, "numerics_ok": None,
                                             "max_abs": None,
                                             "error": "skipped: cudagraph static copies "
                                                      "would double the >2.4 GiB footprint"})
        r_nocg = compile_and_time(make_epi_full_nocg(gamma), (x, Wo, res), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        r_forced = compile_and_time(make_epi_full_forced(gamma), (x, Wo, res), iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        for r in (r_def, r_nocg, r_forced):
            o = r.pop("_cfn_out", None)
            if o is not None:
                r["numerics_ok"] = bool(_rel_max(o[:ns], ref32) <= rel_tol)
                r["max_abs"] = _maxabs(o[:ns], ref[:ns])
            if r["kernels"]:
                r["fused"], r["evidence"] = judge_fused(r["kernels"], EPI_MARKERS)

        # hand wide-tile ATTEMPT (A.4 option 1) at smallest MMA-legal BM
        wide = {"ms": None, "numerics_ok": None, "infeasible_reason": None,
                "smem_budget_note": f"fp32 stage BM*N*4 = 16*{hidden}*4 = "
                                    f"{16 * hidden * 4} B vs smem_per_block 101376 B"}
        if _HAVE_TRITON:
            try:
                yw = wide_tile_rms_epi(x, Wo, res, gamma, BM=16, BK=32)
                w_rel = _rel_max(yw[:ns], ref32)
                wide["numerics_ok"] = bool(w_rel <= rel_tol)
                wide["rel_max_vs_fp32"] = w_rel
                if wide["numerics_ok"]:
                    wide["ms"] = med_time(lambda: wide_tile_rms_epi(x, Wo, res, gamma,
                                                                    BM=16, BK=32),
                                          iters, warmup) * 1e3
            except Exception as exc:
                wide["infeasible_reason"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        else:
            wide["infeasible_reason"] = "triton unavailable"

        gemm_repeat_s = med_time(lambda: x @ Wo, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    tail_full_ms = tail_full_s * 1e3 if tail_full_s else None
    unf_cand = [v for v in (unfused_ms, tail_full_ms) if v is not None]
    best_unfused_ms = min(unf_cand)
    fused_paths = {"compiled_ms": r_def["ms"], "compiled_nocg_ms": r_nocg["ms"],
                   "compiled_forced_triton_ms": r_forced["ms"],
                   "triton_wide_tile_ms": wide["ms"]}
    cand = [v for v in fused_paths.values() if v is not None]
    best_fused_ms = min(cand) if cand else None
    measured_gain = (best_unfused_ms / best_fused_ms) if best_fused_ms else None
    vcand = [ms for ms, r in ((r_def["ms"], r_def), (r_nocg["ms"], r_nocg),
                              (r_forced["ms"], r_forced))
             if ms is not None and r.get("fused") and r.get("numerics_ok")]
    if wide["ms"] is not None and wide["numerics_ok"]:
        vcand.append(wide["ms"])
    measured_gain_verified = (best_unfused_ms / min(vcand)) if vcand else None

    est_unf, est_fus = est_rms_epilogue_ms(M, hidden, kv, gpu)
    row = {
        "name": name, "kind": "rms_epilogue", "regime": regime,
        "dims": {"M": M, "N": hidden, "K": kv, "residual_included": True},
        "gemm_only_ms": gemm_only_ms, "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": gemm_repeat_s * 1e3 / gemm_only_ms,
        "unfused_ms": unfused_ms, "unfused_gemm_plus_tail_ms": tail_full_ms,
        "best_unfused_ms": best_unfused_ms,
        "addmm_ms": None,
        "addmm_reason": "DROP: cuBLASLt beta-accumulate epilogue is tile-local elementwise; "
                        "no RMSNorm/cross-N-reduction epilogue exists in torch's binding",
        "fused_paths": fused_paths,
        "best_fused_ms": best_fused_ms,
        "measured_gain": measured_gain,
        "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf, "est_fused_ms": est_fus,
        "estimated_gain": est_unf / est_fus,
        "est_invalid_structure_blind": True,
        "est_note": "estimate_fused_gemm has NO cross-N-reduction term (tile-local Epilogue "
                    "only) -> this cell is NOT a prediction; excluded from est-vs-measured "
                    "aggregates; standalone structural-mismatch datapoint only",
        "fused_verified": r_def.get("fused", False),
        "fused_verified_nocg": r_nocg.get("fused", False),
        "fused_verified_forced": r_forced.get("fused", False),
        "kernel_evidence": r_def["kernels"], "fusion_evidence": r_def["evidence"],
        "nocg_kernel_evidence": r_nocg["kernels"], "nocg_fusion_evidence": r_nocg["evidence"],
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced["evidence"],
        "wide_tile_attempt": wide,
        "numerics": {"eager_rel_max_vs_fp32": eager_rel, "rel_tol": rel_tol,
                     "rows_checked": ns,
                     "compiled_ok": r_def.get("numerics_ok"),
                     "nocg_ok": r_nocg.get("numerics_ok"),
                     "forced_ok": r_forced.get("numerics_ok")},
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "compile_errors": {"compiled": r_def["error"], "nocg": r_nocg["error"],
                           "forced": r_forced["error"], "tail": tail["error"]},
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a_unf, a_fus = est_rms_epilogue_ms(M, hidden, kv, gpu_adj)
        row["est_unfused_ms_adj"], row["est_fused_ms_adj"] = a_unf, a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    del x, Wo, res, ref, ref32
    torch.cuda.empty_cache()
    print(f"    gemm={gemm_only_ms:.3f} unf={unfused_ms:.3f} best_unf={best_unfused_ms:.3f} "
          f"nocg={r_nocg['ms']} forced={r_forced['ms']} wide={wide['ms']} "
          f"gain={measured_gain} gain_ver={measured_gain_verified} "
          f"fused(def/nocg/forced)={row['fused_verified']}/{row['fused_verified_nocg']}/"
          f"{row['fused_verified_forced']} drift={row['gemm_drift_ratio']:.3f}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# Measurement: PROLOGUE                                                         #
# --------------------------------------------------------------------------- #
def measure_rms_prologue(name, M, n, hidden, host, regime, iters, warmup, gpu, gpu_adj,
                         smoke=False):
    print(f"[rms_prologue] {name}: M={M} N={n} K={hidden} host={host} ({regime}) ...",
          flush=True)
    gamma = gamma_for(hidden)
    h, W = bf(M, hidden), bf(hidden, n)
    pro_full = make_pro_full(gamma)

    ref32 = rmsnorm_ref32(h.float(), gamma).to(torch.bfloat16).float() @ W.float()
    ref = pro_full(h, W)
    eager_rel = _rel_max(ref, ref32)
    rel_tol = max(2.0 * eager_rel, 5e-2)

    with ClockSampler() as cs:
        gemm_only_s = med_time(lambda: h @ W, iters, warmup)
        unfused_s = med_time(lambda: pro_full(h, W), iters, warmup)  # norm kernel + GEMM
        r_def = compile_and_time(make_pro_full(gamma), (h, W), iters, warmup, "max-autotune")
        r_nocg = compile_and_time(make_pro_full_nocg(gamma), (h, W), iters, warmup,
                                  "max-autotune-no-cudagraphs")
        r_forced = compile_and_time(make_pro_full_forced(gamma), (h, W), iters, warmup,
                                    "max-autotune-no-cudagraphs", forced=True)
        for r in (r_def, r_nocg, r_forced):
            o = r.pop("_cfn_out", None)
            if o is not None:
                r["numerics_ok"] = bool(_rel_max(o, ref32) <= rel_tol)
                r["max_abs"] = _maxabs(o, ref)
            if r["kernels"]:
                r["fused"], r["evidence"] = judge_fused(r["kernels"], EPI_MARKERS)

        # hand P2 (2 kernels: sumsq -> prologue GEMM) and P1 (single, two-pass)
        p2 = {"ms": None, "numerics_ok": None, "config": None, "error": None, "kernels": []}
        p1 = {"ms": None, "numerics_ok": None, "config": None, "error": None}
        if _HAVE_TRITON:
            inv_buf = torch.empty(M, device="cuda:0", dtype=torch.float32)
            try:
                cands = PRO_CANDS if iters >= 10 else PRO_CANDS[:2]
                cfg = tune_pro(lambda c: (lambda: p2_rms_pro(h, W, gamma, inv_buf,
                                                             *c[:3], num_warps=c[3],
                                                             num_stages=c[4])), cands)
                if cfg is not None:
                    o2 = p2_rms_pro(h, W, gamma, inv_buf, *cfg[:3], num_warps=cfg[3],
                                    num_stages=cfg[4])
                    rel2 = _rel_max(o2, ref32)
                    p2["numerics_ok"] = bool(rel2 <= rel_tol)
                    p2["rel_max_vs_fp32"] = rel2
                    p2["config"] = f"BM{cfg[0]} BN{cfg[1]} BK{cfg[2]} w{cfg[3]} s{cfg[4]}"
                    p2["ms"] = med_time(lambda: p2_rms_pro(h, W, gamma, inv_buf, *cfg[:3],
                                                           num_warps=cfg[3],
                                                           num_stages=cfg[4]),
                                        iters, warmup) * 1e3
                    p2["kernels"] = profile_kernels(
                        lambda: p2_rms_pro(h, W, gamma, inv_buf, *cfg[:3],
                                           num_warps=cfg[3], num_stages=cfg[4]))
                else:
                    p2["error"] = "no P2 tile candidate compiled"
            except Exception as exc:
                p2["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                traceback.print_exc()
            try:
                cands = PRO_CANDS if iters >= 10 else PRO_CANDS[:2]
                cfg = tune_pro(lambda c: (lambda: p1_rms_pro(h, W, gamma, *c[:3],
                                                             num_warps=c[3],
                                                             num_stages=c[4])), cands)
                if cfg is not None:
                    o1 = p1_rms_pro(h, W, gamma, *cfg[:3], num_warps=cfg[3],
                                    num_stages=cfg[4])
                    rel1 = _rel_max(o1, ref32)
                    p1["numerics_ok"] = bool(rel1 <= rel_tol)
                    p1["rel_max_vs_fp32"] = rel1
                    p1["config"] = f"BM{cfg[0]} BN{cfg[1]} BK{cfg[2]} w{cfg[3]} s{cfg[4]}"
                    p1["ms"] = med_time(lambda: p1_rms_pro(h, W, gamma, *cfg[:3],
                                                           num_warps=cfg[3],
                                                           num_stages=cfg[4]),
                                        iters, warmup) * 1e3
                else:
                    p1["error"] = "no P1 tile candidate compiled"
            except Exception as exc:
                p1["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

        gemm_repeat_s = med_time(lambda: h @ W, iters, warmup)
    clocks = cs.summary()
    sleep_cooldown()

    gemm_only_ms = gemm_only_s * 1e3
    unfused_ms = unfused_s * 1e3
    nocg_ms = r_nocg["ms"]
    # nocg (vendor GEMM + one norm kernel) is an UNFUSED realization here (the norm's
    # normalized-A round-trip survives) unless its evidence shows a template fold
    unf_cand = [v for v in (unfused_ms, None if r_nocg.get("fused") else nocg_ms)
                if v is not None]
    best_unfused_ms = min(unf_cand)
    fused_paths = {"compiled_ms": r_def["ms"],
                   "compiled_forced_triton_ms": r_forced["ms"],
                   "triton_p2_ms": p2["ms"], "triton_p1_ms": p1["ms"]}
    cand = [v for v in fused_paths.values() if v is not None]
    best_fused_ms = min(cand) if cand else None
    measured_gain = (best_unfused_ms / best_fused_ms) if best_fused_ms else None
    vcand = [ms for ms, ok in ((p2["ms"], p2["numerics_ok"]), (p1["ms"], p1["numerics_ok"]))
             if ms is not None and ok]
    for ms, r in ((r_def["ms"], r_def), (r_forced["ms"], r_forced)):
        if ms is not None and r.get("fused") and r.get("numerics_ok"):
            vcand.append(ms)
    measured_gain_verified = (best_unfused_ms / min(vcand)) if vcand else None

    count = 1  # standalone host GEMM (per-expert count handled at layer level, T6.0.2)
    est_unf, est_fus2 = est_rms_prologue_ms(M, n, hidden, count, gpu, a_factor=1.0)
    _, est_fus1 = est_rms_prologue_ms(M, n, hidden, count, gpu, a_factor=2.0)
    est_sumsq = est_p2_sumsq_ms(M, hidden, gpu)
    row = {
        "name": name, "kind": "rms_prologue", "regime": regime, "host": host,
        "dims": {"M": M, "N": n, "K": hidden},
        "gemm_only_ms": gemm_only_ms, "gemm_only_repeat_ms": gemm_repeat_s * 1e3,
        "gemm_drift_ratio": gemm_repeat_s * 1e3 / gemm_only_ms,
        "unfused_ms": unfused_ms, "unfused_compiled_nocg_ms": nocg_ms,
        "best_unfused_ms": best_unfused_ms,
        "addmm_ms": None, "addmm_reason": "N/A (prologue on the input; no GEMM epilogue)",
        "fused_paths": fused_paths, "best_fused_ms": best_fused_ms,
        "measured_gain": measured_gain, "measured_gain_verified": measured_gain_verified,
        "est_unfused_ms": est_unf,
        "est_fused_ms": est_fus2, "estimated_gain": est_unf / est_fus2,
        "est_fused_p1_ms": est_fus1, "estimated_gain_p1": est_unf / est_fus1,
        "est_p2_sumsq_kernel_ms": est_sumsq,
        "estimated_gain_p2_incl_sumsq": est_unf / (est_fus2 + est_sumsq),
        "est_note": "spec A.5 formula (F3-analog, VALID yardstick); baseline norm traffic "
                    "modeled as M*H*BPE + M*4 per the F3 assumption while the standalone "
                    "baseline physically writes normalized x (2*M*H*BPE) - stated deviation; "
                    "est_fused_ms omits P2's kernel-1 (see est_p2_sumsq_kernel_ms)",
        "triton_p2": p2, "triton_p1": p1,
        "fused_verified": r_def.get("fused", False),
        "fused_verified_forced": r_forced.get("fused", False),
        "kernel_evidence": r_def["kernels"], "fusion_evidence": r_def["evidence"],
        "nocg_kernel_evidence": r_nocg["kernels"], "nocg_fusion_evidence": r_nocg["evidence"],
        "forced_kernel_evidence": r_forced["kernels"],
        "forced_fusion_evidence": r_forced["evidence"],
        "numerics": {"eager_rel_max_vs_fp32": eager_rel, "rel_tol": rel_tol,
                     "compiled_ok": r_def.get("numerics_ok"),
                     "nocg_ok": r_nocg.get("numerics_ok"),
                     "forced_ok": r_forced.get("numerics_ok")},
        "compiled_cudagraph_input_copy_us": _cudagraph_copy_us(r_def["kernels"]),
        "compile_errors": {"compiled": r_def["error"], "nocg": r_nocg["error"],
                           "forced": r_forced["error"]},
        "clocks": clocks,
    }
    if gpu_adj is not None:
        a_unf, a_fus = est_rms_prologue_ms(M, n, hidden, count, gpu_adj, a_factor=1.0)
        row["est_unfused_ms_adj"], row["est_fused_ms_adj"] = a_unf, a_fus
        row["estimated_gain_adj"] = a_unf / a_fus
    del h, W, ref, ref32
    torch.cuda.empty_cache()
    print(f"    gemm={gemm_only_ms:.3f} unf={unfused_ms:.3f} nocg={nocg_ms} "
          f"p2={p2['ms']} p1={p1['ms']} forced={r_forced['ms']} gain={measured_gain} "
          f"gain_ver={measured_gain_verified} est_gain={est_unf / est_fus2:.4f} "
          f"drift={row['gemm_drift_ratio']:.3f}", flush=True)
    return row


# --------------------------------------------------------------------------- #
def epilogue_configs(smoke):
    if smoke:
        return [("rms_epi_smoke", 256, 512, 512, "decode")]
    cfgs = [(f"rms_epi_M{m}", m, HIDDEN, KV, "decode")
            for m in (512, 1024, 2048, 4096, 8192, 16384)]
    cfgs.append(("rms_epi_M32768", 32768, HIDDEN, KV, "prefill"))
    return cfgs


def prologue_configs(smoke):
    if smoke:
        return [("rms_pro_smoke_upgate", 64, 256, 512, "up_gate", "decode"),
                ("rms_pro_smoke_router", 256, 256, 512, "router", "decode")]
    cfgs = [(f"rms_pro_upgate_tpe{t}", t, TWOI, HIDDEN, "up_gate",
             "decode" if t <= 512 else "prefill")
            for t in (16, 32, 64, 128, 256, 512, 1024, 4096)]
    cfgs += [(f"rms_pro_router_M{m}", m, EXPERTS, HIDDEN, "router",
              "decode" if m <= 16384 else "prefill") for m in (2048, 32768)]
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--t2-json", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    iters, warmup = (3, 3) if args.smoke else (30, 15)
    gpu = GPUS["rtx4060-measured"]
    gpu_adj, adj_info = (None, None)
    if args.t2_json:
        gpu_adj, adj_info = build_adjusted_profile(args.t2_json, gpu)

    env = {
        "task": "T6.A RMSNorm epilogue (structurally infeasible) + prologue fallback",
        "torch": torch.__version__, "triton": triton.__version__ if _HAVE_TRITON else None,
        "device": torch.cuda.get_device_name(0),
        "clocks_locked": True,
        "clocks_note": "clocks LOCKED 1500/5501 by host (round 3); ClockSampler per config",
        "host_decisions": {
            "epilogue_prefill_cap": 32768,
            "epilogue_prefill_cap_reason": "tokens=131072 exceeds 8 GiB (A 4.0 GiB + out "
                                           "1.5 GiB + normalized copy); host chose cap over "
                                           "row-chunking",
            "epilogue_residual_included": True,
            "epilogue_residual_note": "host chose the layer-faithful variant y = "
                                      "RMSNorm(mla_o(x) + res, gamma); residual-add is "
                                      "tile-local (F1) - the cross-tile N-reduction stays "
                                      "the crux",
        },
        "smoke": args.smoke, "adjusted_profile": adj_info,
    }
    conventions = {
        "time_unit": "ms for *_ms fields (med_time/estimator return SECONDS, *1e3 here)",
        "gain": "unfused/fused, >1 => fusion faster; measured_gain_verified uses only "
                "profiler-verified + numerics-ok fused paths",
        "epilogue_est": "STRUCTURE-BLIND (est_invalid_structure_blind=true): "
                        "estimate_fused_gemm's Epilogue is tile-local only - excluded from "
                        "est-vs-measured aggregates (spec T6.0.4 guard)",
        "prologue_est": "F3-analog per spec A.5 (VALID yardstick); a_factor 1.0=P2 2.0=P1",
        "numerics": "rel_max vs fp32 <= max(2*eager_rel_max, 5e-2); epilogue checked on "
                    "first min(M,4096) rows (rowwise reduction -> slice-exact)",
        "timing": f"median of {iters} cuda events after {warmup} warmup + clock warmer",
    }

    results = []
    out = {"conventions": conventions, "env": env, "configs": results}

    def run_config(thunk, name, kind):
        try:
            row = thunk()
        except Exception as exc:
            traceback.print_exc()
            row = {"name": name, "kind": kind, "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)
        save_json(args.out, out)

    for name, m, hid, kv, regime in epilogue_configs(args.smoke):
        run_config(lambda: measure_rms_epilogue(name, m, hid, kv, regime, iters, warmup,
                                                gpu, gpu_adj, smoke=args.smoke),
                   name, "rms_epilogue")
    for name, m, n, hid, host, regime in prologue_configs(args.smoke):
        run_config(lambda: measure_rms_prologue(name, m, n, hid, host, regime, iters,
                                                warmup, gpu, gpu_adj, smoke=args.smoke),
                   name, "rms_prologue")

    ok = [r for r in results if "error" not in r]
    pro = [r for r in ok if r["kind"] == "rms_prologue" and r.get("measured_gain_verified")
           and abs(r["gemm_drift_ratio"] - 1) <= 0.05]
    print("\n=== T6.A summary ===")
    for r in ok:
        print(f"{r['name']:<26} {r['kind']:<14} gain={r.get('measured_gain')} "
              f"gain_ver={r.get('measured_gain_verified')} "
              f"est={r.get('estimated_gain'):.4f}"
              f"{' [EST INVALID structure-blind]' if r.get('est_invalid_structure_blind') else ''}")
    if pro:
        print(f"prologue drift-clean verified-gain geomean (n={len(pro)}): "
              f"{geomean([r['measured_gain_verified'] for r in pro]):.4f}  vs est "
              f"{geomean([r['estimated_gain'] for r in pro]):.4f}")
    n_epi_fused = sum(1 for r in ok if r["kind"] == "rms_epilogue"
                      and (r.get("fused_verified") or r.get("fused_verified_nocg")
                           or r.get("fused_verified_forced")))
    print(f"epilogue stock-path fusions verified: {n_epi_fused}/"
          f"{sum(1 for r in ok if r['kind'] == 'rms_epilogue')} "
          f"(expected 0: cross-tile N-reduction)")


if __name__ == "__main__":
    main()
