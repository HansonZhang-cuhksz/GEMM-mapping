"""T3 - Single-GEMM estimator validation on the RTX 4060 (est vs measured).

Measures real bf16 GEMM times (a @ b, row-major A[M,K], B[K,N], OUT[M,N]) for a spread of
shapes and compares each to the snowcat-roofline estimator's optimal_mapping_by_time()
prediction using GPUS['rtx4060-measured'] (the profile calibrated on this machine).

Two clean halves, kept separate:
  * MEASURE  (GPU): med_time(lambda: a @ b) under a ClockSampler; clocks are UNLOCKED on
    this box (WSL2, no root) so the sampler summary is recorded per-sweep.
  * ESTIMATE (CPU): optimal_mapping_by_time(m, n, k, gpu).time_s -- no kernel launched.

All reported times are MILLISECONDS (med_time returns SECONDS; we multiply by 1e3).
ratio_est_over_meas = est_ms / meas_ms  (<1.0 => estimator predicts FASTER than reality,
i.e. optimistic). Prior 4060 calibration expectation: est/meas geomean in the 0.72-0.96 band.

Run from /home/shuhan/snowcat-demo/GEMM-mapping:
    python rtx4060_gemm_validate.py --out rtx4060_gemm_validate.json
    python rtx4060_gemm_validate.py --out ... --t2-json rtx4060_measured.json   # + adjusted profile
    python rtx4060_gemm_validate.py --smoke --out /tmp/.../validate_smoke.json    # tiny end-to-end
"""
import argparse
import dataclasses
import json
import os
import sys

# Make the repo-root modules (gemm_time_estimator + the snowcat_demo package it imports,
# and rtx4060_common) importable regardless of the caller's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from gemm_time_estimator import GPUS, optimal_mapping_by_time  # noqa: E402
from rtx4060_common import (  # noqa: E402
    ClockSampler,
    bf,
    geomean,
    med_time,
    save_json,
    sleep_cooldown,
)

GIB = float(1 << 30)
BYTES_BUDGET = 6.5 * GIB  # keep peak allocation comfortably under 8 GiB VRAM
FULL_ITERS = 40
FULL_WARMUP = 15
SMOKE_ITERS = 3
SMOKE_WARMUP = 3


# --------------------------------------------------------------------------- #
# Shape list                                                                   #
# --------------------------------------------------------------------------- #
def build_shapes(smoke=False):
    """Return an ordered list of {m, n, k, tags:set, is_ffn} dicts, deduped by (m, n, k).

    tags label provenance (square / tall_wide / up_gate / down / mla_o). A shape can carry
    several tags (e.g. a square that is also an FFN 'down' stage). is_ffn is True if any
    FFN-stage tag applies -- used for the FFN-restricted summary.
    """
    order = []  # (m, n, k)
    tagmap = {}

    def add(m, n, k, tag):
        key = (m, n, k)
        if key not in tagmap:
            tagmap[key] = set()
            order.append(key)
        tagmap[key].add(tag)

    if smoke:
        # Tiny shapes that still exercise squares / tall_wide / all three FFN stages,
        # dedup, the estimator, ratios, and both (all + ffn) summaries.
        for n in (128, 256):
            add(n, n, n, "square")
        add(512, 128, 256, "tall_wide")
        for M in (128, 256):
            for h in (64, 128):
                add(M, 2 * h, h, "up_gate")   # up_gate: [M, h] @ [h, 2h]
                add(M, h, h, "down")          # down:    [M, h] @ [h, h] (inter = hidden)
            add(M, 512, 384, "mla_o")         # scaled mla_o stand-in
    else:
        # squares
        for n in (1024, 1536, 2048, 3072, 4096, 6144):
            add(n, n, n, "square")
        # tall / wide
        for (m, n, k) in (
            (8192, 1024, 4096),
            (1024, 8192, 4096),
            (4096, 1024, 8192),
            (16384, 2048, 1024),
            (2048, 16384, 1024),
        ):
            add(m, n, k, "tall_wide")
        # FFN-stage shapes (T4 regime) so T3 covers it
        for M in (2048, 8192):
            for h in (1024, 2048, 4096):
                add(M, 2 * h, h, "up_gate")   # up_gate: [M, h] @ [h, 2*inter]
                add(M, h, h, "down")          # down:    [M, inter] @ [inter, h], inter = h
            add(M, 16384, 6144, "mla_o")      # mla_o:   [M, 6144] @ [6144, 16384]

    ffn_tags = {"up_gate", "down", "mla_o"}
    shapes = []
    for (m, n, k) in order:
        tags = tagmap[(m, n, k)]
        nbytes = (m * k + k * n + m * n) * 2  # bf16 a, b, c
        shapes.append({
            "m": m, "n": n, "k": k,
            "tags": sorted(tags),
            "is_ffn": bool(tags & ffn_tags),
            "bytes_gib": nbytes / GIB,
        })
    return shapes


# --------------------------------------------------------------------------- #
# Estimator side (CPU only -- no GPU work)                                     #
# --------------------------------------------------------------------------- #
def estimate_ms(m, n, k, gpu):
    """Estimator-optimal predicted time in ms, plus a compact mapping label."""
    mp, e = optimal_mapping_by_time(m, n, k, gpu)
    label = f"{mp.bm}x{mp.bn}x{mp.bk} s{e.num_stages} {''.join(mp.loop_order)}"
    return e.time_s * 1e3, label


def load_adjusted_profile(path, base_gpu):
    """Build an adjusted GpuModel from a T2 json's 'adjusted_profile' via dataclasses.replace.

    Expects t2['adjusted_profile'] to carry clock_hz and/or bw_bytes_per_s. Returns
    (adjusted_gpu_or_None, info_dict). Missing fields are simply not overridden; if the
    key is absent entirely, returns (None, {}).
    """
    with open(path) as f:
        t2 = json.load(f)
    ap = t2.get("adjusted_profile")
    if not isinstance(ap, dict):
        return None, {"note": "no 'adjusted_profile' dict in t2 json"}
    kw = {}
    if ap.get("clock_hz") is not None:
        kw["clock_hz"] = float(ap["clock_hz"])
    if ap.get("bw_bytes_per_s") is not None:
        kw["bw_bytes_per_s"] = float(ap["bw_bytes_per_s"])
    if not kw:
        return None, {"note": "adjusted_profile lacked clock_hz / bw_bytes_per_s", "raw": ap}
    adj = dataclasses.replace(base_gpu, **kw)
    return adj, {"applied": kw, "source": path}


# --------------------------------------------------------------------------- #
# Measurement side (GPU)                                                        #
# --------------------------------------------------------------------------- #
def measure_ms(m, n, k, iters, warmup):
    """Measured bf16 a@b time in ms + achieved TFLOP/s. Inputs allocated once, reused."""
    a = bf(m, k)
    b = bf(k, n)
    t_s = med_time(lambda: a @ b, iters=iters, warmup=warmup)
    tflops = 2.0 * m * n * k / t_s / 1e12
    del a, b
    torch.cuda.empty_cache()
    return t_s * 1e3, tflops


# --------------------------------------------------------------------------- #
# Summary                                                                       #
# --------------------------------------------------------------------------- #
def band_stats(ratios):
    if not ratios:
        return None
    n = len(ratios)
    w15 = sum(1 for r in ratios if (1.0 / 1.5) <= r <= 1.5)
    w2 = sum(1 for r in ratios if 0.5 <= r <= 2.0)
    return {
        "n": n,
        "geomean": geomean(ratios),
        "pct_within_1.5x": 100.0 * w15 / n,
        "pct_within_2x": 100.0 * w2 / n,
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
    }


def build_summary(rows, key):
    """key = 'ratio_est_over_meas_stock' or '..._adjusted'. Returns {all, ffn}."""
    all_r = [r[key] for r in rows if r.get(key) is not None]
    ffn_r = [r[key] for r in rows if r["is_ffn"] and r.get(key) is not None]
    return {"all": band_stats(all_r), "ffn": band_stats(ffn_r)}


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="T3 single-GEMM estimator validation (RTX 4060)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--smoke", action="store_true", help="tiny dims, iters=3 warmup=3")
    ap.add_argument("--t2-json", default=None,
                    help="T2 json with adjusted_profile -> also report est_ms_adjusted")
    args = ap.parse_args()

    iters = SMOKE_ITERS if args.smoke else FULL_ITERS
    warmup = SMOKE_WARMUP if args.smoke else FULL_WARMUP

    gpu = GPUS["rtx4060-measured"]
    gpu_adj, adj_info = (None, None)
    if args.t2_json:
        gpu_adj, adj_info = load_adjusted_profile(args.t2_json, gpu)
        if gpu_adj is None:
            print(f"[warn] --t2-json given but no usable adjusted_profile: {adj_info}")

    shapes = build_shapes(smoke=args.smoke)

    # Budget guard: every listed shape must fit comfortably.
    for s in shapes:
        if s["bytes_gib"] * GIB >= BYTES_BUDGET:
            raise SystemExit(f"shape {s['m']}x{s['n']}x{s['k']} "
                             f"needs {s['bytes_gib']:.2f} GiB >= 6.5 GiB budget")

    def profile_dict(g):
        return None if g is None else {
            "name": g.name,
            "clock_hz": g.clock_hz,
            "bw_bytes_per_s": g.bw_bytes_per_s,
            "peak_tensor_tflops": g.peak_tensor_flops / 1e12,
        }

    rows = []

    # --- ESTIMATOR (CPU, no GPU) ------------------------------------------- #
    for s in shapes:
        m, n, k = s["m"], s["n"], s["k"]
        try:
            est_stock_ms, mapping_stock = estimate_ms(m, n, k, gpu)
        except Exception as exc:  # pragma: no cover - shapes chosen to resolve
            est_stock_ms, mapping_stock = None, f"estimator error: {exc}"
        row = dict(s)
        row["est_ms_stock"] = est_stock_ms
        row["mapping_stock"] = mapping_stock
        if gpu_adj is not None:
            try:
                row["est_ms_adjusted"], row["mapping_adjusted"] = estimate_ms(m, n, k, gpu_adj)
            except Exception as exc:  # pragma: no cover
                row["est_ms_adjusted"], row["mapping_adjusted"] = None, f"estimator error: {exc}"
        rows.append(row)

    # --- MEASUREMENT (GPU, wrapped in a per-sweep ClockSampler) ------------ #
    print(f"[measure] {len(rows)} shapes, iters={iters} warmup={warmup} "
          f"(smoke={args.smoke})", flush=True)
    try:
        with ClockSampler() as clk:
            for i, row in enumerate(rows):
                m, n, k = row["m"], row["n"], row["k"]
                try:
                    meas_ms, tflops = measure_ms(m, n, k, iters, warmup)
                except Exception as exc:
                    # One shape's failure (OOM / driver hiccup) must not kill the sweep.
                    torch.cuda.empty_cache()
                    row["measure_error"] = f"{type(exc).__name__}: {exc}"
                    row["meas_ms"] = None
                    row["ratio_est_over_meas_stock"] = None
                    if gpu_adj is not None:
                        row["ratio_est_over_meas_adjusted"] = None
                    print(f"  [{i + 1:>2}/{len(rows)}] {m}x{n}x{k:<6} "
                          f"MEASURE ERROR: {row['measure_error']}", flush=True)
                    sleep_cooldown()
                    continue
                row["meas_ms"] = meas_ms
                row["achieved_tflops"] = tflops
                # ratios (est / meas): <1 means estimator optimistic
                row["ratio_est_over_meas_stock"] = (
                    row["est_ms_stock"] / meas_ms if row["est_ms_stock"] else None
                )
                if gpu_adj is not None:
                    ea = row.get("est_ms_adjusted")
                    row["ratio_est_over_meas_adjusted"] = (ea / meas_ms) if ea else None
                est_s = (f"{row['est_ms_stock']:8.4f}ms"
                         if row["est_ms_stock"] is not None else "   None ")
                ratio_s = (f"{row['ratio_est_over_meas_stock']:.3f}"
                           if row["ratio_est_over_meas_stock"] is not None else "None")
                print(f"  [{i + 1:>2}/{len(rows)}] {m}x{n}x{k:<6} "
                      f"meas={meas_ms:8.4f}ms est={est_s} ratio={ratio_s} "
                      f"({row['achieved_tflops']:.1f} TFLOP/s) tags={row['tags']}", flush=True)
                sleep_cooldown()  # bleed off thermal state between measurement groups
    except BaseException:
        # Hard failure (device lost, Ctrl-C): salvage everything measured so far.
        save_json(args.out + ".partial", {
            "task": "T3-validate",
            "partial": True,
            "params": {"iters": iters, "warmup": warmup, "smoke": args.smoke},
            "shapes": rows,
        })
        raise
    clocks = clk.summary()

    # --- SUMMARY ----------------------------------------------------------- #
    summary = {
        "stock": build_summary(rows, "ratio_est_over_meas_stock"),
        "prior_calibration_band": "est/meas geomean expected in 0.72-0.96 (locked-clock calibration)",
    }
    if gpu_adj is not None:
        summary["adjusted"] = build_summary(rows, "ratio_est_over_meas_adjusted")

    out = {
        "task": "T3-validate",
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "params": {"iters": iters, "warmup": warmup, "smoke": args.smoke},
        "conventions": {
            "time_units": "all *_ms fields are MILLISECONDS (med_time returns seconds; *1e3)",
            "ratio_est_over_meas": "est_ms / meas_ms; <1.0 => estimator predicts FASTER than measured (optimistic)",
            "achieved_tflops": "2*m*n*k / measured_seconds / 1e12 (2 FLOP per multiply-add)",
            "gemm_layout": "row-major a[M,K] @ b[K,N] -> c[M,N], bf16 inputs",
            "gain_convention": "study-wide: gain = unfused_time / fused_time, >1.0 => fusion FASTER (not used in T3)",
            "clocks": "UNLOCKED on this WSL2 box (no root); ClockSampler summary recorded per-sweep",
        },
        "profile_stock": profile_dict(gpu),
        "profile_adjusted": profile_dict(gpu_adj),
        "adjusted_profile_source": adj_info,
        "clocks": clocks,
        "shapes": rows,
        "summary": summary,
    }
    save_json(args.out, out)

    # --- Console comparison against the prior calibration band -------------- #
    st = summary["stock"]["all"]
    print("\n=== T3 estimator validation summary ===")
    if st is None:
        print("[stock all ] NO valid (est, meas) pairs -- see per-shape errors in the JSON")
        return
    print(f"[stock all ] n={st['n']} geomean(est/meas)={st['geomean']:.3f}  "
          f"within1.5x={st['pct_within_1.5x']:.0f}%  within2x={st['pct_within_2x']:.0f}%  "
          f"min={st['min_ratio']:.3f} max={st['max_ratio']:.3f}")
    sf = summary["stock"]["ffn"]
    if sf:
        print(f"[stock ffn ] n={sf['n']} geomean(est/meas)={sf['geomean']:.3f}  "
              f"within1.5x={sf['pct_within_1.5x']:.0f}%  within2x={sf['pct_within_2x']:.0f}%  "
              f"min={sf['min_ratio']:.3f} max={sf['max_ratio']:.3f}")
    verdict = "IN BAND" if 0.72 <= st["geomean"] <= 0.96 else "OUT OF BAND"
    print(f"[calibration] stock geomean {st['geomean']:.3f} vs prior 0.72-0.96 -> {verdict}")
    if gpu_adj is not None and summary["adjusted"]["all"]:
        sa = summary["adjusted"]["all"]
        print(f"[adjusted all] n={sa['n']} geomean(est/meas)={sa['geomean']:.3f}  "
              f"within1.5x={sa['pct_within_1.5x']:.0f}%  within2x={sa['pct_within_2x']:.0f}%")


if __name__ == "__main__":
    main()
