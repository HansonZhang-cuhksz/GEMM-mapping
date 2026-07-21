"""T2-peak: validate the `rtx4060-measured` estimator profile against real hardware.

Measures, on the RTX 4060 Laptop GPU (bf16, cuda:0):
  1. Peak BF16 GEMM TFLOP/s over square matmuls n in {2048,4096,6144,8192}.
  2. HBM bandwidth two ways: device-to-device bf16 copy (read+write) and a
     read-heavy reduction (torch.sum).
  3. A comparison block vs GPUS['rtx4060-measured'] (18.4 TFLOP/s / 170 GB/s),
     including the sustained tensor clock implied by the measured peak.
  4. An `adjusted_profile` dict (clock_hz / bw_bytes_per_s overrides) that
     downstream scripts consume via --t2-json to re-calibrate the estimator.

Clocks are UNLOCKED on this box (WSL2, no root), so every measurement group is
wrapped in a ClockSampler and its .summary() is recorded next to the number.

Run from /home/shuhan/snowcat-demo/GEMM-mapping:
    python rtx4060_peak.py --out rtx4060_measured.json
    python rtx4060_peak.py --smoke --out /tmp/.../peak_smoke.json

All reported times are MILLISECONDS (fields end _ms); med_time returns SECONDS.
"""
import argparse
import dataclasses
import os
import sys

import torch

# The estimator + shared timing module are co-located top-level modules in the
# repo dir; make the import work regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemm_time_estimator import GPUS
from rtx4060_common import (
    ClockSampler,
    bf,
    med_time,
    save_json,
    sleep_cooldown,
)

GIB = 1 << 30


# --------------------------------------------------------------------------- #
# GPU measurement (each group wrapped in a ClockSampler; times SECONDS -> ms)  #
# --------------------------------------------------------------------------- #
def measure_peak_gemm(ns, iters, warmup, cooldown_s):
    """Square bf16 matmuls a@b for each n; report per-shape ms + TFLOP/s."""
    shapes = []
    for n in ns:
        a = bf(n, n)
        b = bf(n, n)
        with ClockSampler() as cs:
            t_s = med_time(lambda: a @ b, iters=iters, warmup=warmup)
        tflops = 2.0 * n ** 3 / t_s / 1e12
        shapes.append({
            "n": n,
            "time_ms": t_s * 1e3,
            "tflops": tflops,
            "clocks": cs.summary(),
        })
        del a, b
        torch.cuda.empty_cache()
        sleep_cooldown(cooldown_s)
    measured_peak_tflops = max(s["tflops"] for s in shapes)
    return {"shapes": shapes, "measured_peak_tflops": measured_peak_tflops}


def measure_bandwidth(nbytes, iters, warmup, cooldown_s):
    """(a) d2d bf16 copy (read+write) and (b) read-heavy sum, on one ~nbytes tensor."""
    numel = nbytes // 2  # bf16 = 2 bytes/elem
    src = bf(numel)
    dst = torch.empty_like(src)

    # (a) device-to-device copy: reads src + writes dst -> 2 * numel * 2 bytes.
    with ClockSampler() as cs_copy:
        t_copy = med_time(lambda: dst.copy_(src), iters=iters, warmup=warmup)
    copy_bytes = 2 * numel * 2
    copy_gbs = copy_bytes / t_copy / 1e9
    sleep_cooldown(cooldown_s)

    # (b) read-heavy reduction: reads src once -> numel * 2 bytes.
    with ClockSampler() as cs_sum:
        t_sum = med_time(lambda: torch.sum(src), iters=iters, warmup=warmup)
    sum_bytes = numel * 2
    sum_gbs = sum_bytes / t_sum / 1e9

    del src, dst
    torch.cuda.empty_cache()

    measured_bw_gbs = max(copy_gbs, sum_gbs)
    return {
        "tensor_bytes": nbytes,
        "tensor_numel": numel,
        "d2d_copy": {
            "time_ms": t_copy * 1e3,
            "bytes": copy_bytes,
            "gbs": copy_gbs,
            "clocks": cs_copy.summary(),
        },
        "read_sum": {
            "time_ms": t_sum * 1e3,
            "bytes": sum_bytes,
            "gbs": sum_gbs,
            "clocks": cs_sum.summary(),
        },
        "measured_bw_gbs": measured_bw_gbs,
    }


# --------------------------------------------------------------------------- #
# Estimator-only (CPU) derivations: compare to profile, emit adjusted profile  #
# --------------------------------------------------------------------------- #
def build_comparison(measured_peak_tflops, measured_bw_gbs, gpu):
    profile_peak_tflops = gpu.peak_tensor_flops / 1e12          # 18.432
    profile_bw_gbs = gpu.bw_bytes_per_s / 1e9                   # 170.0
    tensor_cores = gpu.tensor_cores                             # 96
    flops_per_clock = gpu.tensor_flops_per_core_per_clock       # 128.0
    profile_tensor_clock_mhz = gpu.clock_hz / 1e6              # 1500.0

    implied_tensor_clock_mhz = (
        measured_peak_tflops * 1e12 / (tensor_cores * flops_per_clock) / 1e6
    )
    return {
        "profile_name": gpu.name,
        "profile_peak_tflops": profile_peak_tflops,
        "profile_bw_gbs": profile_bw_gbs,
        "measured_peak_tflops": measured_peak_tflops,
        "measured_bw_gbs": measured_bw_gbs,
        "tflops_ratio_measured_over_profile": measured_peak_tflops / profile_peak_tflops,
        "bw_ratio_measured_over_profile": measured_bw_gbs / profile_bw_gbs,
        "tensor_cores": tensor_cores,
        "flops_per_core_per_clock": flops_per_clock,
        "profile_tensor_clock_mhz": profile_tensor_clock_mhz,
        "implied_tensor_clock_mhz": implied_tensor_clock_mhz,
        "implied_over_profile_clock": implied_tensor_clock_mhz / profile_tensor_clock_mhz,
    }


def build_adjusted_profile(measured_peak_tflops, measured_bw_gbs, gpu):
    """dataclasses.replace() overrides implied by the measurements.

    Downstream re-calibration is: replace(gpu, clock_hz=..., bw_bytes_per_s=...).
    """
    clock_hz = measured_peak_tflops * 1e12 / (
        gpu.tensor_cores * gpu.tensor_flops_per_core_per_clock
    )
    bw_bytes_per_s = measured_bw_gbs * 1e9
    # Sanity-check the override reproduces the measured peak through the model.
    adjusted = dataclasses.replace(gpu, clock_hz=clock_hz, bw_bytes_per_s=bw_bytes_per_s)
    return {
        "clock_hz": clock_hz,
        "bw_bytes_per_s": bw_bytes_per_s,
        "check_peak_tflops": adjusted.peak_tensor_flops / 1e12,
        "check_bw_gbs": adjusted.bw_bytes_per_s / 1e9,
    }


def device_info():
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "l2_cache_bytes": p.L2_cache_size,
        "smem_per_block_optin_bytes": p.shared_memory_per_block_optin,
        "smem_per_sm_bytes": p.shared_memory_per_multiprocessor,
        "total_memory_bytes": p.total_memory,
        "total_memory_gib": p.total_memory / GIB,
    }


def versions():
    v = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "device_name": torch.cuda.get_device_name(0),
    }
    try:
        import triton
        v["triton"] = triton.__version__
    except Exception:
        v["triton"] = None
    return v


def main():
    ap = argparse.ArgumentParser(description="T2-peak: RTX 4060 peak-spec validation.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny dims, iters=3/warmup=3, exercises the whole path fast.")
    args = ap.parse_args()

    # Fail fast on an unwritable --out BEFORE spending GPU time (save_json only
    # runs after all measurements; a typo'd path would discard the whole run).
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available.")

    gpu = GPUS["rtx4060-measured"]

    if args.smoke:
        ns = [128, 256]
        gemm_iters, gemm_warmup = 3, 3
        bw_iters, bw_warmup = 3, 3
        bw_bytes = 16 << 20      # 16 MiB
        cooldown_s = 0.1
    else:
        ns = [2048, 4096, 6144, 8192]
        gemm_iters, gemm_warmup = 30, 15
        bw_iters, bw_warmup = 50, 15
        bw_bytes = int(1.5 * GIB)  # ~1.5 GiB (src) + 1.5 GiB (dst) = ~3 GiB peak
        cooldown_s = 2.0

    # --- GPU measurement ---------------------------------------------------- #
    peak = measure_peak_gemm(ns, gemm_iters, gemm_warmup, cooldown_s)
    sleep_cooldown(cooldown_s)
    bw = measure_bandwidth(bw_bytes, bw_iters, bw_warmup, cooldown_s)

    # --- estimator-only (CPU) derivations ----------------------------------- #
    comparison = build_comparison(
        peak["measured_peak_tflops"], bw["measured_bw_gbs"], gpu)
    adjusted_profile = build_adjusted_profile(
        peak["measured_peak_tflops"], bw["measured_bw_gbs"], gpu)

    out = {
        "task": "T2-peak (peak-spec validation of rtx4060-measured profile)",
        "smoke": args.smoke,
        "conventions": {
            "time_fields": "all *_ms fields are MILLISECONDS (med_time returns seconds; *1e3)",
            "tflops": "2 * n^3 / time_s / 1e12 for a square n GEMM (bf16)",
            "gbs": "bytes / time_s / 1e9; copy counts read+write (2x), sum counts read only",
            "gain_convention": "gains/speedups = unfused_time/fused_time (>1.0 => fusion faster); "
                               "not used in this T2 script but documented for the study",
            "clocks": "unlocked on this WSL2 box; per-group nvidia-smi ClockSampler summaries recorded",
        },
        "versions": versions(),
        "device": device_info(),
        "clocks_locked": False,
        "peak_gemm": peak,
        "hbm_bandwidth": bw,
        "comparison": comparison,
        "adjusted_profile": adjusted_profile,
    }

    save_json(args.out, out)

    # Human-readable one-liners for the log.
    print(f"[peak] measured_peak_tflops = {peak['measured_peak_tflops']:.2f} "
          f"(profile {comparison['profile_peak_tflops']:.2f}, "
          f"ratio {comparison['tflops_ratio_measured_over_profile']:.3f})")
    print(f"[bw]   measured_bw_gbs = {bw['measured_bw_gbs']:.1f} "
          f"(profile {comparison['profile_bw_gbs']:.1f}, "
          f"ratio {comparison['bw_ratio_measured_over_profile']:.3f})")
    print(f"[clk]  implied tensor clock = {comparison['implied_tensor_clock_mhz']:.0f} MHz "
          f"(profile {comparison['profile_tensor_clock_mhz']:.0f} MHz)")
    print(f"[adj]  adjusted_profile: clock_hz={adjusted_profile['clock_hz']:.4g}, "
          f"bw_bytes_per_s={adjusted_profile['bw_bytes_per_s']:.4g}")


if __name__ == "__main__":
    main()
