#!/usr/bin/env python3
"""Calibrate the estimator's L2 effective-capacity `alpha` against Nsight Compute.

Pipeline:
  1. Run `ncu` on the CUTLASS `gemm` binary for one shape + mapping (or read a
     previously-captured ncu CSV), extracting the *measured* DRAM bytes and L2
     sector hit rate for the main GEMM kernel.
  2. Ask `gemm_time_estimator` for the *modeled* DRAM traffic as a function of
     `alpha` (C_eff = alpha * L2), and pick the alpha that matches the measurement.
  3. Report measured-vs-model, the fitted alpha, and the measured L2 hit rate vs
     the model-implied hit rate.  Append the row to `calibration_runs.csv` so
     several shapes can be pooled into one global alpha.

Counter access needs permission (root, or NVreg_RestrictProfilingToAdminUsers=0;
on WSL2 a recent driver + admin).  If blocked you'll get ERR_NVGPUCTRPERM -- use
`--print-cmd` to get the exact command to run yourself (e.g. `! sudo ...`), then
feed the saved CSV back with `--from-csv`.

Examples
  # 1) print the ncu command to run with elevated permissions:
  python l2_calibrate.py --m 128 --n 4096 --k 6144 --dtype fp16 \
      --config tb128x128x32_splitK3 --bm 128 --bn 128 --bk 32 --order MNK --splitk 3 \
      --print-cmd

  # 2) after capturing:  sudo ncu ... --csv --log-file run.csv ./gemm ...
  python l2_calibrate.py --m 128 --n 4096 --k 6144 --bm 128 --bn 128 --bk 32 \
      --order MNK --splitk 3 --from-csv run.csv

  # 3) live (if counters are permitted):
  python l2_calibrate.py --m 128 --n 4096 --k 6144 --dtype fp16 \
      --config tb128x128x32_splitK3 --bm 128 --bn 128 --bk 32 --order MNK --splitk 3
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNOWCAT = os.environ.get("SNOWCAT_DIR", "/home/shuhan/snowcat-demo")
GEMM_BIN = HERE / "gemm"
CALIB_CSV = HERE / "calibration_runs.csv"

# Metrics: DRAM read/write bytes (ground-truth traffic) + L2 sector hit rate.
METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors.sum",
    "lts__t_sector_hit_rate.pct",
    "gpu__time_duration.sum",
]

# ncu reports auto-scaled *decimal* units; convert everything to base (bytes / %).
_UNIT_SCALE = {
    "": 1.0, "%": 1.0,
    "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
    "nsecond": 1.0, "usecond": 1e3, "msecond": 1e6, "second": 1e9,
}


# --------------------------------------------------------------------------- #
# Estimator import (for the model side of the fit)                             #
# --------------------------------------------------------------------------- #
def _load_estimator():
    # SNOWCAT provides the snowcat_demo package but also holds a *stale* copy of
    # gemm_time_estimator.py; append it (low priority) and put HERE first so the
    # local, up-to-date estimator wins.
    if SNOWCAT not in sys.path:
        sys.path.append(SNOWCAT)
    sys.path.insert(0, str(HERE))
    try:
        import gemm_time_estimator as est
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise SystemExit(
            f"cannot import gemm_time_estimator/snowcat_demo ({exc}); set SNOWCAT_DIR "
            f"(currently {SNOWCAT!r}) to the snowcat-demo checkout."
        )
    return est


# --------------------------------------------------------------------------- #
# ncu invocation + CSV parsing                                                 #
# --------------------------------------------------------------------------- #
def build_ncu_cmd(args) -> list[str]:
    cmd = [
        args.ncu, "--csv", "--target-processes", "all",
        "--kernel-name-base", "demangled",
        "--launch-count", str(args.launch_count),
        "--metrics", ",".join(METRICS),
    ]
    if args.launch_skip:
        cmd += ["--launch-skip", str(args.launch_skip)]
    if args.kernel_regex:
        cmd += ["--kernel-name", f"regex:{args.kernel_regex}"]
    cmd += [
        str(GEMM_BIN), "--m", str(args.m), "--n", str(args.n), "--k", str(args.k),
        "--dtype", args.dtype, "--iters", str(args.iters), "--warmup", str(args.warmup),
    ]
    if args.config:
        cmd += ["--config", args.config]
    return cmd


def run_ncu(args) -> str:
    """Run ncu, return its CSV stdout. Raises SystemExit with guidance on failure."""
    cmd = build_ncu_cmd(args)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if "ERR_NVGPUCTRPERM" in proc.stderr or "ERR_NVGPUCTRPERM" in proc.stdout:
        raise SystemExit(
            "ERR_NVGPUCTRPERM: no permission to read GPU performance counters.\n"
            "  Fixes: run under `sudo`, or set NVreg_RestrictProfilingToAdminUsers=0\n"
            "  (echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \\\n"
            "     sudo tee /etc/modprobe.d/nvidia-profiler.conf, then reboot).\n"
            "  On WSL2: use a recent NVIDIA Windows driver and an elevated shell.\n"
            "  Meanwhile: rerun with --print-cmd, capture with `sudo ncu ... "
            "--log-file run.csv`, then use --from-csv run.csv."
        )
    if proc.returncode != 0 and "==PROF==" not in proc.stderr:
        raise SystemExit(f"ncu failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    return proc.stdout


def parse_ncu_csv(text: str) -> list[dict]:
    """Parse ncu --csv (long format) into per-kernel dicts of metric->base value."""
    # The CSV header is the first line containing both "Metric Name" and "Metric Value".
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if "Metric Name" in ln and "Metric Value" in ln), None)
    if start is None:
        raise SystemExit("could not find a metric table in the ncu CSV output.")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    kernels: dict[str, dict] = {}
    order: list[str] = []
    for row in reader:
        kid = row.get("ID") or row.get("Kernel Name", "")
        name = row.get("Kernel Name", "")
        metric = row.get("Metric Name", "")
        unit = (row.get("Metric Unit") or "").strip()
        raw = (row.get("Metric Value") or "").strip().replace(",", "")
        if not metric or raw == "":
            continue
        try:
            val = float(raw) * _UNIT_SCALE.get(unit, 1.0)
        except ValueError:
            continue
        if kid not in kernels:
            kernels[kid] = {"_name": name}
            order.append(kid)
        kernels[kid][metric] = val
    return [kernels[k] for k in order]


def pick_kernel(kernels: list[dict], regex: str | None) -> dict:
    """Pick the target GEMM kernel: regex match if given, else the longest-running."""
    import re
    cands = kernels
    if regex:
        rx = re.compile(regex)
        cands = [k for k in kernels if rx.search(k.get("_name", ""))] or kernels
    return max(cands, key=lambda k: k.get("gpu__time_duration.sum", 0.0))


# --------------------------------------------------------------------------- #
# Fit                                                                          #
# --------------------------------------------------------------------------- #
def fit_alpha(est, args, measured_dram_bytes: float) -> tuple[float, float, float]:
    """Sweep alpha, return (best_alpha, model_bytes_at_best, snowcat_miss_bytes)."""
    mapping = est.Mapping(
        bm=args.bm, bn=args.bn, bk=args.bk,
        loop_order=est.parse_loop_order(args.order),
        num_stages=args.stages, split_k=args.splitk,
    )
    gpu = est.RTX4060_LAPTOP

    def model_bytes(alpha):
        e = est.estimate_gemm_time(args.m, args.n, args.k, mapping, gpu,
                                   l2=True, l2_alpha=alpha)
        return float(e.traffic_bytes)

    # snowcat "L2 always miss" upper bound (alpha-independent), for the hit-rate cross-check.
    miss = est.estimate_gemm_time(args.m, args.n, args.k, mapping, gpu, l2=False)
    snowcat_miss_bytes = float(miss.traffic_bytes)

    best_a, best_err, best_bytes = 1.0, float("inf"), model_bytes(1.0)
    a = 0.02
    while a <= 1.0001:
        mb = model_bytes(a)
        err = abs(mb - measured_dram_bytes)
        if err < best_err:
            best_a, best_err, best_bytes = a, err, mb
        a += 0.02
    return best_a, best_bytes, snowcat_miss_bytes


def append_calib_row(row: dict) -> None:
    new = not CALIB_CSV.exists()
    with CALIB_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # problem + mapping (model side)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--bm", type=int, required=True)
    p.add_argument("--bn", type=int, required=True)
    p.add_argument("--bk", type=int, required=True)
    p.add_argument("--order", default="MNK")
    p.add_argument("--stages", type=int, default=None)
    p.add_argument("--splitk", type=int, default=1)
    # ncu side
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--config", default=None, help="gemm --config filter (which mapping)")
    p.add_argument("--kernel-regex", default="cutlass|gemm|Kernel",
                   help="regex to pick the profiled kernel (default matches CUTLASS)")
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--launch-count", type=int, default=1)
    p.add_argument("--launch-skip", type=int, default=0)
    p.add_argument("--ncu", default="ncu")
    # modes
    p.add_argument("--print-cmd", action="store_true",
                   help="print the ncu command (prefix with sudo if needed) and exit")
    p.add_argument("--from-csv", default=None,
                   help="parse this saved ncu --csv file instead of running ncu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    est = _load_estimator()

    if args.print_cmd:
        print(" ".join(build_ncu_cmd(args)))
        print("\n# If ERR_NVGPUCTRPERM: prepend 'sudo', or add "
              "'--log-file run.csv' and reuse via --from-csv run.csv", file=sys.stderr)
        return

    if args.from_csv:
        csv_text = Path(args.from_csv).read_text()
    else:
        csv_text = run_ncu(args)

    kernels = parse_ncu_csv(csv_text)
    if not kernels:
        raise SystemExit("no kernels parsed from ncu output.")
    kern = pick_kernel(kernels, args.kernel_regex)

    read_b = kern.get("dram__bytes_read.sum", 0.0)
    write_b = kern.get("dram__bytes_write.sum", 0.0)
    meas_total = read_b + write_b
    hit_pct = kern.get("lts__t_sector_hit_rate.pct", float("nan"))
    dur_ns = kern.get("gpu__time_duration.sum", float("nan"))

    best_a, model_b, snowcat_miss = fit_alpha(est, args, meas_total)
    implied_hit = (1.0 - model_b / snowcat_miss) * 100 if snowcat_miss > 0 else float("nan")

    mib = 2 ** 20
    print(f"=== L2 calibration  GEMM {args.m}x{args.n}x{args.k}  "
          f"tile {args.bm}x{args.bn}x{args.bk} {args.order} splitK={args.splitk} ===")
    print(f"  kernel                : {kern.get('_name','?')[:70]}")
    print(f"  measured DRAM read    : {read_b / mib:8.2f} MiB")
    print(f"  measured DRAM write   : {write_b / mib:8.2f} MiB")
    print(f"  measured DRAM total   : {meas_total / mib:8.2f} MiB   "
          f"(kernel {dur_ns/1e6:.3f} ms)")
    print(f"  measured L2 hit rate  : {hit_pct:8.2f} %")
    print(f"  snowcat L2-miss T     : {snowcat_miss / mib:8.2f} MiB  (alpha->0 upper bound)")
    print(f"  --> fitted alpha      : {best_a:8.2f}   "
          f"(model DRAM {model_b / mib:.2f} MiB vs measured {meas_total / mib:.2f} MiB)")
    print(f"  model-implied L2 hit  : {implied_hit:8.2f} %   "
          f"(cross-check vs measured {hit_pct:.1f} %)")

    append_calib_row({
        "m": args.m, "n": args.n, "k": args.k,
        "bm": args.bm, "bn": args.bn, "bk": args.bk,
        "order": args.order, "splitk": args.splitk, "dtype": args.dtype,
        "meas_dram_read_B": int(read_b), "meas_dram_write_B": int(write_b),
        "meas_L2_hit_pct": round(hit_pct, 3),
        "snowcat_miss_B": int(snowcat_miss),
        "fitted_alpha": best_a, "model_dram_B": int(model_b),
        "model_implied_hit_pct": round(implied_hit, 3),
        "kernel_ms": round(dur_ns / 1e6, 4) if dur_ns == dur_ns else "",
    })
    print(f"\n  appended to {CALIB_CSV.name} (pool shapes, then use a median alpha).")


if __name__ == "__main__":
    main()
