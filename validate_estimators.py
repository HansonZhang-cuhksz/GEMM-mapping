#!/usr/bin/env python
"""Validate both time estimators (snowcat-roofline and optimistic min-traffic)
against measured CUTLASS times across many GEMM shapes, and log every mapping
selection (CUTLASS auto-tune, both estimators' --optimal picks, cuBLAS algo).

Protocol per size:
  * measured : ./gemm --no-splitk fair sweep -> best mapping (tile+stages) + median ms,
               plus the cuBLAS baseline ms (cuBLAS may use split-K internally; it is a
               reference, not the estimation target).
  * estimators: --optimal search (BM,BN>=64, BK>=32, split_k=1) -> tile + est ms;
               plus an estimate of the MEASURED-BEST mapping (per-mapping accuracy).
  * cublas map: CUBLASLT_LOG_LEVEL=5 one-shot run -> tile/stages/splitK cuBLAS chose.

Writes estimator_validation.csv (raw) and prints a progress log. Run inside the
profiling env from the repo root:
  PYTHONPATH=/home/shuhan/GEMM-mapping:/home/shuhan/snowcat-demo \
    conda run -n profiling python validate_estimators.py
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

sys.path.insert(0, "/home/shuhan/GEMM-mapping")
sys.path.append("/home/shuhan/snowcat-demo")

from gemm_time_estimator import (  # noqa: E402
    GPUS, Mapping, estimate_best_order, optimal_mapping_by_time)
from gemm_time_estimator_min import (  # noqa: E402
    estimate_gemm_time_min, optimal_mapping_min_by_time)

REPO = "/home/shuhan/GEMM-mapping"
GPU = GPUS["rtx4060-measured"]

# (pattern, M, N, K, iters)  -- iters scaled so each mapping times ~1-2 s
SIZES = [
    ("square-small", 512,  512,  512,  400),
    ("square-mid",   2048, 2048, 2048, 60),
    ("square-large", 4096, 4096, 4096, 15),
    ("skinny-M128",  128,  4096, 4096, 250),
    ("skinny-M64",   64,   4096, 4096, 300),
    ("skinny-N128",  4096, 128,  4096, 250),
    ("short-K256",   4096, 4096, 256,  100),
    ("deep-K8192",   512,  512,  8192, 150),
    ("wide",         1024, 8192, 2048, 40),
    ("tall",         8192, 1024, 2048, 40),
]

ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = f"{ENV.get('CONDA_PREFIX','')}/lib:" + ENV.get("LD_LIBRARY_PATH", "")


def run_measured(m, n, k, iters):
    """Fair CUTLASS sweep (non-split-K). Returns dict with best mapping + times."""
    cmd = ["./gemm", "--dtype", "fp16", "--m", str(m), "--n", str(n), "--k", str(k),
           "--no-splitk", "--iters", str(iters), "--rounds", "5", "--warmup", "20"]
    out = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True,
                         timeout=1800).stdout
    at = out[out.find("[auto-tune]"):]
    name = re.search(r'best CUTLASS mapping for .*: "(.*)"', at)
    tile = re.search(r"ThreadblockShape = (\d+) x (\d+) x (\d+)", at)
    stages = re.search(r"Stages\s+= (\d+)", at)
    best = re.search(r"-> ([\d.]+) ms, [\d.]+ TFLOP/s, ([\d.]+)x cuBLAS", at)
    cub = re.search(r"\[cuBLAS\s*\] ([\d.]+) ms", out)
    if not (name and tile and stages and best):
        raise RuntimeError(f"parse failure; tail:\n{out[-2000:]}")
    return {
        "name": name.group(1),
        "bm": int(tile.group(1)), "bn": int(tile.group(2)), "bk": int(tile.group(3)),
        "stages": int(stages.group(1)),
        "ms": float(best.group(1)), "vs_cublas": float(best.group(2)),
        "cublas_ms": float(cub.group(1)) if cub else float("nan"),
    }


def run_cublas_algo(m, n, k):
    """One-shot cuBLASLt-logged run; return cuBLAS's chosen algo string."""
    env = dict(ENV); env["CUBLASLT_LOG_LEVEL"] = "5"
    cmd = ["./gemm", "--dtype", "fp16", "--m", str(m), "--n", str(n), "--k", str(k),
           "--config", "__none__", "--iters", "1", "--warmup", "0", "--rounds", "1"]
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=600)
    text = out.stdout + out.stderr
    algos = re.findall(r"algo=\[([^\]]*)\]", text)
    if not algos:
        return "n/a"
    a = algos[-1]  # last Matmul trace = the benchmark-size GEMM
    tile = re.search(r"tile=MATMUL_TILE_(\d+x\d+)", a)
    stg = re.search(r"stages=MATMUL_STAGES_(\d+x\d+)", a)
    spk = re.search(r"numSplitsK=(\d+)", a)
    return (f"tile{tile.group(1) if tile else '?'}"
            f"_stg{stg.group(1) if stg else '?'}"
            f"_splitK{spk.group(1) if spk else '1'}")


def est_on(m, n, k, bm, bn, bk, which):
    """Estimate a specific tile with one estimator; 'n/a' if tile doesn't divide."""
    try:
        mp = Mapping(bm=bm, bn=bn, bk=bk, loop_order=("M", "N", "K"),
                     num_stages=None, split_k=1)
        if which == "snow":
            e, _ = estimate_best_order(m, n, k, mp, GPU)
        else:
            e = estimate_gemm_time_min(m, n, k, mp, GPU)
        return e.time_s * 1e3
    except ValueError:
        return None


def main():
    path = os.path.join(REPO, "estimator_validation.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "M", "N", "K",
                    "cutlass_best", "cutlass_tile", "cutlass_stages",
                    "meas_ms", "vs_cublas", "cublas_ms", "cublas_algo",
                    "snow_opt_tile", "snow_opt_order", "snow_opt_est_ms",
                    "min_opt_tile", "min_opt_est_ms",
                    "snow_on_best_ms", "min_on_best_ms"])
        f.flush()
        for pattern, m, n, k, iters in SIZES:
            print(f"=== {pattern}: {m}x{n}x{k} ===", flush=True)
            meas = run_measured(m, n, k, iters)
            print(f"  measured best: {meas['name']} "
                  f"({meas['bm']}x{meas['bn']}x{meas['bk']} s{meas['stages']}) "
                  f"{meas['ms']:.3f} ms  ({meas['vs_cublas']:.2f}x cuBLAS "
                  f"{meas['cublas_ms']:.3f} ms)", flush=True)
            algo = run_cublas_algo(m, n, k)
            print(f"  cuBLAS algo  : {algo}", flush=True)

            smap, se = optimal_mapping_by_time(m, n, k, GPU)
            mmap, me = optimal_mapping_min_by_time(m, n, k, GPU)
            print(f"  snow --optimal: {smap.bm}x{smap.bn}x{smap.bk} "
                  f"order={'-'.join(smap.loop_order)}  est {se.time_s*1e3:.3f} ms", flush=True)
            print(f"  min  --optimal: {mmap.bm}x{mmap.bn}x{mmap.bk}  "
                  f"est {me.time_s*1e3:.3f} ms", flush=True)

            sob = est_on(m, n, k, meas["bm"], meas["bn"], meas["bk"], "snow")
            mob = est_on(m, n, k, meas["bm"], meas["bn"], meas["bk"], "min")
            print(f"  est on measured-best tile: snow "
                  f"{'n/a' if sob is None else f'{sob:.3f}'} ms, "
                  f"min {'n/a' if mob is None else f'{mob:.3f}'} ms", flush=True)

            w.writerow([pattern, m, n, k,
                        meas["name"], f"{meas['bm']}x{meas['bn']}x{meas['bk']}",
                        meas["stages"], f"{meas['ms']:.4f}",
                        f"{meas['vs_cublas']:.2f}", f"{meas['cublas_ms']:.4f}", algo,
                        f"{smap.bm}x{smap.bn}x{smap.bk}", "-".join(smap.loop_order),
                        f"{se.time_s*1e3:.4f}",
                        f"{mmap.bm}x{mmap.bn}x{mmap.bk}", f"{me.time_s*1e3:.4f}",
                        "" if sob is None else f"{sob:.4f}",
                        "" if mob is None else f"{mob:.4f}"])
            f.flush()
    print("DONE ->", path, flush=True)


if __name__ == "__main__":
    main()
