"""Shared timing methodology for the RTX 4060 sim-real measurements (task RTX4060_SIM_REAL_TASK.md).

Mirrors the C500 methodology in metax_glm_measure.py (per-iter cuda Events, median) with the task's
tightened parameters (warmup >= 15, iters >= 30) plus a clock sampler, since clocks cannot be locked
on this machine (WSL2, no root).
"""
import json
import statistics
import subprocess
import threading
import time as _time

import torch

DEV = "cuda:0"
BPE = 2  # bf16


def bf(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


_WARM = {}


def warm_clocks(target_ms=200.0):
    """Ramp DVFS to sustained-boost clocks immediately before a timed region.

    Clocks cannot be locked on this box (WSL2, no root). After any idle gap (e.g.
    sleep_cooldown) the GPU parks at 210 MHz and a short measurement window (< ~50 ms
    of work) completes before the clocks ramp, under-reporting short kernels by up to
    ~7x (observed: 2048x1024x1024 at 2.2 TF/s cold vs ~17 TF/s warm). A ~200 ms busy
    GEMM right before the timed region restores boost; continuous measurement load
    then keeps it there."""
    if "a" not in _WARM:
        _WARM["a"] = torch.randn(2048, 2048, device=DEV, dtype=torch.bfloat16)
        _WARM["b"] = torch.randn(2048, 2048, device=DEV, dtype=torch.bfloat16)
    a, b = _WARM["a"], _WARM["b"]
    torch.cuda.synchronize()
    t0 = _time.perf_counter()
    while (_time.perf_counter() - t0) * 1e3 < target_ms:
        # queue a continuous batch (no per-iter sync: idle micro-gaps between kernels
        # keep this laptop's DVFS from committing to boost clocks)
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()


def med_time(fn, iters=30, warmup=15, warm=True):
    """Median wall time of fn() in SECONDS via per-iteration cuda Events.

    warm=True ramps DVFS clocks first (see warm_clocks) -- mandatory on this unlocked-
    clock box for any short measurement; harmless for long ones."""
    if warm:
        warm_clocks()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


class ClockSampler:
    """Samples nvidia-smi clocks/power/temp on a background thread while measurements run."""

    QUERY = "clocks.gr,clocks.mem,temperature.gpu,power.draw"

    def __init__(self, period_s=0.5):
        self.period_s = period_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _sample_once(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().split(", ")
            self.samples.append({
                "gr_mhz": float(out[0]), "mem_mhz": float(out[1]),
                "temp_c": float(out[2]),
                "power_w": float(out[3]) if out[3] not in ("[N/A]", "N/A") else None,
            })
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return {}
        def stats(key):
            vals = [s[key] for s in self.samples if s[key] is not None]
            if not vals:
                return None
            return {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        return {k: stats(k) for k in ("gr_mhz", "mem_mhz", "temp_c", "power_w")} | {
            "n_samples": len(self.samples)
        }


def geomean(xs):
    import math
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[wrote {path}]")


def sleep_cooldown(seconds=2.0):
    """Brief pause between measurement groups so the previous group's thermal state bleeds off."""
    torch.cuda.synchronize()
    _time.sleep(seconds)
