from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .mapping import MappingPoint
from .pareto import best_at_capacity
from .performance import operational_intensity, throughput_tflops
from .workload import GemmWorkload


class Bottleneck(StrEnum):
    MEMORY = "memory"
    COMPUTE = "compute"
    BALANCED = "balanced"


class Investment(StrEnum):
    SRAM = "sram"
    COMPUTE = "compute"
    TIE = "tie"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CapacityResult:
    capacity_bytes: int
    traffic_bytes: int
    operational_intensity: float
    performance_tflops: float
    bottleneck: Bottleneck
    mapping: MappingPoint


@dataclass(frozen=True, slots=True)
class DecisionResult:
    baseline: CapacityResult | None
    sram_option: CapacityResult | None
    compute_option: CapacityResult | None
    recommendation: Investment
    sram_gain_percent: float
    compute_gain_percent: float


def evaluate_capacity(
    workload: GemmWorkload,
    points: list[MappingPoint],
    capacity_bytes: int,
    memory_bandwidth_gb_s: float,
    peak_compute_tflops: float,
) -> CapacityResult | None:
    best = best_at_capacity(points, capacity_bytes)
    if best is None:
        return None
    oi = operational_intensity(workload, best.backing_store_bytes)
    bandwidth_limited = memory_bandwidth_gb_s * 1e9 * oi / 1e12
    performance = min(peak_compute_tflops, bandwidth_limited)
    if abs(bandwidth_limited - peak_compute_tflops) <= max(peak_compute_tflops, 1.0) * 1e-6:
        bottleneck = Bottleneck.BALANCED
    elif bandwidth_limited < peak_compute_tflops:
        bottleneck = Bottleneck.MEMORY
    else:
        bottleneck = Bottleneck.COMPUTE
    return CapacityResult(
        capacity_bytes=capacity_bytes,
        traffic_bytes=best.backing_store_bytes,
        operational_intensity=oi,
        performance_tflops=performance,
        bottleneck=bottleneck,
        mapping=best,
    )


def compare_next_area_increment(
    workload: GemmWorkload,
    points: list[MappingPoint],
    capacity_bytes: int,
    memory_bandwidth_gb_s: float,
    peak_compute_tflops: float,
    sram_increment_bytes: int,
    compute_increment_tflops: float,
) -> DecisionResult:
    baseline = evaluate_capacity(
        workload, points, capacity_bytes, memory_bandwidth_gb_s, peak_compute_tflops
    )
    sram_option = evaluate_capacity(
        workload,
        points,
        capacity_bytes + max(sram_increment_bytes, 0),
        memory_bandwidth_gb_s,
        peak_compute_tflops,
    )
    compute_option = evaluate_capacity(
        workload,
        points,
        capacity_bytes,
        memory_bandwidth_gb_s,
        peak_compute_tflops + max(compute_increment_tflops, 0.0),
    )

    if baseline is None or sram_option is None or compute_option is None:
        return DecisionResult(
            baseline=baseline,
            sram_option=sram_option,
            compute_option=compute_option,
            recommendation=Investment.UNAVAILABLE,
            sram_gain_percent=0.0,
            compute_gain_percent=0.0,
        )

    sram_gain = _gain_percent(baseline.performance_tflops, sram_option.performance_tflops)
    compute_gain = _gain_percent(
        baseline.performance_tflops, compute_option.performance_tflops
    )
    tolerance = 1e-9
    if abs(sram_gain - compute_gain) <= tolerance:
        recommendation = Investment.TIE
    elif sram_gain > compute_gain:
        recommendation = Investment.SRAM
    else:
        recommendation = Investment.COMPUTE

    return DecisionResult(
        baseline=baseline,
        sram_option=sram_option,
        compute_option=compute_option,
        recommendation=recommendation,
        sram_gain_percent=sram_gain,
        compute_gain_percent=compute_gain,
    )


def doubling_buffer_gain_percent(
    workload: GemmWorkload,
    points: list[MappingPoint],
    capacity_bytes: int,
    memory_bandwidth_gb_s: float,
    peak_compute_tflops: float,
) -> float | None:
    baseline = evaluate_capacity(
        workload, points, capacity_bytes, memory_bandwidth_gb_s, peak_compute_tflops
    )
    doubled = evaluate_capacity(
        workload, points, capacity_bytes * 2, memory_bandwidth_gb_s, peak_compute_tflops
    )
    if baseline is None or doubled is None:
        return None
    return _gain_percent(baseline.performance_tflops, doubled.performance_tflops)


def _gain_percent(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (after - before) / before * 100.0

