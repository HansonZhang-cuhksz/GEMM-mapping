from __future__ import annotations

from dataclasses import dataclass

from .mapping import MappingPoint
from .pareto import pareto_frontier
from .workload import GemmWorkload


@dataclass(frozen=True, slots=True)
class AttainableMetric:
    capacity_bytes: int
    traffic_bytes: int
    operational_intensity: float
    performance_tflops: float


def operational_intensity(workload: GemmWorkload, traffic_bytes: int) -> float:
    if traffic_bytes <= 0:
        raise ValueError("traffic_bytes must be positive")
    return workload.operations / traffic_bytes


def throughput_tflops(
    operational_intensity_ops_per_byte: float,
    memory_bandwidth_gb_s: float,
    peak_compute_tflops: float,
) -> float:
    bandwidth_limited_tflops = (
        memory_bandwidth_gb_s * 1e9 * operational_intensity_ops_per_byte / 1e12
    )
    return min(peak_compute_tflops, bandwidth_limited_tflops)


def attainable_metrics(
    workload: GemmWorkload,
    points: list[MappingPoint],
    memory_bandwidth_gb_s: float,
    peak_compute_tflops: float,
) -> list[AttainableMetric]:
    metrics: list[AttainableMetric] = []
    seen_capacities: set[int] = set()
    for point in pareto_frontier(points):
        if point.buffer_bytes in seen_capacities:
            continue
        seen_capacities.add(point.buffer_bytes)
        oi = operational_intensity(workload, point.backing_store_bytes)
        metrics.append(
            AttainableMetric(
                capacity_bytes=point.buffer_bytes,
                traffic_bytes=point.backing_store_bytes,
                operational_intensity=oi,
                performance_tflops=throughput_tflops(
                    oi, memory_bandwidth_gb_s, peak_compute_tflops
                ),
            )
        )
    return metrics

