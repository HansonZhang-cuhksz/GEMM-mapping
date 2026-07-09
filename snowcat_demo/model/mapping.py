from __future__ import annotations

from dataclasses import dataclass

from .traffic import LOOP_ORDERS, TrafficBreakdown, estimate_mapping_traffic
from .workload import GemmWorkload, divisors


@dataclass(frozen=True, slots=True)
class GemmMapping:
    m0: int
    k0: int
    n0: int
    loop_order: tuple[str, str, str]

    @property
    def label(self) -> str:
        return f"M0={self.m0}, K0={self.k0}, N0={self.n0}, order={self.order_label}"

    @property
    def order_label(self) -> str:
        return "-".join(self.loop_order)


@dataclass(frozen=True, slots=True)
class MappingPoint:
    mapping: GemmMapping
    traffic: TrafficBreakdown

    @property
    def buffer_bytes(self) -> int:
        return self.traffic.buffer_bytes

    @property
    def backing_store_bytes(self) -> int:
        return self.traffic.total_bytes


def enumerate_mappings(workload: GemmWorkload) -> list[MappingPoint]:
    points: list[MappingPoint] = []
    for m0 in divisors(workload.m):
        for k0 in divisors(workload.k):
            for n0 in divisors(workload.n):
                for loop_order in LOOP_ORDERS:
                    mapping = GemmMapping(m0, k0, n0, loop_order)
                    traffic = estimate_mapping_traffic(
                        workload, m0, k0, n0, loop_order
                    )
                    points.append(MappingPoint(mapping=mapping, traffic=traffic))
    return points

