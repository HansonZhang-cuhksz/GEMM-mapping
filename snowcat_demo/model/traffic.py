from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .workload import GemmWorkload


LOOP_ORDERS: tuple[tuple[str, str, str], ...] = (
    ("M", "K", "N"),
    ("M", "N", "K"),
    ("K", "M", "N"),
    ("K", "N", "M"),
    ("N", "M", "K"),
    ("N", "K", "M"),
)


@dataclass(frozen=True, slots=True)
class TrafficBreakdown:
    buffer_bytes: int
    a_read_bytes: int
    w_read_bytes: int
    b_read_bytes: int
    b_write_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.a_read_bytes
            + self.w_read_bytes
            + self.b_read_bytes
            + self.b_write_bytes
        )


def _loop_coordinates(
    mt: int, kt: int, nt: int, loop_order: tuple[str, str, str]
) -> Iterable[tuple[int, int, int]]:
    extents = {"M": mt, "K": kt, "N": nt}
    for values in product(*(range(extents[dim]) for dim in loop_order)):
        coords = dict(zip(loop_order, values, strict=True))
        yield coords["M"], coords["K"], coords["N"]


def estimate_mapping_traffic(
    workload: GemmWorkload,
    m0: int,
    k0: int,
    n0: int,
    loop_order: tuple[str, str, str],
) -> TrafficBreakdown:
    return estimate_mapping_traffic_closed_form(
        workload, m0, k0, n0, loop_order
    )


def estimate_mapping_traffic_closed_form(
    workload: GemmWorkload,
    m0: int,
    k0: int,
    n0: int,
    loop_order: tuple[str, str, str],
) -> TrafficBreakdown:
    if loop_order not in LOOP_ORDERS:
        raise ValueError(f"unsupported loop order: {loop_order}")

    mt, kt, nt = workload.tile_counts(m0, k0, n0)
    a_tile_bytes, w_tile_bytes, b_tile_bytes = workload.tile_bytes(m0, k0, n0)
    buffer_bytes = a_tile_bytes + w_tile_bytes + b_tile_bytes
    extents = {"M": mt, "K": kt, "N": nt}

    def run_count(key_dims: tuple[str, ...]) -> int:
        varying_key_positions = [
            loop_order.index(dim) for dim in key_dims if extents[dim] > 1
        ]
        if not varying_key_positions:
            return 1
        deepest_key_position = max(varying_key_positions)
        count = 1
        for dim in loop_order[: deepest_key_position + 1]:
            count *= extents[dim]
        return count

    a_reads = run_count(("M", "K")) * a_tile_bytes
    w_reads = run_count(("K", "N")) * w_tile_bytes

    varying_output_positions = [
        loop_order.index(dim) for dim in ("M", "N") if extents[dim] > 1
    ]
    deepest_output_position = max(varying_output_positions, default=-1)
    k_is_inside_output_run = loop_order.index("K") > deepest_output_position
    output_tiles = mt * nt

    if k_is_inside_output_run:
        b_reads = 0
        b_writes = output_tiles * b_tile_bytes
    else:
        b_reads = output_tiles * max(kt - 1, 0) * b_tile_bytes
        b_writes = output_tiles * kt * b_tile_bytes

    return TrafficBreakdown(
        buffer_bytes=buffer_bytes,
        a_read_bytes=a_reads,
        w_read_bytes=w_reads,
        b_read_bytes=b_reads,
        b_write_bytes=b_writes,
    )


def trace_mapping(
    workload: GemmWorkload,
    m0: int,
    k0: int,
    n0: int,
    loop_order: tuple[str, str, str],
    limit: int = 24,
) -> list[str]:
    _, trace = _simulate_mapping(
        workload, m0, k0, n0, loop_order, capture_trace=True, trace_limit=limit
    )
    return trace


def _simulate_mapping(
    workload: GemmWorkload,
    m0: int,
    k0: int,
    n0: int,
    loop_order: tuple[str, str, str],
    capture_trace: bool,
    trace_limit: int = 24,
) -> tuple[TrafficBreakdown, list[str]]:
    if loop_order not in LOOP_ORDERS:
        raise ValueError(f"unsupported loop order: {loop_order}")

    mt, kt, nt = workload.tile_counts(m0, k0, n0)
    a_tile_bytes, w_tile_bytes, b_tile_bytes = workload.tile_bytes(m0, k0, n0)
    buffer_bytes = a_tile_bytes + w_tile_bytes + b_tile_bytes

    a_cache: tuple[int, int] | None = None
    w_cache: tuple[int, int] | None = None
    b_cache: tuple[int, int] | None = None
    b_dirty = False
    b_progress: defaultdict[tuple[int, int], int] = defaultdict(int)

    a_reads = 0
    w_reads = 0
    b_reads = 0
    b_writes = 0
    trace: list[str] = []

    def add_trace(line: str) -> None:
        if capture_trace and len(trace) < trace_limit:
            trace.append(line)

    for mi, ki, ni in _loop_coordinates(mt, kt, nt, loop_order):
        a_key = (mi, ki)
        if a_cache != a_key:
            a_reads += a_tile_bytes
            a_cache = a_key
            add_trace(f"Load A[{mi},{ki}]")

        w_key = (ki, ni)
        if w_cache != w_key:
            w_reads += w_tile_bytes
            w_cache = w_key
            add_trace(f"Load W[{ki},{ni}]")

        b_key = (mi, ni)
        if b_cache != b_key:
            if b_cache is not None and b_dirty:
                b_writes += b_tile_bytes
                add_trace(f"Spill partial B[{b_cache[0]},{b_cache[1]}]")
            if 0 < b_progress[b_key] < kt:
                b_reads += b_tile_bytes
                add_trace(f"Reload partial B[{mi},{ni}]")
            b_cache = b_key
            b_dirty = False

        if b_progress[b_key] >= kt:
            raise RuntimeError("encountered an already-complete output tile")

        b_progress[b_key] += 1
        b_dirty = True
        add_trace(f"Update B[{mi},{ni}] with K tile {ki}")

        if b_progress[b_key] == kt:
            b_writes += b_tile_bytes
            b_dirty = False
            add_trace(f"Write final B[{mi},{ni}]")

    if b_cache is not None and b_dirty:
        b_writes += b_tile_bytes
        add_trace(f"Flush partial B[{b_cache[0]},{b_cache[1]}]")

    if any(progress != kt for progress in b_progress.values()):
        raise RuntimeError("simulation ended with incomplete output tiles")

    breakdown = TrafficBreakdown(
        buffer_bytes=buffer_bytes,
        a_read_bytes=a_reads,
        w_read_bytes=w_reads,
        b_read_bytes=b_reads,
        b_write_bytes=b_writes,
    )
    return breakdown, trace
