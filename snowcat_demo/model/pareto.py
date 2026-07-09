from __future__ import annotations

from itertools import groupby

from .mapping import MappingPoint


def pareto_frontier(points: list[MappingPoint]) -> list[MappingPoint]:
    if not points:
        return []

    frontier: list[MappingPoint] = []
    best_traffic = float("inf")
    ordered = sorted(points, key=lambda point: (point.buffer_bytes, point.backing_store_bytes))

    for _, group_iter in groupby(ordered, key=lambda point: point.buffer_bytes):
        group = list(group_iter)
        group_best = min(point.backing_store_bytes for point in group)
        if group_best < best_traffic:
            frontier.extend(
                point for point in group if point.backing_store_bytes == group_best
            )
            best_traffic = group_best

    return frontier


def best_at_capacity(
    points: list[MappingPoint], capacity_bytes: int
) -> MappingPoint | None:
    eligible = [point for point in points if point.buffer_bytes <= capacity_bytes]
    if not eligible:
        return None
    return min(eligible, key=lambda point: (point.backing_store_bytes, point.buffer_bytes))


def is_pareto_point(point: MappingPoint, frontier: list[MappingPoint]) -> bool:
    return any(
        point.mapping == candidate.mapping
        and point.buffer_bytes == candidate.buffer_bytes
        and point.backing_store_bytes == candidate.backing_store_bytes
        for candidate in frontier
    )

