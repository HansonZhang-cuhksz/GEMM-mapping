"""Model logic for the Snowcat GEMM demo."""

from .mapping import GemmMapping, MappingPoint, enumerate_mappings
from .pareto import best_at_capacity, pareto_frontier
from .performance import attainable_metrics
from .traffic import estimate_mapping_traffic, trace_mapping
from .workload import GemmWorkload, PRECISIONS, divisors
from .decision import (
    Bottleneck,
    DecisionResult,
    Investment,
    compare_next_area_increment,
    doubling_buffer_gain_percent,
    evaluate_capacity,
)

__all__ = [
    "GemmMapping",
    "GemmWorkload",
    "MappingPoint",
    "PRECISIONS",
    "Bottleneck",
    "DecisionResult",
    "Investment",
    "attainable_metrics",
    "best_at_capacity",
    "compare_next_area_increment",
    "divisors",
    "doubling_buffer_gain_percent",
    "enumerate_mappings",
    "evaluate_capacity",
    "estimate_mapping_traffic",
    "pareto_frontier",
    "trace_mapping",
]
