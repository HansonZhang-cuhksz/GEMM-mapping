from __future__ import annotations

from dataclasses import dataclass


PRECISIONS: dict[str, int] = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "INT8": 1,
}


def divisors(n: int) -> list[int]:
    if n <= 0:
        raise ValueError("n must be positive")
    small: list[int] = []
    large: list[int] = []
    candidate = 1
    while candidate * candidate <= n:
        if n % candidate == 0:
            small.append(candidate)
            paired = n // candidate
            if paired != candidate:
                large.append(paired)
        candidate += 1
    return small + large[::-1]


@dataclass(frozen=True, slots=True)
class GemmWorkload:
    m: int
    k: int
    n: int
    bytes_per_element: int

    def __post_init__(self) -> None:
        for name, value in (("m", self.m), ("k", self.k), ("n", self.n)):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")

    @property
    def operations(self) -> int:
        return 2 * self.m * self.k * self.n

    @property
    def a_bytes(self) -> int:
        return self.m * self.k * self.bytes_per_element

    @property
    def w_bytes(self) -> int:
        return self.k * self.n * self.bytes_per_element

    @property
    def b_bytes(self) -> int:
        return self.m * self.n * self.bytes_per_element

    @property
    def algorithmic_minimum_bytes(self) -> int:
        return self.a_bytes + self.w_bytes + self.b_bytes

    def tile_counts(self, m0: int, k0: int, n0: int) -> tuple[int, int, int]:
        if self.m % m0 != 0 or self.k % k0 != 0 or self.n % n0 != 0:
            raise ValueError("tile sizes must evenly divide workload dimensions")
        return self.m // m0, self.k // k0, self.n // n0

    def tile_bytes(self, m0: int, k0: int, n0: int) -> tuple[int, int, int]:
        a_tile = m0 * k0 * self.bytes_per_element
        w_tile = k0 * n0 * self.bytes_per_element
        b_tile = m0 * n0 * self.bytes_per_element
        return a_tile, w_tile, b_tile
