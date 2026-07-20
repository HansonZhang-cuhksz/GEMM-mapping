# 2-GEMM chain fusion — per-case mappings & times (NVIDIA H100 SXM5 (spec-sheet, analytical))

Chain `C=A@B, E=C@D`. All GEMMs timed by the snowcat-roofline estimator (`estimate_gemm_time`). Tiles are `BM×BN×BK`. Unfused = GEMM1(A@B)+GEMM2(C@D), C via L2 if ≤30 MB else HBM. Fused = one kernel over `mt=M/m0` row-blocks (C on chip; B,D re-read once per block). f=4 shapes, dims 256..16384, H100.

| A | B | D | M×K1×N1×N2 | unfused G1 / G2 tile | unf ms | C | fused m0×mt, G1 / G2 tile | fus ms | speedup | winner |
|---|---|---|---|---|---:|:--:|---|---:|---:|:--:|
| square | square | square | 16384×16384×16384×16384 | 512x64x32 / 512x64x32 | 18.049 | HBM | INFEASIBLE (slice too wide) | — | — | infeasible |
| square | square | tall | 16384×16384×16384×4096 | 512x64x32 / 256x64x32 | 11.812 | HBM | m0=16×1024, 16x64x32 / 16x64x64 | 205.333 | 0.058× | unfuse |
| square | square | wide | 4096×4096×4096×16384 | 512x64x32 / 512x64x32 | 0.716 | HBM | m0=16×256, 16x64x64 / 16x64x32 | 12.871 | 0.056× | unfuse |
| square | tall | square | 16384×16384×4096×4096 | 256x64x32 / 512x64x32 | 3.360 | HBM | m0=16×1024, 16x64x64 / 16x64x64 | 51.484 | 0.065× | unfuse |
| square | tall | tall | 16384×16384×4096×1024 | 256x64x32 / 64x512x32 | 2.931 | HBM | m0=64×256, 64x64x32 / 64x64x64 | 10.429 | 0.281× | unfuse |
| square | tall | wide | 16384×16384×4096×16384 | 256x64x32 / 512x64x32 | 5.044 | HBM | m0=16×1024, 16x64x64 / 16x64x32 | 82.374 | 0.061× | unfuse |
| square | wide | square | 4096×4096×16384×16384 | 512x64x32 / 512x64x32 | 2.865 | HBM | INFEASIBLE (slice too wide) | — | — | infeasible |
| square | wide | tall | 4096×4096×16384×4096 | 512x64x32 / 256x64x32 | 1.270 | HBM | m0=16×256, 16x64x32 / 16x64x64 | 20.533 | 0.062× | unfuse |
| square | wide | wide | 1024×1024×4096×16384 | 64x64x32 / 512x64x32 | 0.152 | L2 | m0=16×64, 16x64x64 / 16x64x32 | 2.577 | 0.059× | unfuse |
| tall | square | square | 16384×4096×4096×4096 | 512x64x32 / 512x64x32 | 1.146 | HBM | m0=16×1024, 16x64x64 / 16x64x64 | 20.593 | 0.056× | unfuse |
| tall | square | tall | 16384×4096×4096×1024 | 512x64x32 / 64x512x32 | 0.716 | HBM | m0=64×256, 64x64x32 / 64x64x64 | 2.617 | 0.274× | unfuse |
| tall | square | wide | 16384×4096×4096×16384 | 512x64x32 / 512x64x32 | 2.829 | HBM | m0=16×1024, 16x64x64 / 16x64x32 | 51.484 | 0.055× | unfuse |
| tall | tall | square | 16384×4096×1024×1024 | 64x512x32 / 64x128x32 | 0.179 | HBM | m0=16×1024, 16x64x256 / 16x64x256 | 0.175 | 1.024× | **FUSE** |
| tall | tall | tall | 16384×4096×1024×256 | 64x512x32 / 64x64x32 | 0.156 | HBM | m0=16×1024, 16x64x256 / 16x64x256 | 0.152 | 1.024× | **FUSE** |
| tall | tall | wide | 16384×4096×1024×4096 | 64x512x32 / 64x512x32 | 0.286 | HBM | m0=16×1024, 16x64x256 / 16x64x64 | 0.278 | 1.030× | **FUSE** |
| tall | wide | square | 16384×4096×16384×16384 | 512x64x32 / 512x64x32 | 11.281 | HBM | INFEASIBLE (slice too wide) | — | — | infeasible |
| tall | wide | tall | 16384×4096×16384×4096 | 512x64x32 / 256x64x32 | 5.044 | HBM | m0=16×1024, 16x64x32 / 16x64x64 | 82.133 | 0.061× | unfuse |
| tall | wide | wide | 4096×1024×4096×16384 | 64x64x32 / 512x64x32 | 0.609 | HBM | m0=16×256, 16x64x64 / 16x64x32 | 10.302 | 0.059× | unfuse |
| wide | square | square | 4096×16384×16384×16384 | 512x64x32 / 512x64x32 | 4.584 | HBM | INFEASIBLE (slice too wide) | — | — | infeasible |
| wide | square | tall | 4096×16384×16384×4096 | 512x64x32 / 256x64x32 | 2.989 | HBM | m0=16×256, 16x64x32 / 16x64x64 | 51.333 | 0.058× | unfuse |
| wide | square | wide | 1024×4096×4096×16384 | 512x64x32 / 512x64x32 | 0.179 | L2 | m0=16×64, 16x64x64 / 16x64x32 | 3.218 | 0.056× | unfuse |
| wide | tall | square | 4096×16384×4096×4096 | 256x64x32 / 512x64x32 | 0.840 | HBM | m0=16×256, 16x64x64 / 16x64x64 | 12.871 | 0.065× | unfuse |
| wide | tall | tall | 4096×16384×4096×1024 | 256x64x32 / 64x512x32 | 0.733 | HBM | m0=64×64, 64x64x32 / 64x64x64 | 2.609 | 0.281× | unfuse |
| wide | tall | wide | 4096×16384×4096×16384 | 256x64x32 / 512x64x32 | 1.270 | HBM | m0=16×256, 16x64x64 / 16x64x32 | 20.593 | 0.062× | unfuse |
| wide | wide | square | 1024×4096×16384×16384 | 512x64x32 / 512x64x32 | 0.716 | HBM | INFEASIBLE (slice too wide) | — | — | infeasible |
| wide | wide | tall | 1024×4096×16384×4096 | 512x64x32 / 256x64x32 | 0.318 | HBM | m0=16×64, 16x64x32 / 16x64x64 | 5.133 | 0.062× | unfuse |
| wide | wide | wide | 256×1024×4096×16384 | 64x64x32 / 256x64x32 | 0.045 | L2 | m0=16×16, 16x64x64 / 16x64x32 | 0.646 | 0.070× | unfuse |

**Totals: FUSE 3 / unfuse 19 / infeasible 5** (of 27). Fusion wins only for tall-A + tall-B (large C round-trip avoided, weights L2-resident).

