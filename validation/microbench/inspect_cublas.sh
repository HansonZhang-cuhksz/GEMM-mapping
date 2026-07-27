#!/usr/bin/env bash
# Inspect the mapping cuBLAS selects for a given GEMM size.
# ---------------------------------------------------------------------------
# The classic cuBLAS API (cublasGemmEx) is a black box, BUT it dispatches
# internally through cuBLASLt, whose level-5 log prints the chosen algorithm:
# CTA tile, stages (tileK x num_stages), split-K count, and CTA swizzling.
# We run the benchmark binary with only cuBLAS active (no CUTLASS mapping name
# matches "__none__") and pretty-print the per-call heuristic decision.
#
# Usage: ./inspect_cublas.sh [--m N --n N --k N --dtype fp16|bf16]
#   (defaults come from gemm.cu; pass the size you care about)
#
# NOTE: rows/cols in the log look transposed (e.g. K x M) because a row-major
# C=A*B is issued to column-major cuBLAS with A and B swapped. "Adesc" below is
# actually operand B, "Bdesc" is operand A. The tile/stages/split-K are correct.
set -eo pipefail
ENV_NAME="${ENV_NAME:-profiling}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
[[ -x "$HERE/gemm" ]] || { echo "build first: ./build.sh" >&2; exit 1; }

echo "Querying cuBLAS heuristic (via cuBLASLt log). Legend:"
echo "  tile=MATMUL_TILE_<M>x<N>   stages=MATMUL_STAGES_<tileK>x<numStages>"
echo "  numSplitsK=<split-K>       ctaSwizzling=<raster/loop-order>"
echo "-----------------------------------------------------------------------"
CUBLASLT_LOG_LEVEL=5 "$HERE/gemm" --config __none__ --iters 1 --warmup 0 "$@" 2>&1 \
  | grep -E "\[Trace\].*Matmul\]" \
  | grep -oE "(Adesc=\[[^]]*\]|Bdesc=\[[^]]*\]|algo=\[[^]]*\])" \
  | paste - - - \
  | sed -E 's/\t/\n    /g; s/^/GEMM  /'
