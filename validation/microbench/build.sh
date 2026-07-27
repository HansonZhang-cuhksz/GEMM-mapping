#!/usr/bin/env bash
# Build gemm.cu against the "profiling" conda env (CUDA 12.8) + local CUTLASS.
# Target: RTX 4060 = Ada Lovelace = sm_89.
set -eo pipefail   # not -u: conda's gcc activation script references unbound vars

ENV_NAME="${ENV_NAME:-profiling}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS="${CUTLASS_DIR:-$HERE/cutlass}"

if [[ ! -f "$CUTLASS/include/cutlass/cutlass.h" ]]; then
  echo "ERROR: CUTLASS not found at $CUTLASS" >&2
  echo "  git clone --depth 1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git $CUTLASS" >&2
  exit 1
fi

echo "Compiling gemm.cu (~16 mappings x 2 dtypes; ~5-9 min)..."
nvcc -arch=sm_89 -std=c++17 -O3 \
  --expt-relaxed-constexpr \
  -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include" \
  -I"$CONDA_PREFIX/targets/x86_64-linux/include" \
  "$HERE/gemm.cu" -o "$HERE/gemm" \
  -L"$CONDA_PREFIX/lib" -lcublas

echo "Built: $HERE/gemm"
