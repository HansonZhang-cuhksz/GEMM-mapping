#!/usr/bin/env bash
# Run the gemm benchmark inside the "profiling" env so libcublas is on the loader path.
set -eo pipefail   # not -u: conda's gcc activation script references unbound vars
ENV_NAME="${ENV_NAME:-profiling}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
[[ -x "$HERE/gemm" ]] || { echo "build first: ./build.sh" >&2; exit 1; }
exec "$HERE/gemm" "$@"
