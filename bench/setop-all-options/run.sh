#!/usr/bin/env bash
# Build and run the EXCEPT ALL / INTERSECT ALL option benchmark.
# Run inside a Sirius pixi environment, from any worktree that has one:
#   pixi run bash <path>/bench/setop-all-options/run.sh [mode]
# mode: all (default), verify, float, perf, skew. Extra arguments go to the binary.
set -euo pipefail

[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: run inside the pixi env (pixi run bash $0)"; exit 1; }

here="$(cd "$(dirname "$0")" && pwd)"
build="$here/build"
mode="${1:-all}"
shift || true

cmake -S "$here" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build "$build"

out="$here/results-$(date +%Y%m%d-%H%M%S)-$mode.csv"
"$build/setop_all_bench" --mode "$mode" "$@" | tee "$out"
echo "results: $out"
