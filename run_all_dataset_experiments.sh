#!/usr/bin/env bash
# Run on the GPU server. Defaults: four pairs, both directions, PCA30 + no PCA.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND=Agg
exec "${PYTHON_BIN:-python3}" -u run_dataset_benchmark.py "$@"
