#!/usr/bin/env bash
# Separate workflow without input PCA; see docs/DATASET_BENCHMARK.md for alignment.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT_DIR/run_all_dataset_experiments.sh" \
  --mode no-pca --output "${BENCHMARK_OUTPUT:-benchmark_runs/no_pca}" "$@"
