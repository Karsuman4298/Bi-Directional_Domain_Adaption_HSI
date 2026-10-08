#!/usr/bin/env bash
# Separate PCA-only workflow: every method receives exactly 30 input bands.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT_DIR/run_all_dataset_experiments.sh" \
  --mode pca --output "${BENCHMARK_OUTPUT:-benchmark_runs/pca30}" "$@"
