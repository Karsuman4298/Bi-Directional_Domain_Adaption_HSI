#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MODELS="${MODELS:-TSTnet CLDA SCLUDA MLUDA SSWADA CACL}"
exec bash run_fair_houston.sh 13to18 "${1:-full}"
