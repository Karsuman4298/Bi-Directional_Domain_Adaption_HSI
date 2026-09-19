#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MODELS="${MODELS:-GAHT BiDA AgentBiDA SelfAttentionAgentBiDA}"
exec bash run_fair_houston.sh 18to13 "${1:-full}"
