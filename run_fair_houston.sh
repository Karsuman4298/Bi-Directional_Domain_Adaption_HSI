#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON="${PYTHON:-python3}"
DIRECTION="${1:-both}"
MODE="${2:-full}"
read -r -a SEED_LIST <<< "${SEEDS:-2100 2101 2102 2103 2104}"
read -r -a MODEL_LIST <<< "${MODELS:-GAHT BiDA AgentBiDA SelfAttentionAgentBiDA TSTnet CLDA SCLUDA MLUDA SSWADA CACL}"
EPOCHS="${EPOCHS:-120}"
OUTPUT="${OUTPUT:-fair_results}"
WORKERS="${WORKERS:-2}"
DEVICE="${DEVICE:-cuda:0}"
case "$DIRECTION" in
  13to18) DIRECTIONS=(13to18) ;;
  18to13) DIRECTIONS=(18to13) ;;
  both) DIRECTIONS=(13to18 18to13) ;;
  *) echo 'Usage: bash run_fair_houston.sh [13to18|18to13|both] [full|smoke|ablation]'; exit 2 ;;
esac
case "$MODE" in full|smoke|ablation) ;; *) echo 'Unknown mode'; exit 2 ;; esac
mkdir -p "$OUTPUT/logs"
for direction in "${DIRECTIONS[@]}"; do
  if [[ "$direction" == 13to18 ]]; then SOURCE=Houston13; TARGET=Houston18; else SOURCE=Houston18; TARGET=Houston13; fi
  if [[ "$MODE" == ablation ]]; then MODELS_TO_RUN=(BiDA AgentBiDA SelfAttentionAgentBiDA); else MODELS_TO_RUN=("${MODEL_LIST[@]}"); fi
  for model in "${MODELS_TO_RUN[@]}"; do
    if [[ "$MODE" == ablation ]]; then ABLATIONS=(full source_only no_mmd no_distill no_consistency no_pairing); else ABLATIONS=(full); fi
    for ablation in "${ABLATIONS[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        extra=()
        if [[ "$MODE" == smoke ]]; then extra+=(--smoke --batch-size 4); fi
        "$PYTHON" -u -m experiments.run --model "$model" --source "$SOURCE" --target "$TARGET" \
          --seed "$seed" --epochs "$EPOCHS" --workers "$WORKERS" --device "$DEVICE" \
          --output "$OUTPUT" --ablation "$ablation" --num-agents "${NUM_AGENTS:-4}" \
          --skip-complete "${extra[@]}" 2>&1 | tee "$OUTPUT/logs/${direction}_${model}_${ablation}_${seed}.log"
        if [[ "$MODE" == smoke ]]; then break; fi
      done
    done
  done
  if [[ "$MODE" != smoke ]]; then
    if [[ "$MODE" == ablation ]]; then ABLATIONS=(full source_only no_mmd no_distill no_consistency no_pairing); else ABLATIONS=(full); fi
    for ablation in "${ABLATIONS[@]}"; do
      "$PYTHON" -m experiments.table --root "$OUTPUT" --out "$OUTPUT/tables" --source "$SOURCE" --target "$TARGET" \
        --models "${MODELS_TO_RUN[@]}" --seeds "${SEED_LIST[@]}" --ablation "$ablation" --num-agents "${NUM_AGENTS:-4}"
    done
    if [[ "$MODE" == ablation ]]; then
      "$PYTHON" -m experiments.ablation_table --root "$OUTPUT" --source "$SOURCE" --target "$TARGET" \
        --seeds "${SEED_LIST[@]}" --num-agents "${NUM_AGENTS:-4}"
    fi
  fi
done
