#!/bin/bash

# ==============================================================
# Ablation Study Runner: Houston13 -> Houston18
# ==============================================================
# BEFORE RUNNING: install required packages:
#   pip install torchsampler cleanlab hdf5storage
#   pip install torch_geometric  (follow: https://pytorch-geometric.readthedocs.io)
# ==============================================================

SEEDS=(678 681 774 789)
SOURCE="Houston13"
TARGET="Houston18"
INTERNAL_MODELS=("GAHT" "BiDA" "AgentBiDA" "SelfAttentionAgentBiDA")
EXTERNAL_METHODS=("PCADA" "TSTnet" "MDGTnet" "CLDA" "SCLUDA" "SSWADA" "CACL" "MLUDA")

echo "Starting Ablation Study ($SOURCE -> $TARGET)..."

# Clear stale JSON results to avoid using old buggy cached values
echo "Clearing old result cache..."
rm -f ablation_results/*.json
mkdir -p ablation_results

# 1. Run Internal Models
for MODEL in "${INTERNAL_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "============================================="
        echo "Running Internal Model: $MODEL | Seed: $SEED"
        echo "============================================="
        python main.py --model $MODEL --source_name $SOURCE --target_name $TARGET --seed $SEED
    done
done

# 2. Run External Models
# Note: External models do NOT accept --source_name/--target_name; their data paths are hardcoded.
for EXT in "${EXTERNAL_METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "============================================="
        echo "Running External Model: $EXT | Seed: $SEED"
        echo "============================================="

        pushd external_methods/$EXT > /dev/null

        case "$EXT" in
            "PCADA")    python train_pcada_houston.py --seed $SEED ;;
            "TSTnet")   python train_tstnet.py --seed $SEED ;;
            "MDGTnet")  python train_H1318_com_cls.py --seed $SEED ;;
            "CLDA")     python CLDA_HOUSTON13_2_18.py --seed $SEED ;;
            "SCLUDA")   python SCLUDA_Houston.py --seed $SEED ;;
            "SSWADA")   python main.py --seed $SEED ;;
            "CACL")     python demo_multiDA.py --seed $SEED ;;
            "MLUDA")    python MLUDA_hu.py --seed $SEED ;;
            *) echo "Warning: Unknown model $EXT" ;;
        esac

        popd > /dev/null
    done
done

echo "All training completed!"
echo "Generating Final Ablation Table (Houston13 -> Houston18)..."
python generate_ablation_table.py
