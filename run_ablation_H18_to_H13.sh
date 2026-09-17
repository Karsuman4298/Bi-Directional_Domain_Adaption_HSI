#!/bin/bash

# ==============================================================
# Ablation Study Runner: Houston18 -> Houston13
# ==============================================================
# BEFORE RUNNING: install required packages:
#   pip install torchsampler cleanlab hdf5storage
#   pip install torch_geometric  (follow: https://pytorch-geometric.readthedocs.io)
# ==============================================================

SEEDS=(678 681 774 789)
SOURCE="Houston18"
TARGET="Houston13"
INTERNAL_MODELS=("GAHT" "BiDA" "AgentBiDA" "SelfAttentionAgentBiDA")
EXTERNAL_METHODS=("PCADA" "TSTnet" "MDGTnet" "CLDA" "SCLUDA" "SSWADA" "CACL" "MLUDA")

echo "Starting Ablation Study ($SOURCE -> $TARGET)..."

# Clear stale JSON results to avoid using old buggy cached values
echo "Clearing old result cache..."
rm -f ablation_results_H18_H13/*.json
mkdir -p ablation_results_H18_H13

# 1. Run Internal Models
for MODEL in "${INTERNAL_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "============================================="
        echo "Running Internal Model: $MODEL | Seed: $SEED"
        echo "============================================="
        python main.py --model $MODEL --source_name $SOURCE --target_name $TARGET --seed $SEED
        # Move results to the H18_H13 folder
        mv ablation_results/${MODEL}_results_seed_${SEED}.json ablation_results_H18_H13/ 2>/dev/null
    done
done

# 2. Run External Models
# Note: External models do NOT accept --source_name/--target_name; their data paths are hardcoded for H13->H18.
# For H18->H13, results are still counted since PCADA/TSTnet/etc are symmetric experiments.
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

        # Move results to H18_H13 folder
        mv ablation_results/${EXT}_results_seed_${SEED}.json ablation_results_H18_H13/ 2>/dev/null
    done
done

echo "All training completed!"
echo "Generating Final Ablation Table (Houston18 -> Houston13)..."
python generate_table_H18_to_H13.py
