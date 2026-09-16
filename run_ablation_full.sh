#!/bin/bash

# Configuration
SEEDS=(678 681 774 789)
SOURCE="Houston13"
TARGET="Houston18"
INTERNAL_MODELS=("GAHT" "BiDA" "AgentBiDA" "SelfAttentionAgentBiDA")
EXTERNAL_METHODS=("PCADA" "TSTnet" "MDGTnet" "CLDA" "SCLUDA" "SSWADA" "CACL" "MLUDA")

echo "Starting Ablation Study ($SOURCE -> $TARGET)..."
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
for EXT in "${EXTERNAL_METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "============================================="
        echo "Running External Model: $EXT | Seed: $SEED"
        echo "============================================="
        
        pushd external_methods/$EXT
        
        # Discover the training script for this external method
        TRAIN_SCRIPT=$(ls train*.py 2>/dev/null | head -n 1)
        if [ -z "$TRAIN_SCRIPT" ]; then
             TRAIN_SCRIPT=$(ls *main*.py 2>/dev/null | head -n 1)
        fi
        
        # Execute the script
        if [ -n "$TRAIN_SCRIPT" ]; then
            python $TRAIN_SCRIPT --seed $SEED
        else
            echo "Warning: No training script found for $EXT"
        fi
        
        popd
    done
done

echo "All training completed!"
echo "Generating Final Ablation Table..."
python generate_ablation_table.py
