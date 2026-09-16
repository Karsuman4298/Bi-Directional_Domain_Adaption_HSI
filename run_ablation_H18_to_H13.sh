#!/bin/bash

# Configuration
SEEDS=(678 681 774 789)
SOURCE="Houston18"
TARGET="Houston13"
INTERNAL_MODELS=("GAHT" "BiDA" "AgentBiDA" "SelfAttentionAgentBiDA")
EXTERNAL_METHODS=("PCADA" "TSTnet" "MDGTnet" "CLDA" "SCLUDA" "SSWADA" "CACL" "MLUDA")

echo "Starting Ablation Study ($SOURCE -> $TARGET)..."
# Use a different output directory to avoid overwriting H13->H18 results
export ABLATION_DIR="ablation_results_H18_H13"
mkdir -p $ABLATION_DIR

# 1. Run Internal Models
for MODEL in "${INTERNAL_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "============================================="
        echo "Running Internal Model: $MODEL | Seed: $SEED"
        echo "============================================="
        python main.py --model $MODEL --source_name $SOURCE --target_name $TARGET --seed $SEED
        # The internal model might still save to ablation_results in the python script
        # So we manually move it if it was created there
        mv ablation_results/${MODEL}_results_seed_${SEED}.json $ABLATION_DIR/ 2>/dev/null
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
            python $TRAIN_SCRIPT --seed $SEED --source_name $SOURCE --target_name $TARGET
        else
            echo "Warning: No training script found for $EXT"
        fi
        
        popd
        
        # Move the json output from the default folder to our new isolated folder
        mv ablation_results/${EXT}_results_seed_${SEED}.json $ABLATION_DIR/ 2>/dev/null
    done
done

echo "All training completed!"
echo "Generating Final Ablation Table for H18 -> H13..."
python generate_table_H18_to_H13.py
