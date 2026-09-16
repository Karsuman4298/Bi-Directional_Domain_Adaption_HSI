#!/bin/bash

# Configuration
SEEDS=(678 681 774 789)
SOURCE="Houston18"
TARGET="Houston13"
INTERNAL_MODELS=("GAHT" "BiDA" "AgentBiDA" "SelfAttentionAgentBiDA")
EXTERNAL_METHODS=("PCADA" "TSTnet" "MDGTnet" "CLDA" "SCLUDA" "SSWADA" "CACL" "MLUDA")

echo "Starting Ablation Study ($SOURCE -> $TARGET)..."
export ABLATION_DIR="ablation_results_H18_H13"
mkdir -p $ABLATION_DIR

# 1. Run Internal Models
for MODEL in "${INTERNAL_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "============================================="
        echo "Running Internal Model: $MODEL | Seed: $SEED"
        echo "============================================="
        python main.py --model $MODEL --source_name $SOURCE --target_name $TARGET --seed $SEED
        mv ablation_results/${MODEL}_results_seed_${SEED}.json $ABLATION_DIR/ 2>/dev/null
    done
done

# 2. Run External Models
for EXT in "${EXTERNAL_METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "============================================="
        echo "Running External Model: $EXT | Seed: $SEED"
        echo "============================================="
        
        pushd external_methods/$EXT > /dev/null
        
        # Explicit mapping for known entry points
        TRAIN_SCRIPT=""
        case "$EXT" in
            "PCADA") TRAIN_SCRIPT="train_pcada_houston.py" ;;
            "TSTnet") TRAIN_SCRIPT="train_tstnet.py" ;;
            "MDGTnet") TRAIN_SCRIPT="train_H1318_com_cls.py" ;;
            "CLDA") TRAIN_SCRIPT="CLDA_HOUSTON13_2_18.py" ;;  # CLDA only has 1 script, rely on --source_name
            "SCLUDA") TRAIN_SCRIPT="SCLUDA_Houston.py" ;;
            "SSWADA") TRAIN_SCRIPT="main.py" ;;
            "CACL") TRAIN_SCRIPT="demo_multiDA.py" ;;
            "MLUDA") TRAIN_SCRIPT="MLUDA_hu.py" ;;
        esac
        
        if [ -n "$TRAIN_SCRIPT" ] && [ -f "$TRAIN_SCRIPT" ]; then
            python "$TRAIN_SCRIPT" --seed $SEED --source_name $SOURCE --target_name $TARGET
        else
            echo "Warning: No training script found for $EXT ($TRAIN_SCRIPT)"
        fi
        
        popd > /dev/null
        
        mv ablation_results/${EXT}_results_seed_${SEED}.json $ABLATION_DIR/ 2>/dev/null
    done
done

echo "All training completed!"
echo "Generating Final Ablation Table for H18 -> H13..."
python generate_table_H18_to_H13.py
