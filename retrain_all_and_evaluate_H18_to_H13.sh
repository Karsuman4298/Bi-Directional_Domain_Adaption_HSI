#!/bin/bash

# Global Hyperparameters for fair comparison
EPOCHS=200
BATCH_SIZE=128
PATCH_SIZE=13
SEEDS="2100 2101 2102 2103"
SOURCE="Houston18"
TARGET="Houston13"

# AgentBiDA optimized hyperparameters
DEPTH=3
NUM_AGENTS=5
NUM_HEADS=8

echo "========================================================="
echo " MASTER RETRAINING SCRIPT: $SOURCE -> $TARGET"
echo "========================================================="

for SEED in $SEEDS; do
    echo "========================================"
    echo " Training SEED: $SEED"
    echo "========================================"

    # 1. Train Baseline Models & BiDA
    BASELINES="GAHT cnn3d ablstm dffn m3ddcnn rssan speformer ssftt BiDA"
    for MODEL in $BASELINES; do
        echo "--> Training $MODEL (Seed $SEED)"
        python3 main.py --model $MODEL --source_name $SOURCE --target_name $TARGET \
            --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE --depth $DEPTH --seed $SEED
    done

    # 2. Train SelfAttnAgentBiDA
    echo "--> Training SelfAttnAgentBiDA (Seed $SEED)"
    python3 train_self_attn_agent_bida.py --source_name $SOURCE --target_name $TARGET \
        --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE --depth $DEPTH --seed $SEED

    # 3. Train AgentBiDA
    echo "--> Training AgentBiDA (Seed $SEED)"
    python3 train_agent_bida.py --source_name $SOURCE --target_name $TARGET \
        --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE --depth $DEPTH \
        --num_agents $NUM_AGENTS --num_heads $NUM_HEADS --seed $SEED

done

echo "========================================================="
echo " ALL TRAINING COMPLETE. GENERATING ABLATION TABLE..."
echo "========================================================="

# The full suite of models to evaluate
ALL_MODELS="GAHT cnn3d ablstm dffn m3ddcnn rssan speformer ssftt BiDA SelfAttnAgentBiDA AgentBiDA"

python3 generate_paper_visualizations.py \
    --source_name $SOURCE \
    --target_name $TARGET \
    --output_dir ./paper_visualizations_retrained_H18_to_H13 \
    --models $ALL_MODELS \
    --epoch $EPOCHS \
    --bs $BATCH_SIZE \
    --patch_size $PATCH_SIZE \
    --depth $DEPTH \
    --num_agents $NUM_AGENTS \
    --num_heads $NUM_HEADS \
    --seeds $SEEDS \
    --device 0 \
    --no_vis

echo "DONE! The freshly trained ablation table has been printed above and LaTeX is generated."
