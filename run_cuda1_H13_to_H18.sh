#!/bin/bash

# Global Hyperparameters for fair comparison
EPOCHS=200
BATCH_SIZE=128
PATCH_SIZE=13
SEEDS="2100 2101 2102 2103 "

# AgentBiDA optimized hyperparameters
DEPTH=3
NUM_AGENTS=5
NUM_HEADS=8

# Models to evaluate
MODELS="SelfAttnAgentBiDA AgentBiDA BiDA GAHT cnn3d ablstm dffn m3ddcnn rssan speformer ssftt"

echo "========================================================="
echo " RUN 2: Houston13 (Source) -> Houston18 (Target) on CUDA 1"
echo "========================================================="
python3 generate_paper_visualizations.py \
    --source_name Houston13 \
    --target_name Houston18 \
    --output_dir ./paper_visualizations_SelfAttnAgent_H13_to_H18 \
    --models $MODELS \
    --epoch $EPOCHS \
    --bs $BATCH_SIZE \
    --patch_size $PATCH_SIZE \
    --depth $DEPTH \
    --num_agents $NUM_AGENTS \
    --num_heads $NUM_HEADS \
    --seeds $SEEDS \
    --device 1 \
    --no_vis
