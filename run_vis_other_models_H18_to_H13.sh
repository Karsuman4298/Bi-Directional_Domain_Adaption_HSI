#!/bin/bash

# Global Hyperparameters for fair comparison
EPOCHS=200
BATCH_SIZE=128
PATCH_SIZE=13
SEEDS="2100 2101 2102 2103"

# AgentBiDA optimized hyperparameters (still passed, but unused by most models)
DEPTH=3
NUM_AGENTS=5
NUM_HEADS=8

# Models to evaluate (Excluding AgentBiDA and SelfAttnAgentBiDA as requested)
MODELS="BiDA GAHT cnn3d ablstm dffn m3ddcnn rssan speformer ssftt"

echo "========================================================="
echo " RUN: Houston18 (Source) -> Houston13 (Target) on CUDA 0"
echo " Evaluating all models except AgentBiDA variants"
echo "========================================================="
python3 generate_paper_visualizations.py \
    --source_name Houston18 \
    --target_name Houston13 \
    --output_dir ./paper_visualizations_other_models_H18_to_H13 \
    --models $MODELS \
    --epoch $EPOCHS \
    --bs $BATCH_SIZE \
    --patch_size $PATCH_SIZE \
    --depth $DEPTH \
    --num_agents $NUM_AGENTS \
    --num_heads $NUM_HEADS \
    --seeds $SEEDS \
    --device 0 \
    --no_vis
