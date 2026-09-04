#!/bin/bash

# =================================================================================
# Master Script for Research Paper Visualization and Ablation
# =================================================================================
# This script runs the complete suite of comparative analyses across all models.
# It enforces identical batch size, epochs, and patch sizes for a fair comparison,
# and incorporates the optimized hyperparameters specifically for AgentBiDA.
# =================================================================================

# Global Hyperparameters for fair comparison
EPOCHS=200
BATCH_SIZE=128
PATCH_SIZE=13
SEEDS="2100 2101 2102 2103 "

# AgentBiDA optimized hyperparameters
DEPTH=3
NUM_AGENTS=2
NUM_HEADS=8

# Models to evaluate
MODELS="GAHT cnn3d ablstm dffn m3ddcnn rssan speformer ssftt BiDA AgentBiDA"

echo "Starting full experimental suite..."

# ---------------------------------------------------------------------------------
# Run 1: Source: Houston13 -> Target: Houston18
# ---------------------------------------------------------------------------------
echo "========================================================="
echo " RUN 1: Houston13 (Source) -> Houston18 (Target)"
echo "========================================================="
python3 generate_paper_visualizations.py \
    --source_name Houston13 \
    --target_name Houston18 \
    --output_dir ./paper_visualizations_H13_to_H18 \
    --models $MODELS \
    --epoch $EPOCHS \
    --bs $BATCH_SIZE \
    --patch_size $PATCH_SIZE \
    --depth $DEPTH \
    --num_agents $NUM_AGENTS \
    --num_heads $NUM_HEADS \
    --seeds $SEEDS

# ---------------------------------------------------------------------------------
# Run 2: Source: Houston18 -> Target: Houston13
# ---------------------------------------------------------------------------------
echo "========================================================="
echo " RUN 2: Houston18 (Source) -> Houston13 (Target)"
echo "========================================================="
python3 generate_paper_visualizations.py \
    --source_name Houston18 \
    --target_name Houston13 \
    --output_dir ./paper_visualizations_H18_to_H13 \
    --models $MODELS \
    --epoch $EPOCHS \
    --bs $BATCH_SIZE \
    --patch_size $PATCH_SIZE \
    --depth $DEPTH \
    --num_agents $NUM_AGENTS \
    --num_heads $NUM_HEADS \
    --seeds $SEEDS

echo "========================================================="
echo " All experiments completed! Tables and images are located in:"
echo " - ./paper_visualizations_H13_to_H18/"
echo " - ./paper_visualizations_H18_to_H13/"
echo "========================================================="
