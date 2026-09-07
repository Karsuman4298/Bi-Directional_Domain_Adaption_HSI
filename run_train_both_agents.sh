#!/bin/bash

# =================================================================================
# Train AgentBiDA & SelfAttnAgentBiDA on BOTH directions, 4 seeds each
# Then generate the final scores table
# =================================================================================

EPOCHS=200
BATCH_SIZE=128
PATCH_SIZE=13
SEEDS="2100 2101 2102 2103"
DEPTH=3
NUM_AGENTS=5
NUM_HEADS=8

# ---------------------------------------------------------------------------------
# Direction 1: Houston13 -> Houston18
# ---------------------------------------------------------------------------------
echo "=========================================================================="
echo " DIRECTION 1: Houston13 -> Houston18"
echo "=========================================================================="

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    echo "--> Training SelfAttnAgentBiDA (H13->H18, seed=$SEED)"
    python3 train_self_attn_agent_bida.py \
        --source_name Houston13 --target_name Houston18 \
        --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
        --depth $DEPTH --num_agents $NUM_AGENTS --num_heads $NUM_HEADS \
        --seed $SEED

    echo "--> Training AgentBiDA (H13->H18, seed=$SEED)"
    python3 train_agent_bida.py \
        --source_name Houston13 --target_name Houston18 \
        --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
        --depth $DEPTH --num_agents $NUM_AGENTS --num_heads $NUM_HEADS \
        --seed $SEED
done

echo ""
echo "========== Generating H13->H18 Scores =========="
python3 generate_paper_visualizations.py \
    --source_name Houston13 --target_name Houston18 \
    --output_dir ./paper_visualizations_agents_H13_to_H18 \
    --models SelfAttnAgentBiDA AgentBiDA \
    --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
    --depth $DEPTH --num_agents $NUM_AGENTS --num_heads $NUM_HEADS \
    --seeds "$SEEDS" --device 0 --no_vis

# ---------------------------------------------------------------------------------
# Direction 2: Houston18 -> Houston13
# ---------------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo " DIRECTION 2: Houston18 -> Houston13"
echo "=========================================================================="

for SEED in $SEEDS; do
    echo "--- Seed $SEED ---"

    echo "--> Training SelfAttnAgentBiDA (H18->H13, seed=$SEED)"
    python3 train_self_attn_agent_bida.py \
        --source_name Houston18 --target_name Houston13 \
        --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
        --depth $DEPTH --num_agents $NUM_AGENTS --num_heads $NUM_HEADS \
        --seed $SEED

    echo "--> Training AgentBiDA (H18->H13, seed=$SEED)"
    python3 train_agent_bida.py \
        --source_name Houston18 --target_name Houston13 \
        --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
        --depth $DEPTH --num_agents $NUM_AGENTS --num_heads $NUM_HEADS \
        --seed $SEED
done

echo ""
echo "========== Generating H18->H13 Scores =========="
python3 generate_paper_visualizations.py \
    --source_name Houston18 --target_name Houston13 \
    --output_dir ./paper_visualizations_agents_H18_to_H13 \
    --models SelfAttnAgentBiDA AgentBiDA \
    --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
    --depth $DEPTH --num_agents $NUM_AGENTS --num_heads $NUM_HEADS \
    --seeds "$SEEDS" --device 0 --no_vis

echo ""
echo "=========================================================================="
echo " ALL DONE! Results saved in:"
echo "   ./paper_visualizations_agents_H13_to_H18/"
echo "   ./paper_visualizations_agents_H18_to_H13/"
echo "=========================================================================="
