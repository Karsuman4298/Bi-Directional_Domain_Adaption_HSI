#!/bin/bash

# =================================================================================
# Hyperparameter Ablation Script for AgentBiDA & SelfAttnAgentBiDA
# =================================================================================
# Tests different configurations to find optimal performance.
# Run ONE direction at a time. Change SOURCE/TARGET to flip direction.
# =================================================================================

SOURCE="Houston18"
TARGET="Houston13"
EPOCHS=200
BATCH_SIZE=128
PATCH_SIZE=13
SEEDS="2100 2101"  # Use 2 seeds for quick ablation; expand to 4 for final

# Base config (control)
BASE_DEPTH=3
BASE_AGENTS=5
BASE_HEADS=8

echo "=========================================================================="
echo " Hyperparameter Ablation: AgentBiDA ($SOURCE -> $TARGET)"
echo "=========================================================================="

# ---------------------------------------------------------------------------------
# Experiment 1: Vary num_agents (THE MOST IMPORTANT ONE)
# Current: 5 (= sequence length, so NO compression)
# Try: 3 (real compression), 2 (aggressive compression)
# ---------------------------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 1: Number of Agents ==="
for NUM_AGENTS in 2 3 5; do
    for SEED in $SEEDS; do
        echo "--> AgentBiDA | agents=$NUM_AGENTS | seed=$SEED"
        python3 train_agent_bida.py \
            --source_name $SOURCE --target_name $TARGET \
            --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
            --depth $BASE_DEPTH --num_agents $NUM_AGENTS --num_heads $BASE_HEADS \
            --seed $SEED
    done
done

# ---------------------------------------------------------------------------------
# Experiment 2: Reduce Regularization (dropout)
# Current: drop_rate=0.1, attn_drop=0.1
# The models are small (dim=64, depth=3). High dropout may hurt.
# ---------------------------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 2: Reduced Dropout ==="
# This requires modifying the model code. We'll test by creating a wrapper.
# For now, we test via the lambda hyperparameters instead.

# ---------------------------------------------------------------------------------
# Experiment 3: Vary loss balance (lambda1 = alignment+distillation, lambda2 = consistency)
# Current: lambda1=0.1, lambda2=1.0
# Try: stronger alignment, weaker consistency
# ---------------------------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 3: Loss Balance ==="
for LAMBDA1 in 0.05 0.1 0.5; do
    for LAMBDA2 in 0.1 1.0; do
        for SEED in $SEEDS; do
            echo "--> AgentBiDA | lambda1=$LAMBDA1, lambda2=$LAMBDA2 | seed=$SEED"
            python3 train_agent_bida.py \
                --source_name $SOURCE --target_name $TARGET \
                --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
                --depth $BASE_DEPTH --num_agents 3 --num_heads $BASE_HEADS \
                --lambda1 $LAMBDA1 --lambda2 $LAMBDA2 \
                --seed $SEED
        done
    done
done

# ---------------------------------------------------------------------------------
# Experiment 4: Learning Rate
# Current: SGD lr=0.01, no scheduler
# Try: lower LR with cosine decay
# ---------------------------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 4: Learning Rate ==="
for LR in 0.005 0.01 0.02; do
    for SEED in $SEEDS; do
        echo "--> AgentBiDA | lr=$LR | seed=$SEED"
        python3 train_agent_bida.py \
            --source_name $SOURCE --target_name $TARGET \
            --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
            --depth $BASE_DEPTH --num_agents 3 --num_heads $BASE_HEADS \
            --lr $LR \
            --seed $SEED
    done
done

# ---------------------------------------------------------------------------------
# Experiment 5: Depth (number of transformer blocks)
# Current: 3
# ---------------------------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 5: Depth ==="
for DEPTH in 1 2 3 5; do
    for SEED in $SEEDS; do
        echo "--> AgentBiDA | depth=$DEPTH | seed=$SEED"
        python3 train_agent_bida.py \
            --source_name $SOURCE --target_name $TARGET \
            --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
            --depth $DEPTH --num_agents 3 --num_heads $BASE_HEADS \
            --seed $SEED
    done
done

# ---------------------------------------------------------------------------------
# Experiment 6: SelfAttnAgentBiDA with the same grid
# ---------------------------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 6: SelfAttnAgentBiDA Agent Count ==="
for NUM_AGENTS in 2 3 5; do
    for SEED in $SEEDS; do
        echo "--> SelfAttnAgentBiDA | agents=$NUM_AGENTS | seed=$SEED"
        python3 train_self_attn_agent_bida.py \
            --source_name $SOURCE --target_name $TARGET \
            --epoch $EPOCHS --bs $BATCH_SIZE --patch_size $PATCH_SIZE \
            --depth $BASE_DEPTH --num_agents $NUM_AGENTS --num_heads $BASE_HEADS \
            --seed $SEED
    done
done

echo "=========================================================================="
echo " ABLATION COMPLETE"
echo "=========================================================================="
