#!/bin/bash
EPOCHS=100
SEEDS=(2100 2101 2102)
SOURCE="Houston18"
TARGET="Houston13"
NUM_WORKERS=2
NUM_AGENTS=2

echo "Running ${EPOCHS}-epoch full evaluation for internal models and GAHT..."
for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "SEED: $SEED"
    echo "=========================================="
    
    echo "Running AgentBiDA (Fixed)..."
    python3 train_agent_bida_fix.py --model AgentBiDA --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_agents $NUM_AGENTS --num_workers $NUM_WORKERS
    
    echo "Running SelfAttentionAgentBiDA (Fixed)..."
    python3 train_self_attn_agent_bida_fix.py --model SelfAttentionAgentBiDA --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_agents $NUM_AGENTS --num_workers $NUM_WORKERS

    echo "Running Original BiDA..."
    python3 main.py --model BiDA --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_workers $NUM_WORKERS

    echo "Running GAHT..."
    python3 main.py --model GAHT --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_workers $NUM_WORKERS
done

echo "================================================="
echo "Internal evaluations completed for $SOURCE -> $TARGET"
echo "================================================="
