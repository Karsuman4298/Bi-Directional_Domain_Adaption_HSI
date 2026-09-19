#!/bin/bash
EPOCHS=100
SEEDS=(2100 2101 2102)
SOURCE="Houston13"
TARGET="Houston18"
NUM_WORKERS=2
NUM_AGENTS=2

echo "Running ${EPOCHS}-epoch full evaluation for models..."
for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "SEED: $SEED"
    echo "=========================================="
    
    echo "Running AgentBiDA (Fixed)..."
    python3 train_agent_bida_fix.py --model AgentBiDA --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_agents $NUM_AGENTS --num_workers $NUM_WORKERS | grep -E "Epoch:|Accuracy|OA|AA|Kappa" | tail -n 5
    
    echo "Running SelfAttentionAgentBiDA (Fixed)..."
    python3 train_self_attn_agent_bida_fix.py --model SelfAttentionAgentBiDA --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_agents $NUM_AGENTS --num_workers $NUM_WORKERS | grep -E "Epoch:|Accuracy|OA|AA|Kappa" | tail -n 5

    echo "Running Original BiDA..."
    python3 main.py --model BiDA --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_workers $NUM_WORKERS | grep -E "Epoch:|Accuracy|OA|AA|Kappa" | tail -n 5
done

python3 aggregate_evaluation.py
