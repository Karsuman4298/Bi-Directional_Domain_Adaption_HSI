#!/bin/bash
EPOCHS=1
SEEDS=(2100 2101 2102)
SOURCE="Houston13"
TARGET="Houston18"

echo "Running 1-epoch smoke test for fixed models..."
for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "SEED: $SEED"
    echo "=========================================="
    
    echo "Running AgentBiDA (Fixed)..."
    python3 train_agent_bida_fix.py --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_agents 2 | grep -E "Epoch:|Accuracy|OA|AA|Kappa" | tail -n 5
    
    echo "Running SelfAttentionAgentBiDA (Fixed)..."
    python3 train_self_attn_agent_bida_fix.py --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_agents 2 | grep -E "Epoch:|Accuracy|OA|AA|Kappa" | tail -n 5
done
