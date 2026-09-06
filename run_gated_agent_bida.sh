#!/bin/bash

# Configuration
EPOCHS=200
SEED=2100
HEADS=8
NUM_AGENTS=4

echo "=================================================================================="
echo " Training: GatedAgentBiDA (Houston13 -> Houston18)"
echo "=================================================================================="
python3 train_gated_agent_bida.py \
    --source_name Houston13 \
    --target_name Houston18 \
    --epoch $EPOCHS \
    --seed $SEED \
    --num_agents $NUM_AGENTS \
    --num_heads $HEADS \
    --gate_hidden_ratio 0.25 \
    --gate_init_bias 0.0

echo "=================================================================================="
echo " Training: GatedAgentBiDA (Houston18 -> Houston13)"
echo "=================================================================================="
python3 train_gated_agent_bida.py \
    --source_name Houston18 \
    --target_name Houston13 \
    --epoch $EPOCHS \
    --seed $SEED \
    --num_agents $NUM_AGENTS \
    --num_heads $HEADS \
    --gate_hidden_ratio 0.25 \
    --gate_init_bias 0.0
