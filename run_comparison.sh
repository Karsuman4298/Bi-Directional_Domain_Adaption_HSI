#!/bin/bash

# Configuration
EPOCHS=200
PATCH_SIZE=13
NUM_TOKENS=4
LR=1e-2
SOURCE="Houston13"
TARGET="Houston18"

echo "====================================================="
echo "Running BiDA_Agent (Previous Model)"
echo "====================================================="
python main.py --model BiDA_Agent \
    --source_name $SOURCE \
    --target_name $TARGET \
    --epoch $EPOCHS \
    --patch_size $PATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --lr $LR

echo "====================================================="
echo "Running AgentBiDA (New Independent Model)"
echo "====================================================="
python train_agent_bida.py \
    --source_name $SOURCE \
    --target_name $TARGET \
    --epoch $EPOCHS \
    --patch_size $PATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --lr $LR \
    --num_agents 4 \
    --num_heads 8
