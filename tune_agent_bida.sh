#!/bin/bash

# Setup Dataset and Baseline Configs
SOURCE="Houston13"
TARGET="Houston18"
PATCH_SIZE=13
NUM_TOKENS=4
LR=1e-2
HEADS=8

# Create a directory to store all the tuning logs
mkdir -p tuning_logs

# Define the Hyperparameter Grid to search for the best model
EPOCHS_LIST=(200 300 400)
AGENTS_LIST=(4 8 16)

echo "=========================================================="
echo " Starting AgentBiDA Hyperparameter Tuning Grid Search "
echo "=========================================================="
echo "Running on Source: $SOURCE -> Target: $TARGET"
echo ""

for E in "${EPOCHS_LIST[@]}"; do
    for A in "${AGENTS_LIST[@]}"; do
        LOG_FILE="tuning_logs/AgentBiDA_E${E}_A${A}.log"
        echo "-> Running Configuration: Epochs=$E, Agents=$A"
        echo "   Logs are being saved to $LOG_FILE ..."
        
        # Run the training script
        python train_agent_bida.py \
            --source_name $SOURCE \
            --target_name $TARGET \
            --epoch $E \
            --patch_size $PATCH_SIZE \
            --num_tokens $NUM_TOKENS \
            --lr $LR \
            --num_agents $A \
            --num_heads $HEADS > "$LOG_FILE" 2>&1
            
        echo "   [COMPLETED] Configuration: Epochs=$E, Agents=$A"
        echo "----------------------------------------------------------"
    done
done

echo "Tuning Complete!"
echo "Check the 'tuning_logs/' directory to find the configuration with the highest accuracy."
