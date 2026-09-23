#!/bin/bash

# Exit on error
set -e

# Experiment parameters
MODELS=("BiDA" "AgentBiDA" "SelfAttentionAgentBiDA")
SEEDS=(2100 2101 2102)
OPTIMIZERS=("Adam" "SGD")
EPOCHS=120  # Keeping it at 120 for a quick test as requested

echo "==================================================="
echo "Starting Optimizer Comparison Experiment..."
echo "Models: ${MODELS[*]}"
echo "Optimizers: ${OPTIMIZERS[*]}"
echo "==================================================="

for opt in "${OPTIMIZERS[@]}"; do
    # This environment variable tells scheduler.py which optimizer to load
    export BIDA_OPTIMIZER=$opt
    
    # Set Learning Rate appropriately based on optimizer
    # (Adam requires a smaller learning rate than SGD)
    if [ "$opt" == "Adam" ]; then
        LR=0.001
    else
        LR=0.01
    fi

    for model in "${MODELS[@]}"; do
        
        # Determine the correct entry script
        if [ "$model" == "BiDA" ]; then
            SCRIPT="main.py"
            # BiDA does not take num_agents, so we omit it
            EXTRA_ARGS=""
        elif [ "$model" == "AgentBiDA" ]; then
            SCRIPT="train_agent_bida_fix.py"
            EXTRA_ARGS="--num_agents 4"
        elif [ "$model" == "SelfAttentionAgentBiDA" ]; then
            SCRIPT="train_self_attn_agent_bida_fix.py"
            EXTRA_ARGS="--num_agents 4"
        fi

        for seed in "${SEEDS[@]}"; do
            LOG_FILE="optimizer_test_18to13_${model}_${opt}_${seed}.log"
            echo "Running $model with $opt (LR=$LR) on Seed $seed..."
            echo "Logs saving to -> $LOG_FILE"
            
            CUDA_VISIBLE_DEVICES=0 python3 -u $SCRIPT \
                --model $model \
                --source_name Houston18 \
                --target_name Houston13 \
                --seed $seed \
                --epoch $EPOCHS \
                --lr $LR \
                --num_workers 2 \
                $EXTRA_ARGS > "$LOG_FILE" 2>&1
                
        done
    done
done

echo "==================================================="
echo "All experiments finished!"
echo "Use the following command to check the best accuracies:"
echo "grep -H 'best acc all' optimizer_test_*.log"
echo "==================================================="
