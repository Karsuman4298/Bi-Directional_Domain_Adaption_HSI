#!/bin/bash

# Configuration
SOURCE="Houston13"
TARGET="Houston18"
EPOCHS=200
SEEDS=(2100 2101 2102 2103)
HEADS=8
AGENTS=(2 4 8 16 32)

echo "=================================================================================="
echo " Ablation Study: Number of Agents (Houston13 -> Houston18)"
echo " Model: Cross-Domain AgentBiDA (4 Seeds: ${SEEDS[*]})"
echo "=================================================================================="
printf "%-10s | %-20s | %-20s | %-20s\n" "Agents" "OA (%)" "AA (%)" "Kappa"
echo "----------------------------------------------------------------------------------"

# Function to calculate mean and std dev using awk
calc_mean_std() {
    awk '{
        sum += $1;
        sumsq += ($1 * $1);
    } END {
        if (NR > 0) {
            mean = sum / NR;
            if (NR > 1) {
                # Sample standard deviation
                std = sqrt((sumsq - (sum * sum / NR)) / (NR - 1));
            } else {
                std = 0;
            }
            printf "%.4f ± %.4f", mean, std;
        } else {
            print "Failed";
        }
    }'
}

for NUM_AGENTS in "${AGENTS[@]}"; do
    OA_LIST=""
    AA_LIST=""
    KAPPA_LIST=""
    
    # Run across all seeds
    for SEED in "${SEEDS[@]}"; do
        # Run the training script and capture the output
        OUTPUT=$(python3 train_self_attn_agent_bida.py \
            --source_name $SOURCE \
            --target_name $TARGET \
            --epoch $EPOCHS \
            --seed $SEED \
            --num_agents $NUM_AGENTS \
            --num_heads $HEADS 2>&1)
        
        # Extract the metrics from the output
        OA=$(echo "$OUTPUT" | grep "OA:" | awk '{print $2}')
        AA=$(echo "$OUTPUT" | grep "AA:" | awk '{print $2}')
        KAPPA=$(echo "$OUTPUT" | grep "Kappa:" | awk '{print $2}')
        
        if [ ! -z "$OA" ]; then
            OA_LIST="$OA_LIST $OA"
            AA_LIST="$AA_LIST $AA"
            KAPPA_LIST="$KAPPA_LIST $KAPPA"
        fi
    done
    
    # Calculate means and std devs across the gathered seeds
    OA_STATS=$(echo "$OA_LIST" | tr ' ' '\n' | grep -v '^$' | calc_mean_std)
    AA_STATS=$(echo "$AA_LIST" | tr ' ' '\n' | grep -v '^$' | calc_mean_std)
    KAPPA_STATS=$(echo "$KAPPA_LIST" | tr ' ' '\n' | grep -v '^$' | calc_mean_std)
    
    # Print the aggregated row for this agent configuration
    printf "%-10s | %-20s | %-20s | %-20s\n" "$NUM_AGENTS" "$OA_STATS" "$AA_STATS" "$KAPPA_STATS"
done

echo "=================================================================================="
