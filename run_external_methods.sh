#!/bin/bash
SEEDS="2100 2101 2102"
EPOCHS=100

echo "================================================="
echo "Running External Methods for Houston 13 -> 18"
echo "================================================="

# 1. TSTnet
echo "--> Starting TSTnet"
cd external_methods/TSTnet
for SEED in $SEEDS; do
    python3 train_tstnet.py --seed $SEED --num_epoch $EPOCHS
done
cd ../..

# 2. PCADA
echo "--> Starting PCADA"
cd external_methods/PCADA
for SEED in $SEEDS; do
    python3 train_pcada_houston.py --seed $SEED --epochs $EPOCHS
done
cd ../..

# 3. CLDA
echo "--> Starting CLDA"
cd external_methods/CLDA
for SEED in $SEEDS; do
    python3 CLDA_HOUSTON13_2_18.py --seed $SEED --epochs $EPOCHS
done
cd ../..

# 4. MDGTnet
echo "--> Starting MDGTnet"
cd external_methods/MDGTnet
for SEED in $SEEDS; do
    python3 train_mdgtnet_houston.py --seed $SEED --epochs $EPOCHS
done
cd ../..

# 5. SCLUDA
echo "--> Starting SCLUDA"
cd external_methods/SCLUDA
for SEED in $SEEDS; do
    python3 SCLUDA_Houston.py --seed $SEED
done
cd ../..

# 6. MLUDA
echo "--> Starting MLUDA"
cd external_methods/MLUDA
for SEED in $SEEDS; do
    python3 MLUDA_hu.py --seed $SEED
done
cd ../..

# 7. SSWADA
echo "--> Starting SSWADA"
cd external_methods/SSWADA
for SEED in $SEEDS; do
    python3 main.py --seed $SEED --num_epoch $EPOCHS
done
cd ../..

echo "================================================="
echo "All external methods completed!"
echo "Run 'python3 generate_ablation_table.py' to view all combined results."
echo "================================================="
