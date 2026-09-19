#!/bin/bash
set -e

# =========================================================================
# Fix environments (only runs if the package isn't already installed properly)
# =========================================================================
echo "Checking dependencies..."
pip install cleanlab==1.0.1 opencv-contrib-python > /dev/null 2>&1

ROOT_DIR="$(pwd)"
EPOCHS=100
SEEDS="2100 2101 2102"

echo "================================================="
echo "Running Remaining External Methods for Houston 13 -> 18"
echo "================================================="

# 3. CLDA
echo "--> Starting CLDA"
cd "$ROOT_DIR/external_methods/CLDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 CLDA_HOUSTON13_2_18.py --seed $SEED --epochs $EPOCHS
done

# 4. MDGTnet
echo "--> Starting MDGTnet"
cd "$ROOT_DIR/external_methods/MDGTnet"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 train_mdgtnet_houston.py --seed $SEED --num_epoch $EPOCHS
done

# 5. SCLUDA
echo "--> Starting SCLUDA"
cd "$ROOT_DIR/external_methods/SCLUDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 SCLUDA_Houston.py --seed $SEED --epochs $EPOCHS
done

# 6. MLUDA
echo "--> Starting MLUDA"
cd "$ROOT_DIR/external_methods/MLUDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 MLUDA_hu.py --seed $SEED --epochs $EPOCHS
done

# 7. SSWADA
echo "--> Starting SSWADA"
cd "$ROOT_DIR/external_methods/SSWADA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 main.py --seed $SEED --num_epoch $EPOCHS
done

# 8. CACL
echo "--> Starting CACL"
cd "$ROOT_DIR/external_methods/CACL"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 demo_singleDA.py --dataset M_Houston --source_name Houston13 --target_name Houston18 --in_channel 48 --seed $SEED --epochs $EPOCHS
done

cd "$ROOT_DIR"
echo "================================================="
echo "All remaining external methods completed!"
echo "Run 'python3 generate_ablation_table.py' to view all combined results."
echo "================================================="
