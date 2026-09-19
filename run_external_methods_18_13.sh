#!/bin/bash
SEEDS="2100 2101 2102"
EPOCHS=100
SOURCE="Houston18"
TARGET="Houston13"

echo "================================================="
echo "Running External Methods for Houston 18 -> 13"
echo "================================================="

# Save the root project directory
ROOT_DIR=$(pwd)

# 1. TSTnet (takes --source_name and --target_name)
echo "--> Starting TSTnet"
cd "$ROOT_DIR/external_methods/TSTnet"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 train_tstnet.py --seed $SEED --num_epoch $EPOCHS --source_name $SOURCE --target_name $TARGET
done

# 2. PCADA (uses our new _18_2_13 copy)
echo "--> Starting PCADA"
cd "$ROOT_DIR/external_methods/PCADA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 train_pcada_houston_18_2_13.py --seed $SEED --epochs $EPOCHS
done

# 3. CLDA (uses our new _18_2_13 copy)
echo "--> Starting CLDA"
cd "$ROOT_DIR/external_methods/CLDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 CLDA_HOUSTON18_2_13.py --seed $SEED --epochs $EPOCHS
done

# 4. MDGTnet (takes --source_name and --target_name)
echo "--> Starting MDGTnet"
cd "$ROOT_DIR/external_methods/MDGTnet"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 train_mdgtnet_houston.py --seed $SEED --epochs $EPOCHS --source_name $SOURCE --target_name $TARGET
done

# 5. SCLUDA (uses our new _18_2_13 copy)
echo "--> Starting SCLUDA"
cd "$ROOT_DIR/external_methods/SCLUDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 SCLUDA_Houston_18_2_13.py --seed $SEED --epochs $EPOCHS
done

# 6. MLUDA (uses our new _18_2_13 copy)
echo "--> Starting MLUDA"
cd "$ROOT_DIR/external_methods/MLUDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 MLUDA_hu_18_2_13.py --seed $SEED --epochs $EPOCHS
done

# 7. SSWADA (uses our new _18_2_13 copy)
echo "--> Starting SSWADA"
cd "$ROOT_DIR/external_methods/SSWADA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 main_18_2_13.py --seed $SEED --num_epoch $EPOCHS
done

# 8. CACL (takes --source_name and --target_name)
echo "--> Starting CACL"
cd "$ROOT_DIR/external_methods/CACL"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 demo_singleDA.py --dataset M_Houston --source_name $SOURCE --target_name $TARGET --in_channel 48 --seed $SEED --epochs $EPOCHS
done

cd "$ROOT_DIR"
echo "================================================="
echo "All external methods completed for $SOURCE -> $TARGET!"
echo "================================================="
