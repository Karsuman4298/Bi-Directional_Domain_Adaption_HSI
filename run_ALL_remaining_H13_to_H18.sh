#!/bin/bash
set -e

ROOT_DIR="$(pwd)"
EPOCHS=100
SEEDS="2100 2101 2102"
SOURCE="Houston13"
TARGET="Houston18"
NUM_WORKERS=2

echo "================================================="
echo "Running ALL REMAINING Models for $SOURCE -> $TARGET"
echo "================================================="

# 1. Internal Baselines (GAHT and others)
INTERNAL_MODELS="GAHT cnn3d ablstm dffn m3ddcnn rssan speformer ssftt"

echo "--> Starting Internal Baselines: $INTERNAL_MODELS"
for MODEL in $INTERNAL_MODELS; do
    for SEED in $SEEDS; do
        echo "Running $MODEL (Seed: $SEED)..."
        python3 main.py --model $MODEL --source_name $SOURCE --target_name $TARGET --epoch $EPOCHS --seed $SEED --num_workers $NUM_WORKERS
    done
done

# 2. Remaining External Methods (CLDA, MDGTnet, SCLUDA, MLUDA, SSWADA, CACL)
# (TSTnet and PCADA are excluded as requested since they are already done)

echo "--> Starting CLDA"
cd "$ROOT_DIR/external_methods/CLDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 CLDA_HOUSTON13_2_18.py --seed $SEED --epochs $EPOCHS
done

echo "--> Starting MDGTnet"
cd "$ROOT_DIR/external_methods/MDGTnet"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 train_mdgtnet_houston.py --seed $SEED --num_epoch $EPOCHS
done

echo "--> Starting SCLUDA"
cd "$ROOT_DIR/external_methods/SCLUDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 SCLUDA_Houston.py --seed $SEED --epochs $EPOCHS
done

echo "--> Starting MLUDA"
cd "$ROOT_DIR/external_methods/MLUDA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 MLUDA_hu.py --seed $SEED --epochs $EPOCHS
done

echo "--> Starting SSWADA"
cd "$ROOT_DIR/external_methods/SSWADA"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 main.py --seed $SEED --num_epoch $EPOCHS
done

echo "--> Starting CACL"
cd "$ROOT_DIR/external_methods/CACL"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for SEED in $SEEDS; do
    python3 demo_singleDA.py --dataset M_Houston --source_name $SOURCE --target_name $TARGET --in_channel 48 --seed $SEED --epochs $EPOCHS
done

cd "$ROOT_DIR"
echo "================================================="
echo "All remaining models completed for $SOURCE -> $TARGET!"
echo "Run 'python3 generate_ablation_table.py' to view all combined results."
echo "================================================="
