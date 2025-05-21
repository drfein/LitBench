#!/bin/bash

# Script to train a Bradley-Terry reward model with a fixed target effective batch size

# Target effective batch size
TARGET_BATCH_SIZE=128

# Check if CUDA is available and count GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $NUM_GPUS GPUs"
else
    echo "No GPUs found, defaulting to CPU mode"
    NUM_GPUS=1
fi

# Set a reasonable total batch size that works well with the available GPUs
# This is the total batch size, not per-GPU
BATCH_SIZE=8

# Calculate gradient accumulation steps to reach target effective batch size
GRAD_ACCUM_STEPS=$((TARGET_BATCH_SIZE / BATCH_SIZE))

# If batch size doesn't divide evenly, adjust gradient accumulation steps
if [ $((BATCH_SIZE * GRAD_ACCUM_STEPS)) -lt $TARGET_BATCH_SIZE ]; then
    GRAD_ACCUM_STEPS=$((GRAD_ACCUM_STEPS + 1))
fi

# Calculate the actual effective batch size 
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS))

# Estimate total training steps for one epoch
DATASET_SIZE=43827  # LitBench-Train dataset size
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH_SIZE))
if [ $((DATASET_SIZE % EFFECTIVE_BATCH_SIZE)) -ne 0 ]; then
    STEPS_PER_EPOCH=$((STEPS_PER_EPOCH + 1))
fi

echo "Configuration:"
echo "- Total batch size: $BATCH_SIZE"
echo "- Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "- Number of GPUs: $NUM_GPUS"
echo "- Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "- Dataset size: $DATASET_SIZE"
echo "- Expected gradient steps per epoch: $STEPS_PER_EPOCH"

# Base model to use (change as needed)
BASE_MODEL="meta-llama/Llama-3.2-1B"

# Set output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/nlp/scr/drfein/bt_reward_model_${TIMESTAMP}"

# Make sure data directory exists
DATA_DIR="./data"
mkdir -p $DATA_DIR

# Set HF_TOKEN environment variable if not already set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set. You may need to set it for model access."
fi

# Run the training script
python3 train_BT.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --epochs 1 \
    --max_length 2048 \
    --learning_rate 1e-5 \
    --keep_checkpoints 3

echo "Training complete. Model saved to $OUTPUT_DIR"