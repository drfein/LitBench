#!/bin/bash

# Script to train a generative reward model using verl and a fixed target effective batch size

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

# Set per-device batch size to 2 (maximum each GPU can handle)
PER_DEVICE_BATCH_SIZE=2

# Calculate total batch size per step across all GPUs
TOTAL_PER_STEP=$((PER_DEVICE_BATCH_SIZE * NUM_GPUS))

# Calculate gradient accumulation steps needed to reach target batch size
GRAD_ACCUM_STEPS=$((TARGET_BATCH_SIZE / TOTAL_PER_STEP))

# If division isn't even, round up
if [ $((TOTAL_PER_STEP * GRAD_ACCUM_STEPS)) -lt $TARGET_BATCH_SIZE ]; then
    GRAD_ACCUM_STEPS=$((GRAD_ACCUM_STEPS + 1))
fi

# Calculate the actual effective batch size
EFFECTIVE_BATCH_SIZE=$((TOTAL_PER_STEP * GRAD_ACCUM_STEPS))

# Base model to use (change as needed)
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME="SAA-Lab/LitBench-Rationales"

# Set output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/nlp/scr/drfein/genrm_model_${TIMESTAMP}"

# Number of epochs
EPOCHS=3

# Learning rate
LEARNING_RATE=2e-5

# Maximum sequence length - using a conservative value
MAX_LENGTH=8192

# Flash attention - explicitly disabled
USE_FLASH_ATTENTION=false

echo "Configuration:"
echo "- Model: $MODEL_NAME"
echo "- Dataset: $DATASET_NAME"
echo "- Per-device batch size: $PER_DEVICE_BATCH_SIZE"
echo "- Number of GPUs: $NUM_GPUS"
echo "- Total batch size per step: $TOTAL_PER_STEP"
echo "- Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "- Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "- Number of epochs: $EPOCHS"
echo "- Learning rate: $LEARNING_RATE"
echo "- Max sequence length: $MAX_LENGTH"
echo "- Flash attention: $USE_FLASH_ATTENTION"
echo "- Output directory: $OUTPUT_DIR"

# Set HF_TOKEN environment variable if not already set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set. You may need to set it for model access."
fi

# Run the training script with multi-GPU support via accelerate
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using accelerate launch for multi-GPU training with $NUM_GPUS GPUs"
    accelerate launch --num_processes=$NUM_GPUS scripts/train_GenRM.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size $PER_DEVICE_BATCH_SIZE \
        --grad_accum $GRAD_ACCUM_STEPS \
        --max_length $MAX_LENGTH \
        --learning_rate $LEARNING_RATE \
        --use_flash_attention $USE_FLASH_ATTENTION
else
    echo "Running on a single device"
    python scripts/train_GenRM.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size $PER_DEVICE_BATCH_SIZE \
        --grad_accum $GRAD_ACCUM_STEPS \
        --max_length $MAX_LENGTH \
        --learning_rate $LEARNING_RATE \
        --use_flash_attention $USE_FLASH_ATTENTION
fi

echo "Training complete. Model saved to $OUTPUT_DIR" 