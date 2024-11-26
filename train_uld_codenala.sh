#!/bin/bash

# Define default argument values
STUDENT_NAME="WizardLMTeam/WizardCoder-Python-13B-V1.0"       # Replace with your student model
TEACHER_NAME="codeparrot/codeparrot-small"          # Replace with your teacher model
TEMPERATURE=1.0                  # Temperature for scaling logits
LAMBDA_ULD=1.5                   # Weight for the Wasserstein loss term
BATCH_SIZE=2                     # Batch size
NUM_EPOCHS=3                     # Number of epochs
LEARNING_RATE=5e-5               # Learning rate
LOAD_IN_4BIT="--load_in_4bit"    # Whether to use 4-bit precision
DATASET_PATH="/home/brytech/datasets/conala/train.parquet" # Path to your training dataset
SAVE_DIR="/home/brytech/train_uld_run/output_weights"        # Directory to save model weights

# Ensure the output directory exists
mkdir -p $SAVE_DIR

# Run the Python training script
python train_uld.py \
    --student_name "$STUDENT_NAME" \
    --teacher_name "$TEACHER_NAME" \
    --temperature "$TEMPERATURE" \
    --lambda_uld "$LAMBDA_ULD" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    $LOAD_IN_4BIT \
    --dataset_path "$DATASET_PATH" \
    --save_dir "$SAVE_DIR"