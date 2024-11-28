#!/bin/bash

# Define default argument values
STUDENT_NAME="codellama/CodeLlama-7b-hf"       # Replace with your student model
TEACHER_NAME="WizardLMTeam/WizardCoder-Python-13B-V1.0"          # Replace with your teacher model
TEMPERATURE=2.0                  # Temperature for scaling logits
LAMBDA_ULD=0.00003123438281                  # Weight for the Wasserstein loss term
BATCH_SIZE=4                  # Batch size
NUM_EPOCHS=30                    # Number of epochs
LEARNING_RATE=1e-5               # Learning rate
LOAD_IN_4BIT="--load_in_4bit"    # Whether to use 4-bit precision
TRAIN_DATASET_PATH="/nobackup/users/brytech/promptified_codesearchnet/train7500.parquet"
VAL_DATASET_PATH="/nobackup/users/brytech/promptified_codesearchnet/test100.parquet"
SAVE_DIR="/nobackup/users/brytech/train_uld_run_codesearchnet/output_weights"    # Directory to save model weights

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
    --train_dataset_path "$TRAIN_DATASET_PATH" \
    --val_dataset_path "$VAL_DATASET_PATH" \
    --save_dir "$SAVE_DIR"
