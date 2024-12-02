#!/bin/bash

# Define default argument values
BASE_MODEL_PATH="codellama/CodeLlama-7b-hf"       # Path or name of the base model
CHECKPOINT_PATH="/nobackup/users/brytech/train_uld_run_codesearchnet/output_weights/codellama_lora/trained_on_7500/checkpoint-3"  # Path to checkpoint
OUTPUT_FILE="completions.jsonl"                  # Output file for completions
CACHE_DIR="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights"                        # Cache directory
USE_PEFT=true                                    # Whether to use PEFT weights

# Ensure the cache directory exists
mkdir -p $CACHE_DIR

# Convert USE_PEFT to the appropriate flag
PEFT_FLAG=""
if [ "$USE_PEFT" = false ]; then
    PEFT_FLAG="--no_peft"
fi

# Run the Python evaluation script
python benchmark_checkpoint.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_file "$OUTPUT_FILE" \
    --cache_dir "$CACHE_DIR" \
    $PEFT_FLAG