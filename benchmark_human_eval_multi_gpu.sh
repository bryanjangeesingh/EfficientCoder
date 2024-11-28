#!/bin/bash

# Define default argument values
GPU_IDS="0,1,2,3"                          # Comma-separated list of GPU IDs
OUTPUT_FILE="completions.jsonl"            # Path to save the completions
MODEL_NAME="codellama/CodeLlama-7b-hf"     # Hugging Face model name or path
CACHE_DIR="/nobackup/users/brytech/projects/condas/nlp_4gpus/model_weights_cache/"              # Directory for model cache
TOKENIZER_MAX_LENGTH=512                  # Maximum length for tokenizer input
MAX_NEW_TOKENS=512                        # Maximum number of tokens to generate

# Ensure the cache directory exists
mkdir -p $CACHE_DIR

# Run the Python evaluation script
python benchmark_human_eval_multi_gpu.py \
    --gpu_ids "$GPU_IDS" \
    --output_file "$OUTPUT_FILE" \
    --model_name "$MODEL_NAME" \
    --cache_dir "$CACHE_DIR" \
    --input_tokenizer_max_length "$TOKENIZER_MAX_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS"