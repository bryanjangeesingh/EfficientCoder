#!/bin/bash

# Configuration
DATA_PATH="/nobackup/users/brytech/codesearchnet"  # Update this path
OUTPUT_DIR="./outputs/distillation_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=32
NUM_EPOCHS=10
MAX_LENGTH=512
LEARNING_RATE=1e-4
TEMPERATURE=2.0
MAX_SAMPLES=100000  # Set to None for full dataset

# Create output directory
mkdir -p $OUTPUT_DIR

# Set up logging
exec &> >(tee -a "$OUTPUT_DIR/training.log")
echo "Starting training at $(date)"
echo "Output directory: $OUTPUT_DIR"

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Environment setup
echo "Setting up Python environment..."
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs
export TOKENIZERS_PARALLELISM=true

# Launch training
echo "Launching training..."
python train_on_codesearchnet.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --max_samples $MAX_SAMPLES \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
else
    echo "Training failed at $(date)"
    exit 1
fi

# Compress logs and checkpoints
echo "Compressing output directory..."
tar -czf "$OUTPUT_DIR.tar.gz" $OUTPUT_DIR

echo "All done! Output saved to $OUTPUT_DIR"
echo "Compressed archive saved to $OUTPUT_DIR.tar.gz"
