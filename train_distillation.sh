#!/bin/bash

# Configuration
DATA_PATH="/nobackup/users/brytech/codesearchnet"  # Update this path
OUTPUT_DIR="./outputs/distillation_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=8  # Increased from 4 since we have better parallelization
GRADIENT_ACCUMULATION_STEPS=8  # Reduced since we increased batch size
NUM_EPOCHS=10
MAX_LENGTH=256
LEARNING_RATE=5e-5
TEMPERATURE=2.0
MAX_SAMPLES=100000
NUM_WORKERS=8  # Use more workers for data loading

# Set PyTorch environment variables for performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Create output directory
mkdir -p $OUTPUT_DIR

# Set up logging
exec &> >(tee -a "$OUTPUT_DIR/training.log")
echo "Starting training at $(date)"
echo "Output directory: $OUTPUT_DIR"

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Launch training
echo "Launching training..."
python train_on_codesearchnet.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_epochs $NUM_EPOCHS \
    --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --max_samples $MAX_SAMPLES \
    --num_workers $NUM_WORKERS \
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
