import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from train import CodeSearchNetDataset
from transformers import AutoTokenizer
import logging
import sys
import csv
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def write_metrics(metrics_file, metrics_dict, epoch=None, batch=None):
    """Write metrics to CSV file."""
    # Create file with headers if it doesn't exist
    if not Path(metrics_file).exists():
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['timestamp', 'epoch', 'batch']
            headers.extend(metrics_dict.keys())
            writer.writerow(headers)
    
    # Append metrics
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, batch]
        row.extend(metrics_dict.values())
        writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser(description="Train with CodeSearchNet dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CodeSearchNet data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per language"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for distillation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the model during training"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_metrics_file = output_dir / 'train_metrics.csv'
    val_metrics_file = output_dir / 'val_metrics.csv'
    
    # Initialize the distiller
    from train import MultiTeacherDistillation
    distiller = MultiTeacherDistillation(
        teacher1_model_name="codellama/CodeLlama-13b-hf",
        student_model_name="codellama/CodeLlama-7b-hf",
        temperature=args.temperature
    )
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = CodeSearchNetDataset(
        data_path=args.data_path,
        tokenizer=distiller.tokenizer,
        max_length=args.max_length,
        max_samples_per_language=args.max_samples,
        split="train"
    )
    
    val_dataset = CodeSearchNetDataset(
        data_path=args.data_path,
        tokenizer=distiller.tokenizer,
        max_length=args.max_length,
        max_samples_per_language=args.max_samples // 10 if args.max_samples else None,
        split="valid"
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CodeSearchNetDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CodeSearchNetDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Train the model
    distiller.train(
        train_dataset=train_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_dataset=val_dataset if args.eval else None
    )

if __name__ == "__main__":
    main()
