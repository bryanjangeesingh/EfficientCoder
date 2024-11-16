import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from train import CodeSearchNetDataset
from transformers import AutoTokenizer
import logging
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize wandb for experiment tracking
    wandb.init(
        project="code-distillation",
        config=vars(args)
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize distillation framework
    from train import MultiTeacherDistillation
    distiller = MultiTeacherDistillation(
        temperature=args.temperature,
    )
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = CodeSearchNetDataset(
        data_path=args.data_path,
        tokenizer=distiller.tokenizer,
        max_length=args.max_length,
        max_samples_per_language=args.max_samples
    )
    
    # Create validation set (10% of training data)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
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
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training
        distiller.student.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            loss = distiller.train_step(batch)
            total_train_loss += loss
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss:.4f}")
                wandb.log({
                    "train_batch_loss": loss,
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        distiller.student.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = distiller.train_step(batch)  # Using train_step but in eval mode
                total_val_loss += loss
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        logger.info(f"Average Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Average Val Loss: {avg_val_loss:.4f}")
        
        wandb.log({
            "train_epoch_loss": avg_train_loss,
            "val_epoch_loss": avg_val_loss,
            "epoch": epoch
        })
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'student_state_dict': distiller.student.state_dict(),
                'optimizer_state_dict': distiller.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
