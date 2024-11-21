import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
from accelerate import Accelerator
import torch.distributed as dist
import json
from pathlib import Path
import random
import logging
import os
from tqdm import tqdm
import queue
import threading
from accelerate import DistributedDataParallelKwargs
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class CodeSearchNetDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        languages: List[str] = ["python"],
        max_samples_per_language: int = None,
        split: str = "train"  # Options: train, valid, test
    ):
        """
        Initialize CodeSearchNet dataset.
        
        Args:
            data_path: Path to CodeSearchNet data directory
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            languages: List of programming languages to include
            max_samples_per_language: Maximum number of samples per language
            split: Which data split to use (train, valid, or test)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.languages = languages
        self.split = split
        
        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of: train, valid, test")
        
        # Load and process the data
        self.samples = []
        for lang in languages:
            # Path structure: {data_path}/{lang}/{lang}/final/jsonl/{split}/*.jsonl
            lang_path = self.data_path / lang / lang / "final" / "jsonl" / split
            if not lang_path.exists():
                raise ValueError(f"Path not found: {lang_path}")
            
            # Find all training files (they might be split into multiple parts)
            train_files = list(lang_path.glob("*.jsonl"))
            if not train_files:
                raise ValueError(f"No {split} files found in {lang_path}")
            
            logger.info(f"Found {len(train_files)} {split} files for {lang}")
            
            # Load all files for this split
            lang_samples = []
            for file in train_files:
                with open(file) as f:
                    file_samples = [json.loads(line) for line in f]
                    lang_samples.extend(file_samples)
                    logger.info(f"Loaded {len(file_samples)} samples from {file.name}")
            
            # Filter out samples with empty or invalid code
            lang_samples = [
                sample for sample in lang_samples
                if sample["code"] and len(sample["code"].strip()) > 0
            ]
            
            logger.info(f"Total valid samples for {lang}: {len(lang_samples)}")
            
            # Limit samples if specified
            if max_samples_per_language:
                lang_samples = lang_samples[:max_samples_per_language]
                logger.info(f"Limited to {len(lang_samples)} samples for {lang}")
            
            self.samples.extend(lang_samples)
        
        # Shuffle the samples
        random.shuffle(self.samples)
        logger.info(f"Final dataset size: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        code = sample["code"]
        
        # Add special tokens and formatting
        formatted_code = f"# {sample['docstring']}\n{code}" if "docstring" in sample else code
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_code,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "original_text": formatted_code,
            "language": sample.get("language", "python"),
            "repo": sample.get("repo", ""),
            "path": sample.get("path", ""),
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "original_text": [x["original_text"] for x in batch],
            "language": [x["language"] for x in batch],
            "repo": [x["repo"] for x in batch],
            "path": [x["path"] for x in batch],
        }


class MultiTeacherDistillation:
    def __init__(
        self,
        teacher1_model_name: str = "codellama/CodeLlama-13b-Instruct-hf", # use the instruct model
        student_model_name: str = "codellama/CodeLlama-7b-hf",        # Base student model
        temperature: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_workers: int = 4,  # Number of data loading workers
        checkpoint_path: str = None,  # Path to load checkpoint from
    ):
        """Initialize the distillation framework with models on different GPUs."""
        self.device = device
        self.temperature = temperature
        self.teacher1_model_name = teacher1_model_name
        self.student_model_name = student_model_name
        self.num_workers = num_workers
        self.start_epoch = 0  # Track which epoch to start from
        
        # Set environment variable for memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=16,
            mixed_precision="fp16",
            device_placement=False,  # We'll handle device placement ourselves
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=True,
                    static_graph=False
                )
            ]
        )
        
        # Configure model loading with explicit GPU assignments
        teacher_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": {"": "cuda:0"},
            "use_cache": False,
        }
        
        student_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": {"": "cuda:1"},
            "use_cache": False,
        }

        logger.info("Loading teacher (13B) on GPU 0...")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher1_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_13b_instruct",
            **teacher_kwargs
        )
        self.teacher.gradient_checkpointing_enable()
        
        # Clear GPU 0 cache
        torch.cuda.empty_cache()
        
        logger.info("Loading student model (7B) on GPU 1...")
        self.student = AutoModelForCausalLM.from_pretrained(
            self.student_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_distilled_student",
            **student_kwargs
        )
        self.student.gradient_checkpointing_enable()

        # Initialize tokenizer with parallel processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.teacher1_model_name,
            use_fast=True  # Use fast tokenizer for better performance
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize optimizer with more conservative learning rates
        param_groups = [
            {
                'params': [p for n, p in self.student.named_parameters() if 'layer' in n],
                'lr': 1e-5,  # Reduced from 5e-5
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.student.named_parameters() if 'layer' not in n],
                'lr': 2e-5,  # Reduced from 1e-4
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Add gradient clipping with max norm
        self.max_grad_norm = 1.0
        
        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.student.load_state_dict(checkpoint["student_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"]  # Resume from the next epoch
            logger.info(f"Resuming from epoch {self.start_epoch}")
        
        # Only prepare optimizer and scheduler
        self.optimizer = self.accelerator.prepare(self.optimizer)
        
        # Setup prefetching queue for next batch
        self.next_batch = None
        self.prefetch_queue = queue.Queue(maxsize=2)
        self.prefetch_thread = None

    def prefetch_batch(self, dataloader_iter):
        """Prefetch next batch in background."""
        try:
            batch = next(dataloader_iter)
            self.prefetch_queue.put(batch)
        except StopIteration:
            self.prefetch_queue.put(None)

    def get_next_batch(self, dataloader_iter):
        """Get next batch with prefetching."""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_batch,
                args=(dataloader_iter,)
            )
            self.prefetch_thread.start()
        
        batch = self.prefetch_queue.get()
        if batch is not None:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_batch,
                args=(dataloader_iter,)
            )
            self.prefetch_thread.start()
        
        return batch

    def train_step(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform one training step with single teacher distillation."""
        try:
            self.optimizer.zero_grad()
            
            # Move input tensors to appropriate devices
            teacher_input_ids = input_batch["input_ids"].to("cuda:0")
            student_input_ids = input_batch["input_ids"].to("cuda:1")
            
            # Get teacher outputs with mixed precision
            with torch.no_grad(), torch.cuda.amp.autocast():
                teacher_outputs = self.teacher(teacher_input_ids)
                teacher_logits = teacher_outputs.logits.to("cuda:1").detach()
            
            # Clear GPU 0 cache after teacher forward pass
            torch.cuda.empty_cache()
            
            # Get student outputs with mixed precision
            with torch.cuda.amp.autocast():
                student_outputs = self.student(student_input_ids)

                # Apply stable scaling to logits
                def scale_logits(logits, temp=1.0, max_value=10.0):
                    logits = logits - logits.mean(dim=-1, keepdim=True)
                    logits = torch.clamp(logits, -max_value, max_value)
                    return logits / (temp + 1e-8)

                teacher_logits = scale_logits(teacher_logits, self.temperature)
                student_logits = scale_logits(student_outputs.logits, self.temperature)

                # Compute probabilities with numerical stability
                def compute_probs(logits):
                    probs = F.softmax(logits, dim=-1)
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    return probs / probs.sum(dim=-1, keepdim=True)

                teacher_probs = compute_probs(teacher_logits)
                student_log_probs = F.log_softmax(student_logits, dim=-1)

                # Compute cross-entropy loss
                shift_logits = student_outputs.logits[:, :-1, :].contiguous()
                shift_labels = student_input_ids[:, 1:].contiguous()
                
                # Add label smoothing
                smoothing = 0.1
                n_class = shift_logits.size(-1)
                one_hot = torch.zeros_like(shift_logits).scatter(
                    2, shift_labels.unsqueeze(-1), 1-smoothing
                )
                one_hot = one_hot + smoothing/n_class
                
                # Compute CE loss with label smoothing
                log_probs = F.log_softmax(shift_logits, dim=-1)
                ce_loss = -(one_hot * log_probs).sum(dim=-1).mean()

                if torch.isnan(ce_loss):
                    return torch.tensor(0.0, device="cuda:1", requires_grad=True)

                # Compute KL divergence with stability measures
                kl_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction='batchmean',
                    log_target=False
                )
                kl_loss = torch.clamp(kl_loss, max=10.0)

                # Dynamic loss weighting based on training progress
                try:
                    # Get optimizer step count safely
                    opt_state = self.optimizer.state_dict()["state"]
                    if opt_state:  # Check if state exists
                        first_param_state = opt_state[list(opt_state.keys())[0]]
                        step_count = first_param_state.get("step", 0)
                    else:
                        step_count = 0
                    
                    progress = min(1.0, step_count / 1000)
                except:
                    # Fallback to default weights if any error
                    progress = 0.0
                
                kl_weight = max(0.1, 0.5 * (1 - progress))  # Gradually reduce KL weight
                ce_weight = 1 - kl_weight

                # Combine losses with dynamic weighting
                combined_loss = (
                    ce_weight * ce_loss +
                    kl_weight * kl_loss
                )
            
            if torch.isnan(combined_loss):
                return torch.tensor(0.0, device="cuda:1", requires_grad=True)
            
            # Compute gradients with gradient scaling
            self.accelerator.backward(combined_loss)
            
            # Clear unnecessary tensors
            del teacher_logits, teacher_probs, student_log_probs
            torch.cuda.empty_cache()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=self.max_grad_norm)
            
            return combined_loss
            
        except Exception as e:
            logger.error(f"Error in train_step: {str(e)}")
            raise

    def train(
        self,
        train_dataset: CodeSearchNetDataset,
        num_epochs: int,
        batch_size: int,
        eval_dataset: Optional[CodeSearchNetDataset] = None,
    ):
        """Train the student model."""
        self.student.train()
        
        # Create data loader with efficient settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        
        num_training_steps = len(train_dataset) // batch_size * num_epochs
        num_warmup_steps = num_training_steps // 10  # 10% warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        for epoch in range(self.start_epoch, num_epochs):  # Start from the loaded epoch
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(
                total=len(train_loader),
                desc=f"E{epoch + 1}/{num_epochs}",
                dynamic_ncols=True,
                mininterval=2.0,
                miniters=10
            )
            
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    progress_bar.set_postfix({'l': f'{(total_loss / num_batches):.3f}'})
                    progress_bar.update(10)
                
                if num_batches % 100 == 0:
                    torch.cuda.empty_cache()
            
            progress_bar.close()
            avg_loss = total_loss / num_batches
            
            # Evaluate if dataset provided
            if eval_dataset is not None:
                eval_loss = self.evaluate(eval_dataset, batch_size)
                print(f"E{epoch + 1}: train={avg_loss:.3f}, eval={eval_loss:.3f}")
            else:
                print(f"E{epoch + 1}: train={avg_loss:.3f}")
            
            # Save checkpoint after each epoch
            checkpoint = {
                "student_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch + 1,  # Save the next epoch number
                "train_loss": avg_loss
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch + 1}.pt")

    def evaluate(self, eval_dataset: CodeSearchNetDataset, batch_size: int):
        """Evaluate the student model."""
        self.student.eval()
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        total_loss = 0

        with torch.no_grad():
            for batch in eval_loader:
                loss = self.train_step(batch)
                total_loss += loss

        avg_loss = total_loss / len(eval_loader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            "student": self.student.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, filename)
