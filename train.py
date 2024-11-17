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
        teacher1_model_name: str = "codellama/CodeLlama-7b-Instruct-hf",      # use the instruct model
        student_model_name: str = "codellama/CodeLlama-7b-hf",        # Base student model
        temperature: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the distillation framework with models on different GPUs."""
        self.device = device
        self.temperature = temperature
        self.teacher1_model_name = teacher1_model_name
        self.student_model_name = student_model_name
        
        # Set environment variable for memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Initialize accelerator without device placement
        self.accelerator = Accelerator(
            gradient_accumulation_steps=8,
            mixed_precision="fp16",
            device_placement=False  # We'll handle device placement ourselves
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

        logger.info("Loading teacher (7B) on GPU 0...")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher1_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_7b_instruct",
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

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher1_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize optimizer with larger batch size and learning rate
        param_groups = [
            {
                'params': [p for n, p in self.student.named_parameters() if 'layer' in n],
                'lr': 5e-5,
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.student.named_parameters() if 'layer' not in n],
                'lr': 1e-4,
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Add gradient clipping
        self.max_grad_norm = 1.0
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[5e-5, 1e-4],
            total_steps=1000,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Only prepare optimizer and scheduler, not model
        self.optimizer, self.scheduler = self.accelerator.prepare(
            self.optimizer, self.scheduler
        )

    def train_step(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform one training step with single teacher distillation."""
        try:
            logger.info("Starting train_step")
            self.optimizer.zero_grad()
            
            # Move entire batch to device at once
            input_ids = input_batch["input_ids"].to(self.device)
            batch_size = input_ids.shape[0]
            logger.info(f"Processing batch of size {batch_size}")
            
            # Get teacher outputs for entire batch
            logger.info("Getting teacher outputs")
            with torch.no_grad():
                teacher_outputs = self.teacher(input_ids)
            
            # Get student outputs for entire batch
            logger.info("Getting student outputs")
            student_outputs = self.student(input_ids)

            # Apply stable scaling to logits
            logger.info("Computing logits and probabilities")
            def scale_logits(logits, temp=1.0, max_value=10.0):
                logits = logits - logits.mean(dim=-1, keepdim=True)
                logits = torch.clamp(logits, -max_value, max_value)
                return logits / (temp + 1e-8)

            teacher_logits = scale_logits(teacher_outputs.logits, self.temperature)
            student_logits = scale_logits(student_outputs.logits, self.temperature)

            # Compute probabilities with numerical stability
            def compute_probs(logits):
                probs = F.softmax(logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                return probs / probs.sum(dim=-1, keepdim=True)

            teacher_probs = compute_probs(teacher_logits)
            student_log_probs = F.log_softmax(student_logits, dim=-1)

            logger.info("Computing losses")
            # Compute cross-entropy loss
            shift_logits = student_outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
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
                logger.error("NaN detected in CE loss, skipping batch")
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Compute KL divergence with stability measures
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean',
                log_target=False
            )
            kl_loss = torch.clamp(kl_loss, max=10.0)

            # Combine losses with careful weighting
            combined_loss = (
                0.5 * ce_loss +
                0.5 * kl_loss  # Equal weight between CE and KL since we only have one teacher
            )
            
            if torch.isnan(combined_loss):
                logger.warning(
                    f"NaN detected in loss computation:\n"
                    f"CE Loss: {ce_loss.item():.4f}\n"
                    f"KL Loss: {kl_loss.item():.4f}"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            logger.info(f"Final loss: {combined_loss.item():.4f}")
            
            # Compute gradients
            logger.info("Computing gradients")
            self.accelerator.backward(combined_loss)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=self.max_grad_norm)
            logger.info("Finished train_step")
            
            return combined_loss
            
        except Exception as e:
            logger.error(f"Error in train_step: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                dynamic_ncols=True
            )
            
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                # Update progress bar with current loss
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})
                progress_bar.update()
            
            progress_bar.close()
            avg_epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

            if eval_dataset:
                self.evaluate(eval_dataset, batch_size)

            self.scheduler.step()

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
