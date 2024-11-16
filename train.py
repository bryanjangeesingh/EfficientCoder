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
from typing import List, Tuple, Dict
from accelerate import Accelerator
import torch.distributed as dist
import json
from pathlib import Path
import random
import logging

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
        teacher1_model_name: str = "codellama/CodeLlama-13b-hf",        # General CodeLlama-13B
        teacher2_model_name: str = "codellama/CodeLlama-13b-Instruct-hf",  # Instruct version
        student_model_name: str = "codellama/CodeLlama-7b-hf",        # Base student model
        temperature: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the distillation framework with models on different GPUs."""
        self.device = device
        self.temperature = temperature
        self.teacher1_model_name = teacher1_model_name
        self.teacher2_model_name = teacher2_model_name  
        self.student_model_name = student_model_name
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        # Configure model loading with explicit GPU assignments
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",  # Let HF handle device mapping
            "use_cache": False,  # Disable KV-cache for training
            "use_flash_attention_2": True,  # Enable memory efficient attention
        }

        logger.info("Loading teacher1 (13B) on GPU...")
        self.teacher1 = AutoModelForCausalLM.from_pretrained(
            self.teacher1_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_13b",
            **model_kwargs
        )
        self.teacher1.gradient_checkpointing_enable()  # Enable gradient checkpointing
        
        logger.info("Loading teacher2 instruct (13B) on GPU...")
        self.teacher2 = AutoModelForCausalLM.from_pretrained(
            self.teacher2_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_13b_instruct",
            **model_kwargs
        )
        self.teacher2.gradient_checkpointing_enable()
        
        logger.info("Loading student model (7B) on GPU...")
        self.student = AutoModelForCausalLM.from_pretrained(
            self.student_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_distilled_student",
            **model_kwargs
        )
        self.student.gradient_checkpointing_enable()

        # Initialize tokenizer (all use the same CodeLlama tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher1_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize optimizer with 8-bit Adam for memory efficiency
        from bitsandbytes.optim import AdamW8bit
        self.optimizer = AdamW8bit(self.student.parameters(), lr=1e-4)
        
        # Prepare for distributed training
        self.student, self.optimizer = self.accelerator.prepare(
            self.student, self.optimizer
        )

    def train_step(self, input_batch: Dict[str, torch.Tensor]) -> float:
        """Perform one training step with multi-teacher distillation."""
        self.optimizer.zero_grad()
        total_loss = 0
        batch_size = input_batch["input_ids"].shape[0]

        for i in range(batch_size):
            input_ids = input_batch["input_ids"][i].unsqueeze(0).to(self.device)
            
            # Get teacher outputs
            with torch.no_grad():
                teacher1_outputs = self.teacher1(input_ids)
                
                # Format input with instruction for teacher2
                code_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                instruct_prompt = f"[INST] Generate a Python function:\n{code_text} [/INST]"
                instruct_inputs = self.tokenizer(
                    instruct_prompt,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=input_ids.shape[1]  # Match original sequence length
                ).input_ids.to(self.device)
                
                # Ensure teacher2 output matches sequence length
                teacher2_outputs = self.teacher2(instruct_inputs)
                if teacher2_outputs.logits.shape[1] != input_ids.shape[1]:
                    # Pad or truncate to match sequence length
                    if teacher2_outputs.logits.shape[1] < input_ids.shape[1]:
                        pad_size = input_ids.shape[1] - teacher2_outputs.logits.shape[1]
                        teacher2_outputs.logits = F.pad(teacher2_outputs.logits, (0, 0, 0, pad_size))
                    else:
                        teacher2_outputs.logits = teacher2_outputs.logits[:, :input_ids.shape[1], :]
            
            # Get student outputs
            student_outputs = self.student(input_ids)

            # Apply temperature scaling to logits
            teacher1_logits = teacher1_outputs.logits / self.temperature
            teacher2_logits = teacher2_outputs.logits / self.temperature
            student_logits = student_outputs.logits / self.temperature

            # Convert to probability distributions
            teacher1_probs = F.softmax(teacher1_logits, dim=-1)
            teacher2_probs = F.softmax(teacher2_logits, dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)

            # Compute teacher confidence scores
            teacher1_conf = torch.max(teacher1_probs, dim=-1)[0].mean()
            teacher2_conf = torch.max(teacher2_probs, dim=-1)[0].mean()
            
            # Compute adaptive weights based on confidence
            total_conf = teacher1_conf + teacher2_conf
            teacher1_weight = teacher1_conf / total_conf
            teacher2_weight = teacher2_conf / total_conf

            # Compute KL divergence losses for each teacher
            kl_loss_teacher1 = F.kl_div(
                student_log_probs,
                teacher1_probs,
                reduction='batchmean',
                log_target=False
            ) * (self.temperature ** 2)

            kl_loss_teacher2 = F.kl_div(
                student_log_probs,
                teacher2_probs,
                reduction='batchmean',
                log_target=False
            ) * (self.temperature ** 2)

            # Compute weighted distillation loss
            distill_loss = (teacher1_weight * kl_loss_teacher1 + 
                          teacher2_weight * kl_loss_teacher2)

            # Compute cross-entropy loss with ground truth (next token prediction)
            shift_logits = student_outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Optional: Add consistency loss between teachers
            teacher_consistency_loss = F.kl_div(
                F.log_softmax(teacher1_logits, dim=-1),
                F.softmax(teacher2_logits, dim=-1),
                reduction='batchmean',
                log_target=False
            ) * (self.temperature ** 2)

            # Combine all losses
            alpha = 0.1  # weight for CE loss
            beta = 0.8   # weight for distillation loss
            gamma = 0.1  # weight for teacher consistency loss
            
            combined_loss = (alpha * ce_loss + 
                           beta * distill_loss + 
                           gamma * teacher_consistency_loss)
            
            total_loss += combined_loss

        avg_loss = total_loss / batch_size
        self.accelerator.backward(avg_loss)
        self.optimizer.step()

        return avg_loss.item()

    def train(
        self,
        train_dataset: CodeSearchNetDataset,
        num_epochs: int,
        batch_size: int,
        eval_dataset: CodeSearchNetDataset = None,
    ):
        """Train the student model."""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            self.student.train()
            total_loss = 0

            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss

            avg_epoch_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

            if eval_dataset:
                self.evaluate(eval_dataset, batch_size)

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
