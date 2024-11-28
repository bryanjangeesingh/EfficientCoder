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
        teacher1_model_name: str = "WizardLMTeam/WizardCoder-Python-13B-V1.0",
        student_model_name: str = "codellama/CodeLlama-7b-hf",
        temperature: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_workers: int = 4,
        checkpoint_path: str = None,
        max_length: int = 512  
    ):
        """Initialize the distillation framework with models on different GPUs."""
        self.device = device
        self.temperature = temperature
        self.teacher1_model_name = teacher1_model_name
        self.student_model_name = student_model_name
        self.num_workers = num_workers
        self.start_epoch = 0
        self.max_length = max_length
        
        # Memory optimization settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        torch.backends.cuda.max_memory_split_size = 128 * 1024 * 1024  # 128MB
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=64,  # Increased for better memory efficiency
            mixed_precision="fp16",
            device_placement=False,
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=True,
                    static_graph=False
                )
            ]
        )
        
        teacher_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": {"": "cuda:0"},
            "use_cache": False,
            "low_cpu_mem_usage": True
        }
        
        student_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": {"": "cuda:1"},
            "use_cache": False,
            "low_cpu_mem_usage": True
        }

        logger.info("Loading WizardCoder teacher (13B) on GPU 0...")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher1_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_wizard_coder",
            **teacher_kwargs
        )
        self.teacher.gradient_checkpointing_enable()
        self.teacher.config.use_cache = False
        
        # Enable memory efficient attention
        if hasattr(self.teacher.config, "attention_mode"):
            self.teacher.config.attention_mode = "memory_efficient"
        
        torch.cuda.empty_cache()
        
        logger.info("Loading student model (7B) on GPU 1...")
        self.student = AutoModelForCausalLM.from_pretrained(
            self.student_model_name, 
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_distilled_student",
            **student_kwargs
        )
        self.student.gradient_checkpointing_enable()
        self.student.config.use_cache = False
        
        # Enable memory efficient attention
        if hasattr(self.student.config, "attention_mode"):
            self.student.config.attention_mode = "memory_efficient"
        
        # Initialize tokenizers for both models
        logger.info("Loading tokenizers...")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            self.teacher1_model_name,
            use_fast=True,
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_wizard_coder"
        )
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_name,
            use_fast=True,
            cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_distilled_student"
        )

        # Set pad tokens if not present
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token

        self.tokenizer = self.teacher_tokenizer  # Use teacher tokenizer as default for dataset


        # Rest of the initialization remains the same
        param_groups = [
            {
                'params': [p for n, p in self.student.named_parameters() if 'layer' in n],
                'lr': 1e-5,
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.student.named_parameters() if 'layer' not in n],
                'lr': 2e-5,
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        self.max_grad_norm = 1.0
        
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.student.load_state_dict(checkpoint["student_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            logger.info(f"Resuming from epoch {self.start_epoch}")
        
        self.optimizer = self.accelerator.prepare(self.optimizer)
        
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
    
    def generate_teacher_input(self, code: str, docstring: str = None) -> str:
        """Format input for WizardCoder."""
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Create a Python script for this problem:
        {docstring if docstring else ''}

        {code}

        ### Response:"""
        return prompt
    
    def generate_student_input(self, code: str, docstring: str = None) -> str:
        """Format input for CodeLlama."""
        return f"# Complete the following Python function:\n\n{docstring if docstring else ''}\n{code}"
    
    def post_process_output(self, completion: str) -> str:
        """Post-process model outputs to extract clean code."""
        completion = completion.replace("\r", "")
        
        # Extract code from markdown code blocks if present
        if "```python" in completion:
            def_line = completion.index("```python")
            completion = completion[def_line:].replace("```python", "").strip()
            if "```" in completion:
                next_line = completion.index("```")
                completion = completion[:next_line].strip()
        
        # Remove main block if present
        if "__name__ == \"__main__\"" in completion:
            next_line = completion.index('if __name__ == "__main__":')
            completion = completion[:next_line].strip()
        
        # Remove example usage if present
        if "# Example usage" in completion:
            next_line = completion.index("# Example usage")
            completion = completion[:next_line].strip()
        
        return completion.strip()

    def compute_uld_loss(self, teacher_logits, student_logits, temperature=1.0):
        """
        Memory-efficient implementation of Universal Logit Distillation loss.
        Uses chunked computation to avoid OOM issues.
        """
        # Ensure inputs have the same shape and vocab size
        batch_size, seq_len, _ = teacher_logits.size()
        min_vocab = min(teacher_logits.size(-1), student_logits.size(-1))
        
        # Truncate to smaller vocab size
        teacher_logits = teacher_logits[..., :min_vocab]
        student_logits = student_logits[..., :min_vocab]
        
        # Flatten batch and sequence dimensions
        teacher_flat = teacher_logits.view(-1, min_vocab)  # [B*S, V]
        student_flat = student_logits.view(-1, min_vocab)  # [B*S, V]
        
        total_tokens = teacher_flat.size(0)
        chunk_size = 128  # Process in smaller chunks to save memory
        loss = 0
        num_comparisons = 0
        
        for i in range(0, total_tokens, chunk_size):
            # Get current chunk
            end_idx = min(i + chunk_size, total_tokens)
            teacher_chunk = teacher_flat[i:end_idx]
            student_chunk = student_flat[i:end_idx]
            
            # Compute chunk-wise differences
            teacher_diff = (teacher_chunk.unsqueeze(1) - teacher_chunk.unsqueeze(0)) / temperature
            student_diff = (student_chunk.unsqueeze(1) - student_chunk.unsqueeze(0)) / temperature
            
            # Create mask for valid comparisons (excluding self-comparisons)
            chunk_size_curr = end_idx - i
            mask = ~torch.eye(chunk_size_curr, dtype=torch.bool, device=teacher_diff.device)
            
            # Use binary_cross_entropy_with_logits which is safe with autocast
            teacher_diff = teacher_diff[mask]
            student_diff = student_diff[mask]
            
            # Convert teacher differences to target probabilities
            with torch.cuda.amp.autocast(enabled=False):
                teacher_prob = torch.sigmoid(teacher_diff.float())
            
            # Compute loss using BCEWithLogitsLoss which is safe with autocast
            curr_loss = F.binary_cross_entropy_with_logits(
                student_diff,
                teacher_prob,
                reduction='sum'
            )
            
            loss += curr_loss
            num_comparisons += mask.sum()
            
            # Clean up intermediate tensors
            del teacher_diff, student_diff, teacher_prob
            torch.cuda.empty_cache()
        
        # Average loss over all valid comparisons
        return loss / num_comparisons if num_comparisons > 0 else torch.tensor(0.0, device=teacher_logits.device)

    def train_step(self, input_batch: Dict[str, torch.Tensor]):
        """Modified train step with memory optimizations."""
        try:
            # Get code from batch
            original_text = input_batch.get("original_text", [])  # Changed to match dataset output
            if not original_text:
                logger.warning("No original_text found in batch, trying alternate keys...")
                original_text = input_batch.get("code", []) or input_batch.get("text", [])
            
            docstrings = input_batch.get("docstring", [None] * len(original_text))
            
            if not original_text:
                logger.error("Empty input batch - no code found")
                logger.error(f"Available batch keys: {list(input_batch.keys())}")
                raise ValueError("Empty input batch - no code found")
            
            # Process single example at a time
            max_batch_size = 1
            if len(original_text) > max_batch_size:
                original_text = original_text[:max_batch_size]
                docstrings = docstrings[:max_batch_size]
            
            # Generate inputs
            teacher_inputs = [
                self.generate_teacher_input(code=sample, docstring=doc)
                for sample, doc in zip(original_text, docstrings)
            ]
            
            student_inputs = [
                self.generate_student_input(code=sample, docstring=doc)
                for sample, doc in zip(original_text, docstrings)
            ]
            
            # Minimal sequence length and ensure same padding for both models
            max_length = min(self.max_length, 96)  # Further reduced from 128
            
            try:
                # Tokenize with same padding strategy for both models
                padding_kwargs = {
                    "padding": True,
                    "truncation": True,
                    "max_length": max_length,
                    "return_tensors": "pt",
                    "padding_side": "right",  # Ensure consistent padding side
                    "return_attention_mask": True
                }
                
                teacher_tokens = self.teacher_tokenizer(
                    teacher_inputs,
                    **padding_kwargs
                )
                
                student_tokens = self.student_tokenizer(
                    student_inputs,
                    **padding_kwargs
                )
                
                # Move to appropriate devices and convert to half precision
                teacher_tokens = {
                    k: v.to("cuda:0", dtype=torch.long if k == "input_ids" or k == "attention_mask" else torch.float16) 
                    for k, v in teacher_tokens.items()
                }
                student_tokens = {
                    k: v.to("cuda:1", dtype=torch.long if k == "input_ids" or k == "attention_mask" else torch.float16)
                    for k, v in student_tokens.items()
                }
                
                # Ensure sequences are padded to same length
                if teacher_tokens["input_ids"].size(1) != student_tokens["input_ids"].size(1):
                    max_len = max(teacher_tokens["input_ids"].size(1), student_tokens["input_ids"].size(1))
                    
                    # Pad teacher tokens if needed
                    if teacher_tokens["input_ids"].size(1) < max_len:
                        pad_len = max_len - teacher_tokens["input_ids"].size(1)
                        teacher_tokens["input_ids"] = F.pad(teacher_tokens["input_ids"], (0, pad_len), value=self.teacher_tokenizer.pad_token_id)
                        teacher_tokens["attention_mask"] = F.pad(teacher_tokens["attention_mask"], (0, pad_len), value=0)
                    
                    # Pad student tokens if needed
                    if student_tokens["input_ids"].size(1) < max_len:
                        pad_len = max_len - student_tokens["input_ids"].size(1)
                        student_tokens["input_ids"] = F.pad(student_tokens["input_ids"], (0, pad_len), value=self.student_tokenizer.pad_token_id)
                        student_tokens["attention_mask"] = F.pad(student_tokens["attention_mask"], (0, pad_len), value=0)
                
            except Exception as e:
                logger.error(f"Tokenization error: {str(e)}")
                return torch.tensor(0.0, device="cuda:1", requires_grad=True)
            
            # Get teacher outputs with memory optimization
            try:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    # Process in smaller chunks for teacher
                    chunk_size = 16  # Reduced chunk size
                    seq_length = teacher_tokens["input_ids"].size(1)
                    teacher_logits_list = []
                    
                    for i in range(0, seq_length, chunk_size):
                        chunk_end = min(i + chunk_size, seq_length)
                        chunk_tokens = {
                            k: v[:, i:chunk_end] for k, v in teacher_tokens.items()
                        }
                        chunk_output = self.teacher(**chunk_tokens)
                        teacher_logits_list.append(chunk_output.logits)
                        del chunk_output
                        torch.cuda.empty_cache()
                    
                    teacher_logits = torch.cat(teacher_logits_list, dim=1).to("cuda:1", dtype=torch.float16)
                    del teacher_logits_list
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Teacher forward pass error: {str(e)}")
                return torch.tensor(0.0, device="cuda:1", requires_grad=True)
            
            # Get student outputs with memory optimization
            try:
                with torch.cuda.amp.autocast():
                    # Process in smaller chunks for student
                    student_logits_list = []
                    for i in range(0, seq_length, chunk_size):
                        chunk_end = min(i + chunk_size, seq_length)
                        chunk_tokens = {
                            k: v[:, i:chunk_end] for k, v in student_tokens.items()
                        }
                        chunk_output = self.student(**chunk_tokens)
                        student_logits_list.append(chunk_output.logits)
                        del chunk_output
                        torch.cuda.empty_cache()
                    
                    student_logits = torch.cat(student_logits_list, dim=1)
                    del student_logits_list
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Student forward pass error: {str(e)}")
                return torch.tensor(0.0, device="cuda:1", requires_grad=True)
            
            # Log shapes for debugging
            logger.info(f"Shapes: teacher={teacher_logits.shape}, student={student_logits.shape}")
            
            # Handle NaN/Inf
            teacher_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                    
            # Process loss in smaller chunks
            total_loss = 0
            num_chunks = 0
            vocab_chunk_size = 1024  # Process vocabulary in chunks
            
            for i in range(0, teacher_logits.size(-1), vocab_chunk_size):
                vocab_end = min(i + vocab_chunk_size, teacher_logits.size(-1))
                teacher_vocab_chunk = teacher_logits[..., i:vocab_end]
                student_vocab_chunk = student_logits[..., i:vocab_end]
                
                chunk_loss = self.compute_uld_loss(
                    teacher_vocab_chunk,
                    student_vocab_chunk,
                    temperature=self.temperature
                )
                total_loss += chunk_loss
                num_chunks += 1
                
                del teacher_vocab_chunk, student_vocab_chunk
                torch.cuda.empty_cache()
            
            loss = total_loss / num_chunks if num_chunks > 0 else torch.tensor(0.0, device="cuda:1", requires_grad=True)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss detected")
                return torch.tensor(0.0, device="cuda:1", requires_grad=True)
            
            # Scale loss for gradient accumulation
            loss = loss / self.accelerator.gradient_accumulation_steps
            
            self.accelerator.backward(loss)
            
            # Aggressive cleanup
            del teacher_logits, student_logits, teacher_tokens, student_tokens
            torch.cuda.empty_cache()
            
            # Clip gradients
            if self.accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            
            return loss
            
        except Exception as e:
            logger.error(f"Train step error: {str(e)}")
            return torch.tensor(0.0, device="cuda:1", requires_grad=True)

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
