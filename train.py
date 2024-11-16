import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from accelerate import Accelerator
import torch.distributed as dist
import json
from pathlib import Path
import random

class CodeDistillationDataset(Dataset):
    def __init__(self, code_samples: List[str], tokenizer, max_length: int = 512):
        self.code_samples = code_samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.code_samples)

    def __getitem__(self, idx):
        code = self.code_samples[idx]
        inputs = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "original_text": code,
        }


class CodeSearchNetDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        languages: List[str] = ["python"],
        max_samples_per_language: int = None,
    ):
        """
        Initialize CodeSearchNet dataset.
        
        Args:
            data_path: Path to CodeSearchNet data directory
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            languages: List of programming languages to include
            max_samples_per_language: Maximum number of samples per language
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.languages = languages
        
        # Load and process the data
        self.samples = []
        for lang in languages:
            lang_path = self.data_path / lang / "final" / "jsonl"
            if not lang_path.exists():
                raise ValueError(f"Path not found: {lang_path}")
            
            # Load train.jsonl
            train_file = lang_path / "train.jsonl"
            with open(train_file) as f:
                lang_samples = [json.loads(line) for line in f]
            
            # Filter out samples with empty or invalid code
            lang_samples = [
                sample for sample in lang_samples
                if sample["code"] and len(sample["code"].strip()) > 0
            ]
            
            # Limit samples if specified
            if max_samples_per_language:
                lang_samples = lang_samples[:max_samples_per_language]
            
            self.samples.extend(lang_samples)
        
        # Shuffle the samples
        random.shuffle(self.samples)

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
        teacher1_model_name: str = "codellama/CodeLlama-34b-hf",        # General CodeLlama
        teacher2_model_name: str = "codellama/CodeLlama-7b-Python-hf",  # Python-specialized
        student_model_name: str = "anudaw/distilled-code-llama",        # Distilled version
        temperature: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.temperature = temperature
        self.teacher1_model_name = teacher1_model_name
        self.teacher2_model_name = teacher2_model_name
        self.student_model_name = student_model_name
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        # Configure model loading with optimal settings for V100s
        teacher1_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {i: "28GB" for i in range(torch.cuda.device_count())},
        }

        teacher2_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {i: "16GB" for i in range(torch.cuda.device_count())},
        }

        student_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {i: "8GB" for i in range(torch.cuda.device_count())},
        }

        # Initialize teacher1 (CodeLlama-34B general)
        self.teacher1 = AutoModelForCausalLM.from_pretrained(
            self.teacher1_model_name,
            **teacher1_kwargs
        )
        
        # Initialize teacher2 (CodeLlama-7B Python)
        self.teacher2 = AutoModelForCausalLM.from_pretrained(
            self.teacher2_model_name,
            **teacher2_kwargs
        )

        # Initialize student model (distilled)
        self.student = AutoModelForCausalLM.from_pretrained(
            self.student_model_name,
            **student_kwargs
        )

        # Initialize tokenizer (all use the same CodeLlama tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher1_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-4)
        
        # Prepare for distributed training
        self.student, self.optimizer = self.accelerator.prepare(
            self.student, self.optimizer
        )

    def get_embeddings(self, text: str) -> torch.Tensor:
        """Convert text to embeddings using sentence transformer."""
        return torch.tensor(self.sentence_embedder.encode(text))

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
                teacher2_outputs = self.teacher2(input_ids)
            
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

            # Compute weighted distillation loss # naturally we would want to put more weight on the more confident teacher
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
        train_dataset: CodeDistillationDataset,
        num_epochs: int,
        batch_size: int,
        eval_dataset: CodeDistillationDataset = None,
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

    def evaluate(self, eval_dataset: CodeDistillationDataset, batch_size: int):
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
