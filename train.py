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


class MultiTeacherDistillation:
    def __init__(
        self,
        temperature: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.temperature = temperature
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        # Configure model loading with optimal settings for V100s
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {i: "28GB" for i in range(torch.cuda.device_count())},  # Reserve some memory for gradients
            "offload_folder": "offload_folder",  # For potential CPU offloading if needed
        }

        # Initialize models with distributed setup
        self.code_llama = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-34b-hf",
            **model_kwargs
        )
        
        # Initialize CodeT5p-2B with memory optimizations
        self.code_t5 = T5ForConditionalGeneration.from_pretrained(
            "Salesforce/codet5p-2b",
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={i: "24GB" for i in range(torch.cuda.device_count())},  # Reserve memory for other models
        )

        # Initialize student model (3B distilled CodeLlama)
        student_model_name = "anudaw/distilled-code-llama"
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={i: "8GB" for i in range(torch.cuda.device_count())},  # Smaller memory footprint for 3B model
        )

        # Initialize tokenizers
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            "codellama/CodeLlama-34b-hf"
        )
        self.t5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-2b")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

        # Initialize sentence embedder for alignment
        self.sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)

        # Initialize optimizer with gradient accumulation
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-4)
        
        # Prepare for distributed training
        self.student, self.optimizer, self.code_t5 = self.accelerator.prepare(
            self.student, self.optimizer, self.code_t5
        )

    def get_embeddings(self, text: str) -> torch.Tensor:
        """Convert text to embeddings using sentence transformer."""
        return torch.tensor(self.sentence_embedder.encode(text))

    def compute_adaptive_weights(
        self, llama_confidence: float, t5_confidence: float
    ) -> Tuple[float, float]:
        """Compute adaptive weights based on model confidences."""
        total = llama_confidence + t5_confidence
        return llama_confidence / total, t5_confidence / total

    def get_teacher_outputs(
        self, input_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Get outputs from both teacher models."""
        # Code Llama generation
        llama_inputs = self.llama_tokenizer(input_text, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            llama_outputs = self.code_llama.generate(
                **llama_inputs,
                max_length=100,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        llama_text = self.llama_tokenizer.decode(llama_outputs.sequences[0])
        llama_confidence = torch.mean(
            torch.softmax(llama_outputs.scores[0], dim=-1).max(dim=-1)[0]
        )

        # CodeT5 generation
        t5_inputs = self.t5_tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t5_outputs = self.code_t5.generate(
                **t5_inputs,
                max_length=100,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        t5_text = self.t5_tokenizer.decode(t5_outputs.sequences[0])
        t5_confidence = torch.mean(
            torch.softmax(t5_outputs.scores[0], dim=-1).max(dim=-1)[0]
        )

        # Convert outputs to embeddings
        llama_embeddings = self.get_embeddings(llama_text)
        t5_embeddings = self.get_embeddings(t5_text)

        return (
            llama_embeddings,
            t5_embeddings,
            llama_confidence.item(),
            t5_confidence.item(),
        )

    def compute_kl_divergence(
        self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between student and teacher embeddings."""
        student_dist = F.softmax(student_embeddings / self.temperature, dim=-1)
        teacher_dist = F.softmax(teacher_embeddings / self.temperature, dim=-1)

        return F.kl_div(student_dist.log(), teacher_dist, reduction="batchmean") * (
            self.temperature**2
        )

    def train_step(self, input_batch: Dict[str, torch.Tensor]) -> float:
        """Perform one training step."""
        self.optimizer.zero_grad()

        total_loss = 0
        batch_size = input_batch["input_ids"].shape[0]

        for i in range(batch_size):
            input_text = self.student_tokenizer.decode(input_batch["input_ids"][i])

            # Get teacher outputs and confidences
            llama_emb, t5_emb, llama_conf, t5_conf = self.get_teacher_outputs(
                input_text
            )

            # Generate student output
            student_output = self.student.generate(
                input_batch["input_ids"][i].unsqueeze(0).to(self.device),
                max_length=100,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
            student_text = self.student_tokenizer.decode(student_output.sequences[0])
            student_emb = self.get_embeddings(student_text)

            # Compute adaptive weights
            llama_weight, t5_weight = self.compute_adaptive_weights(llama_conf, t5_conf)

            # Compute distillation losses
            llama_loss = self.compute_kl_divergence(student_emb, llama_emb)
            t5_loss = self.compute_kl_divergence(student_emb, t5_emb)

            # Combine losses with adaptive weights
            combined_loss = llama_weight * llama_loss + t5_weight * t5_loss
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
