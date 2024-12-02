import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_models_and_tokenizers(student_model_name, teacher_model_name, checkpoint_path=None):
    # Load student model
    student = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights"
    )
    student = prepare_model_for_kbit_training(student)

    # Load teacher model
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights"
    )

    # Load tokenizers
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Load checkpoint for student if provided
    if checkpoint_path is not None:
        from peft import PeftModel
        student = PeftModel.from_pretrained(student, checkpoint_path)
        # do a logger here saying that it loaded the peft model 
        print(f"Loaded PEFT checkpoint from {checkpoint_path}")

    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        student = get_peft_model(student, lora_config)

    # Freeze base model parameters and ensure LoRA parameters require gradients
    for name, param in student.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return student, teacher, student_tokenizer, teacher_tokenizer


def compute_probs(logits, temperature=1.0):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)


def pad_distributions(p, q):
    max_len = max(p.shape[-1], q.shape[-1])
    if p.shape[-1] < max_len:
        p = torch.nn.functional.pad(p, (0, max_len - p.shape[-1]), value=0.0)
    if q.shape[-1] < max_len:
        q = torch.nn.functional.pad(q, (0, max_len - q.shape[-1]), value=0.0)
    return p, q

def truncate_output_sequences(teacher_output, student_output):
    # teacher_output has shape [batch_size, seq_len, vocab_size]
    # truncate the output sequence which is dim=1 to the minimum of either teacher or student length
    min_length = min(teacher_output.shape[1], student_output.shape[1])
    teacher_output = teacher_output[:, :min_length, :]
    student_output = student_output[:, :min_length, :]
    return teacher_output, student_output
    

def compute_wasserstein_distance(p, q):
    # p looks like torch.Size([2, 255, 32016])
    # q looks like torch.Size([2, 255, 32016])

    wasserstein_per_instance = torch.sum(torch.abs(p - q), dim=-1)
    return wasserstein_per_instance.mean()


# def cross_entropy_loss_index_based(y_true, logits, pad_token_id):
#     # Shift the target sequence by one
#     y_true = y_true[:, 1:]
#     logits = logits[:, :-1, :]
    
#     # Flatten the tensors
#     logits = logits.reshape(-1, logits.size(-1))
#     y_true = y_true.reshape(-1)
    
#     # Define the loss function
#     loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
#     return loss_fn(logits, y_true)


def compute_uld_loss(teacher_probs, student_probs, lambda_uld):
    # teacher probs shape is [batch_size, seq_len, teacher_vocab_size]
    # student probs shape is [batch_size, seq_len, student_vocab_size]

    # pad distributions to the same vocab length
    student_probs, teacher_probs = pad_distributions(student_probs, teacher_probs)
    # (Pdb) student_probs.shape
    # torch.Size([2, 375, 32016])
    # (Pdb) teacher_probs.shape
    # torch.Size([2, 375, 32016])
    # the two sequences now have the same vocab length

    teacher_probs_sorted = teacher_probs.sort(dim=-1, descending=True)[0]
    student_probs_sorted = student_probs.sort(dim=-1, descending=True)[0]

    wasserstein_loss = compute_wasserstein_distance(teacher_probs_sorted, student_probs_sorted)
    return lambda_uld * wasserstein_loss



def compute_perplexity(model, dataloader, tokenizer, device):
    """
    Compute perplexity of a model on the validation dataset.

    ```
    Args:
        model: The student model.
        dataloader: Validation dataloader.
        tokenizer: Tokenizer for the model.
        device: Device for computation.

    Returns:
        tuple: (average loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    num_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            inputs = batch["student_input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with labels for automatic loss calculation
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss  # Cross-entropy loss provided by HuggingFace models

            # Accumulate loss and token count
            total_loss += loss.item() * labels.size(1)  # Multiply by sequence length
            num_tokens += labels.numel()

    avg_loss = total_loss / num_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def train_model(student, teacher, train_dataloader, optimizer, num_epochs, save_dir, student_tokenizer, lambda_uld, temperature):
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        student.train()
        teacher.eval()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            input_ids = batch['student_input_ids'].to(student.device)
            student_attention_mask = batch['student_attention_mask'].to(student.device)
            labels = batch['labels'].to(student.device)
            teacher_inputs = batch['teacher_input_ids'].to(teacher.device)
            teacher_attention_mask = batch['teacher_attention_mask'].to(teacher.device)

            with torch.no_grad():
                teacher_outputs = teacher(input_ids=teacher_inputs, attention_mask=teacher_attention_mask)

            student_outputs = student(input_ids=input_ids, attention_mask=student_attention_mask, labels=labels)

            teacher_probs = compute_probs(teacher_outputs.logits, temperature)
            # shape is [seq_len, vocab_size]
            student_probs = compute_probs(student_outputs.logits, temperature)

            student_probs, teacher_probs = truncate_output_sequences(teacher_probs, student_probs)
            # (Pdb) student_probs.shape
            # torch.Size([2, 151, 32001])
            # (Pdb) teacher_probs.shape
            # torch.Size([2, 151, 32016])
            # the outputs now have the same sequence length

            cross_entropy_loss = student_outputs.loss
            
            uld_loss = compute_uld_loss(teacher_probs, student_probs, lambda_uld)

            total_loss_batch = cross_entropy_loss + uld_loss

            total_loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += total_loss_batch.item()
            progress_bar.set_postfix({'loss': total_loss_batch.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Average loss = {avg_loss:.4f}")

        student.save_pretrained(os.path.join(save_dir, f"checkpoint-{epoch + 1}"))

class CodeSearchNetDataset(Dataset):
    def __init__(self, path, student_tokenizer, teacher_tokenizer, max_length):
        """
        Args:
        path (str): Path to the folder containing train and test parquet files.
        student_tokenizer (AutoTokenizer): Tokenizer for the student model.
        teacher_tokenizer (AutoTokenizer): Tokenizer for the teacher model.
        max_length (int): Maximum sequence length for tokenization.
        """
        self.data = pd.read_parquet(path)  # Load the dataset from the parquet file

        self.student_tokenizer = student_tokenizer
        # IMPORTANT
        self.student_tokenizer.padding_side = "right"

        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_tokenizer.padding_side = "right"

        self.max_length = max_length

        self.student_data_collator = DataCollatorWithPadding(
            tokenizer=student_tokenizer,
            padding=True,
            return_tensors="pt"
        )

        self.teacher_data_collator = DataCollatorWithPadding(
            tokenizer=teacher_tokenizer,
            padding=True,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = row.get("code", None)

        student_instructions = f"# Complete the following Python function:\n{code}"
        # teacher_instructions = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{code}\n\n### Response:"
        teacher_instructions = f"# Complete the following Python function:\n{code}"

        try:
            # Tokenize instructions for student and teacher with attention masks
            student_tokenized = self.student_tokenizer(
                student_instructions,
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Padding is handled in the collate_fn or data collator
                return_tensors=None
            )

            teacher_tokenized = self.teacher_tokenizer(
                teacher_instructions,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )

            # Tokenize labels
            labels = self.student_tokenizer(
                student_instructions,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )["input_ids"]

            return {
                "student_input_ids": student_tokenized["input_ids"],
                "student_attention_mask": student_tokenized.get("attention_mask"),  # Include attention mask
                "teacher_input_ids": teacher_tokenized["input_ids"],
                "teacher_attention_mask": teacher_tokenized.get("attention_mask"),  # Include attention mask
                "labels": labels
            }
        except Exception as e:
            return None
        
def parse_args():
    parser = argparse.ArgumentParser(description="Train a student model with ULD.")

    # Model names
    parser.add_argument("--student_name", type=str, required=True, help="Name or path of the student model.")
    parser.add_argument("--teacher_name", type=str, required=True, help="Name or path of the teacher model.")

    # Training configuration
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for scaling logits.")
    parser.add_argument("--lambda_uld", type=float, default=0.1, help="Weight for the Wasserstein loss term.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")

    # Model saving and precision
    parser.add_argument("--load_in_4bit", action="store_true", help="Whether to load the models in 4-bit precision.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the trained student model weights.")

    # Dataset path
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training dataset (parquet file).")
    parser.add_argument("--val_dataset_path", type=str, required=True, help="Path to the validation dataset (parquet file).")

    # Checkpoint loading and saving
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the folder containing the previous checkpoint to load.")
    parser.add_argument("--start_checkpoint", type=int, default=0, help="Starting checkpoint number for saving new checkpoints.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    student, teacher, student_tokenizer, teacher_tokenizer = load_models_and_tokenizers(
        args.student_name, args.teacher_name, args.checkpoint_path)

    train_dataset = CodeSearchNetDataset(
        path=args.train_dataset_path,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=1024
    )

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        student_features = [{"input_ids": item["student_input_ids"]} for item in batch]
        teacher_features = [{"input_ids": item["teacher_input_ids"]} for item in batch]
        labels = [item["labels"] for item in batch]

        student_batch = train_dataset.student_data_collator(student_features)
        teacher_batch = train_dataset.teacher_data_collator(teacher_features)

        max_input_length = student_batch["input_ids"].shape[1]
        padded_labels = [
            F.pad(
                torch.tensor(label),
                (0, max_input_length - len(label)),
                value=student_tokenizer.pad_token_id
            )[:max_input_length]
            for label in labels
        ]
        labels = torch.stack(padded_labels)

        return {
            "student_input_ids": student_batch["input_ids"],
            "student_attention_mask": student_batch["attention_mask"],
            "teacher_input_ids": teacher_batch["input_ids"],
            "teacher_attention_mask": teacher_batch["attention_mask"],
            "labels": labels,
        }

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, eps=1e-4)

    train_model(
        student=student,
        teacher=teacher,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        student_tokenizer=student_tokenizer,
        lambda_uld=args.lambda_uld,
        temperature=args.temperature
    )
