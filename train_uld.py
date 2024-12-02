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


def load_model_and_tokenizer(student_model_name):
    # Load base model
    student = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights"
    )

    student = prepare_model_for_kbit_training(student)
    
    # Freeze the base model parameters
    for param in student.parameters():
        param.requires_grad = False

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA and prepare for training
    student = get_peft_model(student, lora_config)
    # student.print_trainable_parameters()
    
    # Load tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
        
    return student, student_tokenizer

def compute_probs(logits, temperature=1.0):
    """
    Compute the probabilities of the logits with optional temperature scaling.
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)

def pad_distributions(p, q):
    '''
    Pad the smaller distribution to match the size of the larger distribution.
    Args:
    p: Tensor of shape (batch_size, num_classes_p)
    q: Tensor of shape (batch_size, num_classes_q)

    ```
    Returns:
        p_padded, q_padded: Padded tensors of shape (batch_size, max(num_classes_p, num_classes_q))
    '''
    max_len = max(p.shape[-1], q.shape[-1])
    if p.shape[-1] < max_len:
        p = torch.nn.functional.pad(p, (0, max_len - p.shape[-1]), value=0.0)
    if q.shape[-1] < max_len:
        q = torch.nn.functional.pad(q, (0, max_len - q.shape[-1]), value=0.0)
    return p, q


def compute_wasserstein_distance(p, q):
    '''
    Compute the Wasserstein distance between two distributions
    Args:
    p: Tensor of shape (batch_size, num_classes)
    q: Tensor of shape (batch_size, num_classes)

    ```
    Returns:
        wasserstein_distance: Scalar tensor (averaged across the batch)
    '''
    # # Compute cumulative distributions
    cdf_p = torch.cumsum(p + 1e-12, dim=-1)
    cdf_q = torch.cumsum(q + 1e-12, dim=-1)

    # Wasserstein distance is the sum of absolute differences of CDFs
    wasserstein_per_instance = torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)
    return wasserstein_per_instance.mean()
    # return 0

def cross_entropy_loss_index_based(y_true, logits, pad_token_id):
    # Shift the target sequence by one
    y_true = y_true[:, 1:]
    logits = logits[:, :-1, :]
    
    # Flatten the tensors
    logits = logits.reshape(-1, logits.size(-1))
    y_true = y_true.reshape(-1)
    
    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fn(logits, y_true)

def compute_uld_loss(teacher_probs, student_probs, lambda_uld):
    '''
    Compute the Universal Logit Distillation loss
    Args:
    teacher_probs: Tensor of shape (batch_size, num_classes_teacher)
    student_probs: Tensor of shape (batch_size, num_classes_student)
    lambda_uld: Scalar weighting the Wasserstein term.

    ```
    Returns:
        uld_loss: Scalar tensor
    '''
    # Pad distributions to the same size
    teacher_probs, student_probs = pad_distributions(teacher_probs, student_probs)

    # Sort probabilities for Wasserstein distance
    teacher_probs_sorted = teacher_probs.sort(dim=-1, descending=True)[0]
    student_probs_sorted = student_probs.sort(dim=-1, descending=True)[0]

    # Compute Wasserstein distance
    wasserstein_loss = compute_wasserstein_distance(teacher_probs_sorted, student_probs_sorted)

    # Return the combined ULD loss (note: cross-entropy is computed separately in training loop)
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


def train_model(model, train_dataloader, optimizer, num_epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            input_ids = batch['student_input_ids'].to(model.device)
            attention_mask = batch['student_attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Average loss = {avg_loss:.4f}")
        
        # Save checkpoint
        model.save_pretrained(os.path.join(save_dir, f"checkpoint-{epoch + 1}"))

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
        code = row.get("for_overfitting", None)

        student_instructions = f"# Complete the following Python function: {code}\\n\\n"
        teacher_instructions = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\\n\\n### Instruction:\\n{code}\\n\\n### Response:"

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

            # snippet = row.get("for_overfitting", None)
            # if snippet is None or snippet.strip() == "":
            #     return None

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load model and tokenizer
    student, student_tokenizer = load_model_and_tokenizer(args.student_name)
    print("Trainable parameters:", student.print_trainable_parameters())

    # Prepare dataset and dataloader
    train_dataset = CodeSearchNetDataset(
        path=args.train_dataset_path,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=student_tokenizer,
        max_length=1024
    )

    def collate_fn(batch):
        # Filter out None items
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        # Extract input features
        student_features = [{"input_ids": item["student_input_ids"]} for item in batch]
        teacher_features = [{"input_ids": item["teacher_input_ids"]} for item in batch]
        labels = [item["labels"] for item in batch]

        # Use data collators for dynamic padding (input_ids + attention_mask)
        student_batch = train_dataset.student_data_collator(student_features)
        teacher_batch = train_dataset.teacher_data_collator(teacher_features)

        # Dynamically pad labels to match the longest sequence
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

        # Return padded inputs and masks
        return {
            "student_input_ids": student_batch["input_ids"],
            "student_attention_mask": student_batch["attention_mask"],  # Add attention mask
            "teacher_input_ids": teacher_batch["input_ids"],
            "teacher_attention_mask": teacher_batch["attention_mask"],  # Add attention mask
            "labels": labels,
        }

                                                                                    # TODO: change this
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)

    val_dataset = CodeSearchNetDataset(
        path=args.val_dataset_path,  # Path to validation parquet file
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=student_tokenizer,
        max_length=1024
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, drop_last=False)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, eps=1e-4)

    # Train the model
    train_model(
        model=student,
        train_dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )

