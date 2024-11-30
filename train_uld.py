import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorWithPadding

def load_model_and_tokenizer(student_model_name, teacher_model_name):
    """
    Load the student and teacher models and their respective tokenizers.

    ```
    Args:
        student_model_name (str): Name or path of the student model.
        teacher_model_name (str): Name or path of the teacher model.

    Returns:
        tuple: Contains the following elements:
            - student (AutoModelForCausalLM): The loaded student model.
            - teacher (AutoModelForCausalLM): The loaded teacher model.
            - student_tokenizer (AutoTokenizer): Tokenizer for the student model.
            - teacher_tokenizer (AutoTokenizer): Tokenizer for the teacher model.
    """
    student = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights"
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights"
    )

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    # Handle the case where the tokenizers do not explicitly define pad tokens
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    return (
        student,
        teacher,
        student_tokenizer,
        teacher_tokenizer
    )

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


def train_model(student, teacher, student_tokenizer, teacher_tokenizer, dataloader, optimizer, num_epochs, save_dir, lambda_uld):
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        student.train()
        teacher.eval()

        epoch_ce_loss = 0.0  # Cross-Entropy loss accumulator
        epoch_uld_loss = 0.0  # ULD loss accumulator
        epoch_total_loss = 0.0  # Total loss accumulator

        # Batch-wise progress bar
        batch_progress = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}", leave=False, total=len(dataloader))

        for idx, batch in batch_progress:
            student_inputs = batch["student_input_ids"].to(student.device)
            student_attention_mask = batch["student_attention_mask"].to(student.device)  # Move attention mask
            teacher_inputs = batch["teacher_input_ids"].to(teacher.device)
            teacher_attention_mask = batch["teacher_attention_mask"].to(teacher.device)
            labels = batch["labels"].to(student.device)

            # Forward pass with attention mask
            student_output = student(input_ids=student_inputs, attention_mask=student_attention_mask)

            # Compute probabilities
            # teacher_probs = compute_probs(teacher_output.logits)
            student_probs = compute_probs(student_output.logits)

            # Compute Cross-Entropy Loss
            ce_loss = cross_entropy_loss_index_based(
                labels, student_output.logits, pad_token_id=student_tokenizer.pad_token_id
            )

            # Compute ULD Loss
            # uld_loss = compute_uld_loss(teacher_probs, student_probs, lambda_uld)

            # Total Loss
            loss = ce_loss # + uld_loss

            # Check for NaN Loss
            if torch.isnan(loss):
                tqdm.write(f"Skipping Batch {idx} due to NaN loss.")
                continue

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()

            # Update epoch accumulators
            epoch_ce_loss += ce_loss.item()
            # epoch_uld_loss += uld_loss.item()
            epoch_total_loss += loss.item()

            # Update tqdm progress bar
            batch_progress.set_postfix({
                "CE Loss": f"{ce_loss.item():.4f}",
                # "ULD Loss": f"{uld_loss.item():.4f}",
                "Total Loss": f"{loss.item():.4f}"
            })

        # Calculate averages
        avg_ce_loss = epoch_ce_loss / len(dataloader)
        # avg_uld_loss = epoch_uld_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)

        # Log epoch-level averages
        tqdm.write(f"Epoch {epoch+1} completed. Avg CE Loss: {avg_ce_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}")

        # Validation step
        avg_val_loss, perplexity = compute_perplexity(student, val_loader, student_tokenizer, student.device)
        tqdm.write(f"Validation Results - Epoch {epoch+1}: Perplexity: {perplexity:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        # save on the 30th epoch
        if epoch == 29:
            torch.save(student.state_dict(), os.path.join(save_dir, f"student_epoch_{epoch+1}_{perplexity:.4f}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer_epoch_{epoch+1}_{perplexity:.4f}.pt"))

        # Save weights after each epoch
        if epoch % 5 == 0:
            torch.save(student.state_dict(), os.path.join(save_dir, f"student_epoch_{epoch+1}_{perplexity:.4f}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer_epoch_{epoch+1}_{perplexity:.4f}.pt"))


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

            # snippet = row.get("code", None)
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

    # Load models and tokenizers
    student, teacher, student_tokenizer, teacher_tokenizer = load_model_and_tokenizer(
        student_model_name=args.student_name,
        teacher_model_name=args.teacher_name
    )

    # Prepare dataset and dataloader
    train_dataset = CodeSearchNetDataset(
        path=args.train_dataset_path,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
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
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)

    val_dataset = CodeSearchNetDataset(
        path=args.val_dataset_path,  # Path to validation parquet file
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=1024
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, eps=1e-4)

    # Train the model
    train_model(
        student=student,
        teacher=teacher,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        lambda_uld=args.lambda_uld
    )

