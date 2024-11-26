import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(student_model_name, teacher_model_name):
    """
    Load the student and teacher models and their respective tokenizers.

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
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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

    Returns:
        wasserstein_distance: Scalar tensor (averaged across the batch)
    '''
    # Compute cumulative distributions
    cdf_p = torch.cumsum(p + 1e-12, dim=-1)
    cdf_q = torch.cumsum(q + 1e-12, dim=-1)
    
    # Wasserstein distance is the sum of absolute differences of CDFs
    wasserstein_per_instance = torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)
    return wasserstein_per_instance.mean()

def cross_entropy_loss_index_based(y_true, logits, pad_token_id):
    logits = logits.view(-1, logits.size(-1))
    y_true = y_true.view(-1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fn(logits, y_true)

def compute_uld_loss(teacher_probs, student_probs, lambda_uld=1.0):
    '''
    Compute the Universal Logit Distillation loss
    Args:
        teacher_probs: Tensor of shape (batch_size, num_classes_teacher)
        student_probs: Tensor of shape (batch_size, num_classes_student)
        lambda_uld: Scalar weighting the Wasserstein term.

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

def train_model(student, teacher, student_tokenizer, teacher_tokenizer, dataloader, optimizer, num_epochs, save_dir, lambda_uld=0.1):
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        student.train()
        teacher.eval()
        total_loss = 0

    for idx, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}", leave=False):
        # Move batch inputs to the correct device
        student_inputs = batch["student_input_ids"].to(student.device)
        teacher_inputs = batch["teacher_input_ids"].to(teacher.device)
        labels = batch["labels"].to(student.device)

        with torch.no_grad():
            teacher_output = teacher(input_ids=teacher_inputs)

        student_output = student(input_ids=student_inputs)

        # Compute probabilities
        teacher_probs = compute_probs(teacher_output.logits)
        student_probs = compute_probs(student_output.logits)
       # breakpoint()
        # Compute cross-entropy loss
        ce_loss = cross_entropy_loss_index_based(
            labels, student_output.logits, student_tokenizer.pad_token_id
        )

        # Compute ULD loss
        uld_loss = compute_uld_loss(teacher_probs, student_probs, lambda_uld)
        loss = ce_loss + uld_loss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}, Batch {idx+1}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save weights after each epoch
    torch.save(student.state_dict(), os.path.join(save_dir, f"student_epoch_{epoch+1}.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer_epoch_{epoch+1}.pt"))


# Create a dataset class for CodeNala 

class CodeNalaDataset(Dataset):
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
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an individual data point.
        Args:
            idx (int): Index of the data point.
        Returns:
            dict or None: A dictionary containing tokenized data, or None if the row is invalid.
        """
        row = self.data.iloc[idx]

        # Check if rewritten_intent is problematic
        rewritten_intent = row.get("rewritten_intent", None)
        if (
            rewritten_intent is None or  # Check if null
            rewritten_intent.strip() == "" or  # Check if empty string
            isinstance(rewritten_intent, float) and pd.isnull(rewritten_intent)  # Handle NaN
        ):
            # Switch to intent if rewritten_intent is problematic
            rewritten_intent = row.get("intent", None)
            if (
                rewritten_intent is None or  # Check if intent is null
                rewritten_intent.strip() == "" or  # Check if empty string
                isinstance(rewritten_intent, float) and pd.isnull(rewritten_intent)  # Handle NaN
            ):
                # Skip this row if intent is also problematic
                return None

        # Tokenize intent for both student and teacher models
        try:
            student_tokenized = self.student_tokenizer(
                rewritten_intent,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            teacher_tokenized = self.teacher_tokenizer(
                rewritten_intent,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            # Skip row if tokenization fails
            return None

        # Tokenize snippet (used as labels)
        snippet = row.get("snippet", None)
        if snippet is None or snippet.strip() == "":
            return None

        try:
            labels = self.student_tokenizer(
                snippet,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
        except Exception as e:
            return None

        return {
            "student_input_ids": student_tokenized["input_ids"].squeeze(0),
            "teacher_input_ids": teacher_tokenized["input_ids"].squeeze(0),
            "labels": labels.squeeze(0)
        }

def collate_fn(batch):
    """
    Custom collate function to filter out None rows.
    Args:
        batch (list): A list of items returned by the Dataset's __getitem__.
    Returns:
        list: Filtered batch with None rows removed.
    """
    # Filter out None rows
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Handle edge case where the entire batch is invalid
    return {
        "student_input_ids": torch.stack([item["student_input_ids"] for item in batch]),
        "teacher_input_ids": torch.stack([item["teacher_input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }

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
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset (parquet file).")
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # Load models and tokenizers
    student, teacher, student_tokenizer, teacher_tokenizer = load_model_and_tokenizer(
        student_model_name=args.student_name,
        teacher_model_name=args.teacher_name
    )

    # Prepare dataset and dataloader
    dataset = CodeNalaDataset(
        path=args.dataset_path,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=32
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)

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