import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        torch_dtype=torch.float16,
        device_map="auto"
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

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
    """
    Computes the cross-entropy loss between tokenized ground truth and student logits.

    Args:
        y_true (torch.Tensor): Tokenized ground truth tensor with shape (batch_size, seq_len).
        logits (torch.Tensor): Student logits with shape (batch_size, seq_len, vocab_size).
        pad_token_id (int): Token ID used for padding in the dataset.

    Returns:
        torch.Tensor: Scalar value representing the mean cross-entropy loss.
    """
    # Flatten the tensors for CrossEntropyLoss
    logits = logits.view(-1, logits.size(-1))
    y_true = y_true.view(-1)
    
    # Compute cross-entropy ignoring the padding tokens
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


def train_model(student, teacher, student_tokenizer, teacher_tokenizer, dataloader, optimizer, num_epochs, lambda_uld=0.1):
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        student.train()
        teacher.eval()

        for idx, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}", leave=False):
            tokenized_labels = student_tokenizer(
                batch["labels"],
                truncation=True,
                padding=True,
                return_tensors="pt"
            )["input_ids"].to(student.device)

            with torch.no_grad():
                teacher_output = teacher(batch)

            student_output = student(batch)

            teacher_probs = compute_probs(teacher_output.logits)
            student_probs = compute_probs(student_output.logits)

            ce_loss = cross_entropy_loss_index_based(
                tokenized_labels, student_output.logits, student_tokenizer.pad_token_id
            )

            uld_loss = compute_uld_loss(teacher_probs, student_probs, lambda_uld)
            total_loss = ce_loss + uld_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}, Batch {idx+1}, Loss: {total_loss.item():.4f}")


# Create a dataset class for CodeNala 

class CodeNalaDataset(Dataset):
    def __init__(self, path, student_tokenizer, teacher_tokenizer, max_length=512):
        # path contains a folder containing a train parquet and a test parquet
        self.path = path
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length
        



