import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(gpu_id=0):
    """Set up model and tokenizer with proper configuration."""
    # Initialize tokenizer with proper settings
    tokenizer = AutoTokenizer.from_pretrained(
        "WizardLMTeam/WizardCoder-Python-13B-V1.0",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_wizard_coder",
        padding_side="left",  # Ensure consistent padding
        legacy=False,  # Use new behavior
    )
    
    # Ensure proper token settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Ensure left padding for better generation
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        "WizardLMTeam/WizardCoder-Python-13B-V1.0",
        torch_dtype=torch.float16,
        device_map=f"cuda:{gpu_id}",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_wizard_coder",
    )
    
    return model, tokenizer

def generate_completion(model, tokenizer, prompt, gpu_id=0):
    """Generate completion with proper attention mask and padding."""
    # Format prompt
    formatted_prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{prompt}\n\n"
        "### Response:\n"
    )
    
    # Tokenize with proper padding and attention mask
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=True,
    )
    
    # Ensure inputs are on the correct device
    inputs = {k: v.to(f"cuda:{gpu_id}") for k, v in inputs.items()}
    
    # Generate with proper settings
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1,  # Disable beam search since we're using sampling
        early_stopping=False  # Disable early stopping since we're not using beam search
    )
    
    # Decode only the new tokens
    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return completion.strip()

problem = {"task_id": "test/0", "prompt": "def return1():\n", "canonical_solution": "    return 1", "test": "def check(candidate):\n    assert candidate() == 1", "entry_point": "return1"}

def main():
    # Test prompt
    test_prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{problem['prompt']}\n\n"
        "### Response:\n"
    )
    
    # Setup
    gpu_id = 0
    model, tokenizer = setup_model_and_tokenizer(gpu_id)
    
    # Generate
    try:
        completion = generate_completion(model, tokenizer, test_prompt, gpu_id)
        import pdb; pdb.set_trace()
        print("Generated completion:")
        print(completion)
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")

if __name__ == "__main__":
    main()
