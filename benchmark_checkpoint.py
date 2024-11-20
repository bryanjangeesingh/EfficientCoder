import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from tqdm import tqdm
from typing import List, Dict
import math
import sys
import os
import argparse

# Change the path below to point to the human-eval directory
sys.path.append("/home/brytech/human-eval/human_eval")
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness

def load_model_and_tokenizer(checkpoint_path):
    """Load CodeLlama model and tokenizer with checkpoint weights"""
    model_name = "codellama/CodeLlama-7b-hf"  # Base model, not instruct
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_distilled_student",
    )
    
    # Load checkpoint weights
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["student_state_dict"])
    print("Successfully loaded checkpoint weights")
    
    return model, tokenizer

def extract_function_body(completion: str) -> str:
    """Extract only the body of the function from the generated code."""
    lines = []
    for line in completion.splitlines():
        # Skip empty lines at the start
        if not lines and not line.strip():
            continue
        # Stop if we encounter a line that starts with `if __name__ == "__main__"` or another function definition
        if line.strip().startswith(("if __name__", "def ")):
            break
        lines.append(line)

    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    # Join the extracted lines and strip trailing/leading whitespace
    return "\n".join(lines).strip()

def remove_placeholder_comment(completion: str) -> str:
    """Removes the line '# Your code here\\n' from the given string."""
    lines = completion.splitlines()
    cleaned_lines = [line for line in lines if line.strip() != "# Your code here"]
    return "\n".join(cleaned_lines).strip()

def batch_generate_completions(
    prompts: List[str],
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 1024,
    num_samples_per_task: int = 1,
) -> List[List[str]]:
    """Generate multiple code completions in batches for each prompt."""
    all_completions = []

    with tqdm(
        total=len(prompts) * num_samples_per_task, desc="Generating completions"
    ) as pbar:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                return_attention_mask=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=num_samples_per_task,
                )

            decoded_outputs = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            # Reshape to group samples for each prompt
            reshaped_outputs = [
                decoded_outputs[j : j + num_samples_per_task]
                for j in range(0, len(decoded_outputs), num_samples_per_task)
            ]
            all_completions.extend(reshaped_outputs)

            pbar.update(len(batch_prompts) * num_samples_per_task)

    return all_completions

def format_prompt(problem: Dict) -> str:
    """Format the prompt for code generation."""
    return problem["prompt"].strip()

def estimate_optimal_batch_size(model) -> int:
    """Estimate optimal batch size based on available GPU memory"""
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # Rough estimate - adjust based on your model and input size
    return max(1, min(8, int(gpu_memory / (1024 * 1024 * 1024))))  # Convert to GB

def clean_completion(completion: str) -> str:
    """Clean up and format the generated completion."""
    completion = extract_function_body(completion)
    completion = remove_placeholder_comment(completion)
    return completion

def main():
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark with checkpoint weights")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="checkpoint_completions.jsonl",
        help="Path to save the completions",
    )
    args = parser.parse_args()

    print("Loading model and tokenizer with checkpoint weights...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)

    batch_size = estimate_optimal_batch_size(model)
    print(f"Using batch size: {batch_size}")

    print("Loading HumanEval problems...")
    problems = read_problems()

    # Format prompts for all problems
    prompts = []
    task_ids = []
    for task_id, problem in problems.items():
        prompts.append(format_prompt(problem))
        task_ids.append(task_id)

    print(f"Generating completions for {len(prompts)} problems...")
    start_time = time.time()

    # Generate completions
    completions = batch_generate_completions(
        prompts,
        model,
        tokenizer,
        batch_size=batch_size,
    )

    # Clean up completions
    cleaned_completions = []
    for completion_set in completions:
        cleaned = [clean_completion(c) for c in completion_set]
        cleaned_completions.extend(cleaned)

    # Format results for evaluation
    results = []
    for task_id, completion in zip(task_ids, cleaned_completions):
        results.append(
            {
                "task_id": task_id,
                "completion": completion,
            }
        )

    # Save results
    write_jsonl(args.output_file, results)
    print(f"Saved completions to {args.output_file}")

    # Run evaluation
    results = evaluate_functional_correctness(args.output_file)
    print(f"Results: {results}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
