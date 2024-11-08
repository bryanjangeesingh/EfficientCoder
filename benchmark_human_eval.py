import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from tqdm import tqdm
from typing import List, Dict
import math
import sys
import os

sys.path.append('/home/brytech/human-eval/human_eval')
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness


def load_model_and_tokenizer():
    """Load CodeLlama model and tokenizer"""
    model_name = "codellama/CodeLlama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def batch_generate_completions(
    prompts: List[str],
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    num_samples_per_task: int = 3  # New parameter for number of completions
) -> List[List[str]]:
    """Generate multiple code completions in batches for each prompt"""
    all_completions = []

    # Initialize progress bar for batch processing
    with tqdm(total=len(prompts) * num_samples_per_task, desc="Generating completions") as pbar:
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=200
            ).to(model.device)

            # Generate completions
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=num_samples_per_task  # Generate multiple completions per input
                )

            # Decode completions
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Process completions per prompt
            num_prompts = len(batch_prompts)
            for j in range(num_prompts):
                task_completions = decoded_outputs[j * num_samples_per_task:(j + 1) * num_samples_per_task]
                all_completions.append([output[len(batch_prompts[j]):].strip() for output in task_completions])

            # Update the progress bar
            pbar.update(num_prompts * num_samples_per_task)

    return all_completions

def format_prompt(problem: Dict) -> str:
    """Format HumanEval problem into prompt"""
    return f"{problem['prompt']}\n"

def estimate_optimal_batch_size(model) -> int:
    """Estimate optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 1
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # Rough estimation - adjust these values based on your specific model
    memory_per_sample = 2 * 1024 * 1024 * 1024  # 2GB per sample
    max_batch_size = max(1, math.floor(gpu_memory / memory_per_sample))
    return min(2, 8)  # Cap at 8 to avoid potential issues

def run_benchmark():
    """Run HumanEval benchmark on CodeLlama"""
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # Estimate optimal batch size
    batch_size = estimate_optimal_batch_size(model)
    print(f"Using batch size: {batch_size}")

    print("Loading HumanEval problems...")
    problems = read_problems()

    # Prepare all prompts
    prompts = [format_prompt(problem) for problem in problems.values()]
    task_ids = list(problems.keys())

    num_samples_per_task = 3  # Number of completions per task
    samples = []
    start_time = time.time()

    print("Generating solutions...")
    completions_per_task = batch_generate_completions(
        prompts,
        model,
        tokenizer,
        batch_size=batch_size,
        num_samples_per_task=num_samples_per_task
    )

    # Combine results
    for task_id, completions in zip(task_ids, completions_per_task):
        for completion in completions:
            sample = {
                "task_id": task_id,
                "completion": completion
            }
            samples.append(sample)

    # Save generations
    output_file = "codellama_completions.jsonl"
    write_jsonl(output_file, samples)

    # Evaluate results
    print("Evaluating solutions...")
    results = evaluate_functional_correctness(output_file)

    # Calculate metrics
    total_time = time.time() - start_time
    pass_k = results["pass@1"]

    # Save results
    benchmark_results = {
        "model": "CodeLlama-7b",
        "num_problems": len(problems),
        "total_time": total_time,
        "pass@k": pass_k,
        "batch_size": batch_size,
        "average_time_per_problem": total_time / len(problems)
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\nResults:")
    print(f"Total problems: {len(problems)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per problem: {total_time/len(problems):.2f} seconds")
    print(f"pass@1: {pass_k[1]:.3f}")
    print(f"pass@10: {pass_k[10]:.3f}")

if __name__ == "__main__":
    run_benchmark()