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


def load_model_and_tokenizer():
    """Load CodeLlama model and tokenizer"""
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_instruct",
    )
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

            for decoded_output in decoded_outputs:
                # Ensure the completion is properly indented and extract the function body
                indented_output = "\n".join(
                    [
                        "    " + line if line else ""
                        for line in extract_function_body(decoded_output).splitlines()
                    ]
                ).strip()

                all_completions.append([remove_placeholder_comment(indented_output)])

            pbar.update(len(batch_prompts) * num_samples_per_task)

    return all_completions


def format_prompt(problem: Dict) -> str:
    """Return a clean prompt that only includes the function signature."""

    prompt = f"# Complete the following Python function:\n\n{problem['prompt']}"
    return prompt


def estimate_optimal_batch_size(model) -> int:
    """Estimate optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 1

    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    memory_per_sample = 2 * 1024 * 1024 * 1024  # 2GB per sample
    max_batch_size = max(1, math.floor(gpu_memory / memory_per_sample))
    return min(max_batch_size, 8)


def clean_completion(completion: str) -> str:
    """Clean up and format the generated completion."""
    return "\n".join(line.rstrip() for line in completion.splitlines()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="completions.jsonl",
        help="Path to save the completions",
    )
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    batch_size = estimate_optimal_batch_size(model)
    print(f"Using batch size: {batch_size}")

    print("Loading HumanEval problems...")
    problems = read_problems()

    prompts = [format_prompt(problem) for problem in problems.values()]
    task_ids = list(problems.keys())

    samples = []
    start_time = time.time()

    print("Generating solutions...")
    completions_per_task = batch_generate_completions(
        prompts,
        model,
        tokenizer,
        batch_size=batch_size,
        max_new_tokens=1024,
        num_samples_per_task=1,
    )

    # Combine results
    for task_id, completions in zip(task_ids, completions_per_task):
        for completion in completions:
            sample = {"task_id": task_id, "completion": clean_completion(completion)}
            samples.append(sample)

    # Save generations
    write_jsonl(args.output_file, samples)

    # Run evaluation
    print("Evaluating solutions...")
    results = evaluate_functional_correctness(args.output_file)

    # Calculate and save metrics
    total_time = time.time() - start_time
    benchmark_results = {
        "model": "CodeLlama-7b",
        "num_problems": len(problems),
        "total_time": total_time,
        "pass@1": results["pass@1"],
        "batch_size": batch_size,
        "average_time_per_problem": total_time / len(problems),
    }

    with open("results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\nResults:")
    print(f"Total problems: {len(problems)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per problem: {total_time/len(problems):.2f} seconds")
    print(f"Pass@1: {results['pass@1']:.3f}")


if __name__ == "__main__":
    main()
