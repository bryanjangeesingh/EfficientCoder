import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import logging
import argparse
import sys
import numpy as np
from typing import List, Dict
from filelock import FileLock

sys.path.append("/home/brytech/human-eval/human_eval")
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu_process(rank):
    """Set up process-specific GPU settings."""
    torch.cuda.set_device(rank)


def evaluate_on_gpu(gpu_id: int, problems: List[Dict], output_file: str):
    """Evaluate problems on a specific GPU."""
    setup_gpu_process(gpu_id)
    logger.info(f"Starting evaluation on GPU {gpu_id}")

    # Initialize model and tokenizer for WizardCoder
    model = AutoModelForCausalLM.from_pretrained(
        "WizardLMTeam/WizardCoder-Python-13B-V1.0",
        torch_dtype=torch.float16,
        device_map=f"cuda:{gpu_id}",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_wizard_coder",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "WizardLMTeam/WizardCoder-Python-13B-V1.0",
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_wizard_coder"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions = []
    for problem in tqdm(problems, desc=f"GPU {gpu_id}", position=gpu_id):
        # Format prompt according to WizardCoder's style
        prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{problem['prompt']}\n\n"
            "### Response:\n"
        )

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            ).to(f"cuda:{gpu_id}")

            # Generate with WizardCoder-specific parameters
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Clean up the completion
            completion = completion.strip()
            
            if completion.startswith("Here's a solution") or completion.startswith("Here is"):
                # Remove the introductory text
                completion = completion.split("\n", 1)[1].strip()
            
            if completion.startswith("```python"):
                # Remove code block markers
                completion = completion.replace("```python", "").replace("```", "").strip()
            
            valid_completion = None
            if completion.startswith("def"):
                # If completion starts with function definition, use as is
                valid_completion = completion
            elif "def" in completion:
                # If "def" is somewhere in the completion, extract from there
                valid_completion = completion[completion.index("def"):]
            
            if valid_completion is None:
                # If no function definition found, create a default failing function
                logger.warning(f"No function definition found in completion for task {problem['task_id']}. Creating default function.")
                # Extract function signature from the prompt
                prompt_lines = problem['prompt'].split('\n')
                function_def = next((line for line in prompt_lines if line.strip().startswith('def ')), None)
                if function_def:
                    # Create a function that raises NotImplementedError
                    valid_completion = f"{function_def}\n    raise NotImplementedError('No valid completion generated')"
                else:
                    logger.error(f"Could not find function definition in prompt for task {problem['task_id']}")
                    valid_completion = "def dummy():\n    raise NotImplementedError('No valid completion generated')"
            
            # Remove any additional content after the function
            if "\n\n" in valid_completion:
                valid_completion = valid_completion.split("\n\n")[0]

            completions.append(
                dict(task_id=problem["task_id"], completion=valid_completion)
            )

            # Save intermediate results
            with FileLock(f"{output_file}.lock"):
                write_jsonl(output_file, completions[-1:])  # Only write the last completion

        except Exception as e:
            logger.error(f"Error processing task {problem['task_id']}: {str(e)}")
            continue

        # Clear GPU cache periodically
        if len(completions) % 10 == 0:
            torch.cuda.empty_cache()

    return completions


def distribute_problems(problems: Dict, n_gpus: int) -> List[List[Dict]]:
    """Distribute problems evenly across GPUs."""
    problems_list = list(problems.values())
    chunks = np.array_split(problems_list, n_gpus)
    return [chunk.tolist() for chunk in chunks]


def main():
    parser = argparse.ArgumentParser(description="Evaluate WizardCoder on HumanEval")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./wizard_coder_completions.jsonl",
        help="Path to save completions"
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use"
    )
    args = parser.parse_args()

    # Load HumanEval problems
    problems = read_problems()

    # Get number of available GPUs
    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    logger.info(f"Using {n_gpus} GPUs")

    # Split problems across GPUs
    problem_splits = distribute_problems(problems, n_gpus)

    # Start multiprocessing
    processes = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=evaluate_on_gpu,
            args=(gpu_id, problem_splits[gpu_id], args.output_path)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Evaluate the results
    logger.info("Evaluating results...")
    results = evaluate_functional_correctness(args.output_path)
    
    # Save metrics
    metrics_file = args.output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {metrics_file}")
    logger.info(f"Pass@1: {results['pass@1']:.3f}")


if __name__ == "__main__":
    main()
