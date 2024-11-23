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


def distribute_problems(problems: Dict, n_gpus: int) -> List[List[Dict]]:
    """Distribute problems evenly across GPUs."""
    problems_list = sorted(problems.items())  # Sort by task_id to ensure consistent ordering
    chunks = np.array_split(problems_list, n_gpus)
    return [dict(chunk) for chunk in chunks]


def evaluate_on_gpu(gpu_id: int, problems: Dict, output_file: str):
    """Evaluate problems on a specific GPU."""
    setup_gpu_process(gpu_id)
    logger.info(f"Starting evaluation on GPU {gpu_id} with {len(problems)} problems")

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

    # Track attempted problems to ensure we don't miss any
    attempted_problems = set()
    
    for task_id, problem in tqdm(problems.items(), desc=f"GPU {gpu_id}", position=gpu_id):
        if task_id in attempted_problems:
            logger.warning(f"Problem {task_id} was already attempted. Skipping.")
            continue
            
        try:
            # Format prompt according to WizardCoder's style
            prompt = (
                "Below is the start of a python program. Continue it.\n\n"
                "### Instruction:\n"
                f"{problem['prompt']}\n\n"
                "### Response:\n"
            )

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
                completion = completion.split("\n", 1)[1].strip()
            
            if completion.startswith("```python"):
                completion = completion.replace("```python", "").replace("```", "").strip()
            
            # Extract just the function body
            def extract_function_body(completion: str, function_def: str) -> str:
                """Extract the function body from the completion."""
                # If completion starts with the function definition, remove it
                if completion.startswith(function_def.strip()):
                    # Split by newline and remove the first line (function def)
                    body_lines = completion.split("\n")[1:]
                elif "def " in completion:
                    # Find the function body after any "def" line
                    body_lines = completion[completion.index("def "):].split("\n")[1:]
                else:
                    # If no function def found, treat the whole completion as body
                    body_lines = completion.split("\n")
                
                # Remove any empty lines at the start
                while body_lines and not body_lines[0].strip():
                    body_lines.pop(0)
                
                # Handle indentation
                if body_lines:
                    # Get the indentation of the first non-empty line
                    first_line = body_lines[0]
                    indent = len(first_line) - len(first_line.lstrip())
                    # Remove that amount of indentation from all lines
                    body_lines = [line[indent:] if line.startswith(" " * indent) else line for line in body_lines]
                
                return "\n".join(body_lines).rstrip()
            
            valid_completion = None
            try:
                # Extract the function body
                valid_completion = extract_function_body(completion, problem['prompt'])
                
                # Add proper indentation (4 spaces) to each line
                valid_completion = "\n".join(f"    {line}" if line.strip() else line 
                                          for line in valid_completion.split("\n"))
                
                # Verify the completion isn't empty
                if not valid_completion.strip():
                    raise ValueError("Empty function body")
                
            except Exception as e:
                logger.warning(f"Error processing completion for task {task_id}: {str(e)}")
                # Create a failing function body
                valid_completion = "    raise NotImplementedError('No valid completion generated')"
            
            result = dict(task_id=task_id, completion=valid_completion)
            
            # Save result atomically
            with FileLock(f"{output_file}.lock"):
                write_jsonl(output_file, [result])
            
            attempted_problems.add(task_id)
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            # Create a failing completion for this task
            result = dict(
                task_id=task_id,
                completion="def dummy():\n    raise NotImplementedError('Error during generation')"
            )
            with FileLock(f"{output_file}.lock"):
                write_jsonl(output_file, [result])
            attempted_problems.add(task_id)

    logger.info(f"GPU {gpu_id} completed {len(attempted_problems)} problems")


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
    
    # Clear the output file if it exists
    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    # Get number of available GPUs
    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    logger.info(f"Using {n_gpus} GPUs")

    # Split problems across GPUs
    problem_splits = distribute_problems(problems, n_gpus)
    total_problems = sum(len(split) for split in problem_splits)
    logger.info(f"Distributed {total_problems} problems across {n_gpus} GPUs")

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

    # Verify all problems were attempted
    with open(args.output_path, 'r') as f:
        completions = [json.loads(line) for line in f]
    completed_tasks = {c['task_id'] for c in completions}
    all_tasks = set(problems.keys())
    missing_tasks = all_tasks - completed_tasks
    
    if missing_tasks:
        logger.error(f"Missing completions for tasks: {missing_tasks}")
        # Add dummy completions for missing tasks
        dummy_completions = [
            dict(
                task_id=task_id,
                completion="def dummy():\n    raise NotImplementedError('Task not attempted')"
            )
            for task_id in missing_tasks
        ]
        with open(args.output_path, 'a') as f:
            for completion in dummy_completions:
                f.write(json.dumps(completion) + '\n')

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
