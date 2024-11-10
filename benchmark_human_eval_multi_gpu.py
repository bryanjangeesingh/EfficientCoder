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

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        device_map=f"cuda:{gpu_id}",
    )
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions = []
    for problem in tqdm(problems, desc=f"GPU {gpu_id}", position=gpu_id):
        # Fix the prompt formatting
        prompt = f"# Complete the following Python function:\n\n{problem['prompt']}"

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                return_attention_mask=True,
            ).to(f"cuda:{gpu_id}")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=1024,
                    do_sample=False,  # Ensure deterministic output
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                )

            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            completions.append(
                {"task_id": problem["task_id"], "completion": completion}
            )

        except Exception as e:
            logger.error(
                f"Error processing task {problem['task_id']} on GPU {gpu_id}: {str(e)}"
            )
            completions.append({"task_id": problem["task_id"], "completion": ""})

    # Create a file lock here, inside the process
    lock = FileLock(f"{output_file}.lock")
    with lock:
        if os.path.exists(output_file):
            # Read existing completions
            try:
                with open(output_file, "r") as f:
                    existing_completions = [json.loads(line) for line in f]
            except:
                existing_completions = []

            # Add new completions
            all_completions = existing_completions + completions

            # Create a dictionary to keep only the latest completion for each task_id
            completion_dict = {c["task_id"]: c for c in all_completions}

            # Convert back to list
            final_completions = list(completion_dict.values())
        else:
            final_completions = completions

        # Write completions
        write_jsonl(output_file, final_completions)

    # Clean up
    del model
    torch.cuda.empty_cache()


def distribute_problems(problems: Dict, n_gpus: int) -> List[List[Dict]]:
    """Distribute problems evenly across GPUs."""
    problems_list = list(problems.values())
    chunks = np.array_split(problems_list, n_gpus)
    return [chunk.tolist() for chunk in chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of GPU IDs to use",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="completions.jsonl",
        help="Path to save the completions",
    )
    args = parser.parse_args()

    # Set up GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    n_gpus = len(gpu_ids)

    # Load HumanEval problems
    problems = read_problems()
    problem_chunks = distribute_problems(problems, n_gpus)

    # Initialize multiprocessing method
    mp.set_start_method("spawn", force=True)

    # Create and start processes
    processes = []
    for gpu_id, problem_chunk in zip(gpu_ids, problem_chunks):
        p = mp.Process(
            target=evaluate_on_gpu, args=(gpu_id, problem_chunk, args.output_file)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Run evaluation in a separate process to avoid any multiprocessing issues
    logger.info("All GPUs finished processing. Running test suites...")

    # Create a temporary script to run evaluation
    eval_script = """
import sys
sys.path.append("/home/brytech/human-eval/human_eval")
from evaluation import evaluate_functional_correctness
import json

def main():
    output_file = "{output_file}"
    results = evaluate_functional_correctness(output_file)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pass@1: {{results['pass@1']:.3f}}")

if __name__ == "__main__":
    main()
""".format(
        output_file=args.output_file
    )

    with open("run_eval.py", "w") as f:
        f.write(eval_script)

    # Run evaluation script
    os.system(f"{sys.executable} run_eval.py")

    # Clean up
    os.remove("run_eval.py")


if __name__ == "__main__":
    main()
