import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from concurrent.futures import ProcessPoolExecutor
import json
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import logging
import argparse
import sys

sys.path.append("/home/brytech/human-eval/human_eval")
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model(gpu_id):
    """Initialize Code Llama model on specified GPU."""
    device = f"cuda:{gpu_id}"
    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        device_map=device,
        cache_dir="/nobackup/users/brytech/projects/condas/nlp_4gpus/weights-7b-hf",
    )
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    return model, tokenizer, device


def generate_completion(prompt, model, tokenizer, device, max_new_tokens=512):
    """Generate code completion for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Deterministic for pass@1
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )

    completion = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return completion.strip()


def process_problem(args):
    """Process a single HumanEval problem."""
    problem, gpu_id = args
    model, tokenizer, device = setup_model(gpu_id)

    # Construct prompt
    prompt = f"# Complete the following Python function:\n\n{problem['prompt']}"

    # Generate completion
    completion = generate_completion(prompt, model, tokenizer, device)

    return {"task_id": problem["task_id"], "completion": completion}


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

    # Distribute problems across GPUs
    problem_gpu_pairs = [
        (prob, gpu_ids[i % n_gpus]) for i, prob in enumerate(problems.values())
    ]

    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)

    # Process problems in parallel
    logger.info(f"Starting evaluation on {n_gpus} GPUs...")
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        completions = list(
            tqdm(
                executor.map(process_problem, problem_gpu_pairs),
                total=len(problem_gpu_pairs),
            )
        )

    # Save completions
    write_jsonl(args.output_file, completions)

    # Evaluate pass@1
    results = evaluate_functional_correctness(args.output_file)

    # Print and save results
    logger.info(f"Pass@1: {results['pass@1']:.3f}")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
