import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from tqdm import tqdm
from typing import List, Dict
import math
import sys
import os
import torch.multiprocessing as mp

sys.path.append("/home/brytech/human-eval/human_eval")
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness


def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()


def load_model_and_tokenizer(rank):
    """Load CodeLlama model and tokenizer with multi-GPU support"""
    model_name = "codellama/CodeLlama-7b-hf"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with specific GPU assignment
    device = torch.device(f"cuda:{rank}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": rank},  # Assign to specific GPU
    )

    return model, tokenizer


def generate_prompt(problem):
    """Generate prompt for the model"""
    return f"""# Here's a Python function that {problem['description']}
{problem['prompt']}
"""


def generate_on_gpu(rank, world_size, problems_chunk, progress_queue=None):
    """Generate completions on a specific GPU"""
    setup(rank, world_size)
    model, tokenizer = load_model_and_tokenizer(rank)
    samples = []

    for task_id, problem in problems_chunk:
        prompt = generate_prompt(problem)

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,  # Greedy decoding for pass@1
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        completion = decoded_output[len(prompt):].strip()

        samples.append({"task_id": task_id, "completion": completion, "gpu_id": rank})

        if progress_queue is not None:
            progress_queue.put(1)

        # Save intermediate results
        with open(f"completions_gpu_{rank}.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    cleanup()
    return samples


def run_multi_gpu_benchmark():
    """Run HumanEval benchmark using multiple GPUs"""
    print("Starting multi-GPU benchmark...")

    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    # Load problems
    problems = read_problems()

    # Split problems across GPUs
    problems_list = list(problems.items())
    chunk_size = math.ceil(len(problems_list) / world_size)
    problem_chunks = [
        problems_list[i : i + chunk_size]
        for i in range(0, len(problems_list), chunk_size)
    ]

    # Create progress queue and progress bar
    progress_queue = mp.Queue()
    total_problems = len(problems)

    # Create and start progress monitoring process
    def monitor_progress():
        pbar = tqdm(total=total_problems, desc="Overall Progress")
        completed = 0
        while completed < total_problems:
            items = progress_queue.get()
            completed += items
            pbar.update(items)
        pbar.close()

    progress_process = mp.Process(target=monitor_progress)
    progress_process.start()

    start_time = time.time()

    # Start parallel processing
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=generate_on_gpu,
            args=(rank, world_size, problem_chunks[rank], progress_queue),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Wait for progress bar to finish
    progress_process.join()

    # Combine results from all GPUs
    all_samples = []
    for rank in range(world_size):
        filename = f"completions_gpu_{rank}.jsonl"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    all_samples.append(json.loads(line))
            os.remove(filename)  # Clean up intermediate files

    # Save final results
    output_file = "codellama_completions.jsonl"
    write_jsonl(output_file, all_samples)

    # Evaluate results
    print("\nEvaluating solutions...")
    results = evaluate_functional_correctness(output_file)

    # Calculate metrics
    total_time = time.time() - start_time
    pass_k = results["pass@1"]

    # Save benchmark results
    benchmark_results = {
        "model": "CodeLlama-7b",
        "num_problems": len(problems),
        "num_gpus": world_size,
        "total_time": total_time,
        "pass@k": pass_k,
        "samples_per_task": 1,
        "average_time_per_problem": total_time / len(problems),
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\nResults:")
    print(f"Total problems: {len(problems)}")
    print(f"Number of GPUs used: {world_size}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per problem: {total_time/len(problems):.2f} seconds")
    print(f"Pass@1: {pass_k:.2%}")


if __name__ == "__main__":
    run_multi_gpu_benchmark()