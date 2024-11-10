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


def generate_on_gpu(rank, world_size, problems_chunk, num_samples_per_task=200):
    """Generate completions on a specific GPU"""
    setup(rank, world_size)

    model, tokenizer = load_model_and_tokenizer(rank)
    samples = []

    for task_id, problem in problems_chunk:
        prompt = f"{problem['prompt']}\n"

        # Process in smaller chunks to manage memory
        chunk_size = 20
        completions = []

        for i in range(0, num_samples_per_task, chunk_size):
            current_chunk_size = min(chunk_size, num_samples_per_task - i)

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=200
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=current_chunk_size,
                )

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_completions = [
                output[len(prompt) :].strip() for output in decoded_outputs
            ]
            completions.extend(batch_completions)

            # Clear cache after each chunk
            torch.cuda.empty_cache()

        for completion in completions:
            samples.append(
                {"task_id": task_id, "completion": completion, "gpu_id": rank}
            )

        # Save intermediate results from this GPU
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

    start_time = time.time()

    # Start parallel processing
    mp.spawn(
        generate_on_gpu,
        args=(world_size, problem_chunks[0], 200),  # 200 samples per task
        nprocs=world_size,
        join=True,
    )

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
    print("Evaluating solutions...")
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
        "samples_per_task": 200,
        "average_time_per_problem": total_time / len(problems),
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\nResults:")
    print(f"Total problems: {len(problems)}")
    print(f"Number of GPUs used: {world_size}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per problem: {total_time/len(problems):.2f} seconds")


if __name__ == "__main__":
    run_multi_gpu_benchmark()
