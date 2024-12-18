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


def evaluate_on_gpu(gpu_id: int, problems: List[Dict], output_file: str, model_name: str, cache_dir: str, tokenizer_max_length: int, max_new_tokens: int):
    """Evaluate problems on a specific GPU."""
    setup_gpu_process(gpu_id)
    logger.info(f"Starting evaluation on GPU {gpu_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{gpu_id}",
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions = []
    for problem in tqdm(problems, desc=f"GPU {gpu_id}", position=gpu_id):
        prompt = f"# Complete the following Python function:\n\n{problem['prompt']}"

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max_length,
                return_attention_mask=True,
            ).to(f"cuda:{gpu_id}")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                )

            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            completions.append(
                {"task_id": problem["task_id"], "completion": completion}
            )

        except Exception as e:
            logger.error(
                f"Error processing task {problem['task_id']} on GPU {gpu_id}: {str(e)}"
            )
            completions.append({"task_id": problem["task_id"], "completion": ""})

    lock = FileLock(f"{output_file}.lock")
    with lock:
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    existing_completions = [json.loads(line) for line in f]
            except:
                existing_completions = []

            all_completions = existing_completions + completions
            completion_dict = {c["task_id"]: c for c in all_completions}
            final_completions = list(completion_dict.values())
        else:
            final_completions = completions

        write_jsonl(output_file, final_completions)

    del model
    torch.cuda.empty_cache()


def distribute_problems(problems: Dict, n_gpus: int) -> List[List[Dict]]:
    """Distribute problems evenly across GPUs."""
    problems_list = list(problems.values())
    chunks = np.array_split(problems_list, n_gpus)
    return [chunk.tolist() for chunk in chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--output_file", type=str, default="completions.jsonl", help="Path to save the completions")
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-hf", help="Hugging Face model name or path")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for model cache")
    parser.add_argument("--input_tokenizer_max_length", type=int, default=4096, help="Maximum length for tokenizer input")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    n_gpus = len(gpu_ids)

    problems = read_problems()
    problem_chunks = distribute_problems(problems, n_gpus)

    mp.set_start_method("spawn", force=True)

    processes = []
    for gpu_id, problem_chunk in zip(gpu_ids, problem_chunks):
        p = mp.Process(
            target=evaluate_on_gpu,
            args=(gpu_id, problem_chunk, args.output_file, args.model_name, args.cache_dir, args.input_tokenizer_max_length, args.max_new_tokens),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("All GPUs finished processing. Running test suites...")

    eval_script = f"""
import sys
sys.path.append("/home/brytech/human-eval/human_eval")
from evaluation import evaluate_functional_correctness
import json

def main():
    output_file = "{args.output_file}"
    results = evaluate_functional_correctness(output_file)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pass@1: {{results['pass@1']:.3f}}")

if __name__ == "__main__":
    main()
"""

    with open("run_eval.py", "w") as f:
        f.write(eval_script)

    os.system(f"{sys.executable} run_eval.py")
    os.remove("run_eval.py")


if __name__ == "__main__":
    main()