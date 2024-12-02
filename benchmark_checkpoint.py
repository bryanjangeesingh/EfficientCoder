import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import logging
import argparse
import sys
import numpy as np
from typing import List, Dict
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
sys.path.append("/home/brytech/human-eval/human_eval")
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness
from filelock import FileLock
import torch.multiprocessing as mp
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_gpu_process(rank):
    """Set up process-specific GPU settings."""
    torch.cuda.set_device(rank)


def extract_function_body(completion: str) -> str:
    """Extract only the body of the function from the generated code."""
    lines = []
    for line in completion.splitlines():
        if not lines and not line.strip():
            continue
        if line.strip().startswith(("if __name__", "def ")):
            break
        lines.append(line)

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines).strip()


def remove_placeholder_comment(completion: str) -> str:
    """Removes the line '# Your code here\\n' from the given string."""
    lines = completion.splitlines()
    cleaned_lines = [line for line in lines if line.strip() != "# Your code here"]
    return "\n".join(cleaned_lines).strip()


def evaluate_on_gpu(
    gpu_id: int,
    problems: List[Dict],
    base_model_path: str,
    checkpoint_path: str,
    output_file: str,
    cache_dir: str,
    use_peft: bool,
):
    """Evaluate problems on a specific GPU."""
    setup_gpu_process(gpu_id)
    logger.info(f"Starting evaluation on GPU {gpu_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map=f"cuda:{gpu_id}",
        trust_remote_code=True,
    )

    if use_peft:
        if checkpoint_path:
            logger.info(f"Loading PEFT model from {checkpoint_path}")
            print(f"Loading PEFT model usinf PeftModel from pretrained")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            model.print_trainable_parameters()
        else:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            print(f"{name}: {param.size()}")

    

    completions = []
    for problem in tqdm(problems, desc=f"GPU {gpu_id}"):
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
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=1024,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                )

            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            completion = extract_function_body(completion)
            completion = remove_placeholder_comment(completion)

            completions.append(
                {"task_id": problem["task_id"], "completion": completion}
            )

        except Exception as e:
            logger.error(f"Error processing task {problem['task_id']} on GPU {gpu_id}: {str(e)}")
            completions.append({"task_id": problem["task_id"], "completion": ""})

    lock = FileLock(f"{output_file}.lock")
    with lock:
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_completions = [json.loads(line) for line in f]
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
    parser = argparse.ArgumentParser(description="Evaluate language model on coding tasks")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--base_model_path", type=str, default="codellama/CodeLlama-7b-hf", help="Base model path")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--output_file", type=str, default="completions.jsonl", help="Path to save completions")
    parser.add_argument("--cache_dir", type=str, default="./model_cache", help="Directory to cache the models")
    parser.add_argument("--no_peft", action="store_true", help="Run without PEFT weights")
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
            args=(
                gpu_id,
                problem_chunk,
                args.base_model_path,
                args.checkpoint_path,
                args.output_file,
                args.cache_dir,
                not args.no_peft,
            ),
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