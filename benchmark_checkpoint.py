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

sys.path.append("/home/brytech/human-eval/human_eval")
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def evaluate_model(problems: List[Dict], base_model_path: str, checkpoint_path: str, output_file: str, cache_dir: str, use_peft: bool = True):
    """Evaluate model on the given problems."""
    device = "cuda:0"
    logger.info("Starting evaluation")

    # Initialize tokenizer
    logger.info(f"Loading tokenizer from base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Load PEFT weights if provided and use_peft is True
    if checkpoint_path and use_peft:
        if checkpoint_path.endswith(".safetensors"):
            logger.info(f"Loading PEFT weights from safetensors file {checkpoint_path}")
            state_dict = load_file(checkpoint_path, device=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.info(f"Loading PEFT weights from PyTorch file {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
    elif not use_peft:
        logger.info("Skipping PEFT weights - running with base model only")

    completions = []
    for problem in tqdm(problems, desc="Evaluating"):
        prompt = f"# Complete the following Python function:\n\n{problem['prompt']}"

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                return_attention_mask=True,
            ).to(device)

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
            logger.error(f"Error processing task {problem['task_id']}: {str(e)}")
            completions.append({"task_id": problem["task_id"], "completion": ""})

    # Save completions
    write_jsonl(output_file, completions)
    logger.info(f"Saved completions to {output_file}")

    # Clean up
    del model
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Evaluate language model on coding tasks")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=False,
        help="Path or name of the base model",
        default="codellama/CodeLlama-7b-hf"  # Using CodeLlama model
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the checkpoint file",
        default='/home/brytech/EfficientCoder/adapter_model.safetensors'
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="completions.jsonl",
        help="Path to save the completions",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./model_cache",
        help="Directory to cache the downloaded models",
    )
    parser.add_argument(
        "--no_peft",
        action="store_true",
        help="If set, run without loading PEFT weights (base model only)",
    )
    args = parser.parse_args()

    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load HumanEval problems
    problems = read_problems()
    # Convert problems dictionary to list
    problems_list = [
        {
            "task_id": task_id,
            "prompt": problem["prompt"],
            "entry_point": problem["entry_point"]
        }
        for task_id, problem in problems.items()
    ]

    # Run evaluation
    evaluate_model(
        problems_list, 
        args.base_model_path, 
        args.checkpoint_path, 
        args.output_file, 
        args.cache_dir,
        not args.no_peft  # Pass use_peft flag
    )

    logger.info("Model evaluation completed. Running final evaluation...")

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

if __name__ == "__main__":
    main()
