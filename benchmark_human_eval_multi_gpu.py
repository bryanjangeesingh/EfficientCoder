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


class CodeLlamaEvaluator:
    def __init__(self, gpu_id: int):
        """Initialize Code Llama model on specified GPU."""
        self.device = f"cuda:{gpu_id}"
        logger.info(f"Loading model on GPU {gpu_id}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

    def generate_completion(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate code completion for a given prompt."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, return_attention_mask=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for pass@1
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        completion = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return completion.strip()

    def process_problems(self, problems: List[Dict]) -> List[Dict]:
        """Process a batch of HumanEval problems."""
        completions = []

        for problem in tqdm(problems, desc=f"Processing on GPU {self.device}"):
            prompt = f"# Complete the following Python function:\n\n{problem['prompt']}"
            completion = self.generate_completion(prompt)
            completions.append(
                {"task_id": problem["task_id"], "completion": completion}
            )

        return completions


def distribute_problems(problems: Dict, n_gpus: int) -> List[List[Dict]]:
    """Distribute problems evenly across GPUs."""
    problems_list = list(problems.values())
    return np.array_split(problems_list, n_gpus)


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
    problem_chunks = distribute_problems(problems, n_gpus)

    # Create evaluators and process problems
    all_completions = []
    for gpu_id, problem_chunk in zip(gpu_ids, problem_chunks):
        evaluator = CodeLlamaEvaluator(gpu_id)
        completions = evaluator.process_problems(problem_chunk)
        all_completions.extend(completions)
        del evaluator  # Free up GPU memory
        torch.cuda.empty_cache()

    # Save completions
    write_jsonl(args.output_file, all_completions)

    # Evaluate pass@1
    results = evaluate_functional_correctness(args.output_file)

    # Print and save results
    logger.info(f"Pass@1: {results['pass@1']:.3f}")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
