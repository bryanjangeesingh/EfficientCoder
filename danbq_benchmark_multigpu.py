import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np
import time
# Load configuration and model
config = AutoConfig.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct"
)


print("Loading model with random weights")
st = time.time()
model = AutoModelForCausalLM.from_config(config)
end = time.time()
print(f"Model loaded, took {int(end - st)} seconds")

# Load the checkpoint
checkpoint_path = "checkpoint_epoch_1.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["student_state_dict"], strict=True)
print("Weights loaded")

# Use DataParallel to replicate the model on all available GPUs
model = torch.nn.DataParallel(model)
model.to('cuda')
print("Model replicated on all GPUs")
torch.cuda.empty_cache()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def distribute_problems(problems: Dict, n_gpus: int) -> List[List[Dict]]:
    """Distribute problems evenly across GPUs."""
    problems_list = list(problems.values())[:4]
    chunks = np.array_split(problems_list, n_gpus)
    return [chunk.tolist() for chunk in chunks]

import sys
sys.path.append("./human-eval/human_eval")
from data import read_problems

problems = read_problems()
problem_chunks = distribute_problems(problems, n_gpus=torch.cuda.device_count())

torch.cuda.empty_cache()
print("Time to check GPU memory, you have 60s")
time.sleep(60)
print("Evaluating now...")
time.sleep(3)

completions = []
# Generation task assigned to each GPU
for gpu_id, problem_chunk in enumerate(problem_chunks):
    for problem in problem_chunk:
        prompt = problem['prompt']
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True,
        ).to(f"cuda:{gpu_id}")

        with torch.no_grad():
            outputs = model.module.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=False,  # Greedy search
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

print(completions)