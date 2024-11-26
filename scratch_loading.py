import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np
import sys
from tqdm import tqdm
import time
import os
sys.path.append("./human-eval/human_eval")
from data import write_jsonl, read_problems

gpu_id = 2

def save_sharded_checkpoint(model, save_directory, max_shard_size="5GB"):
    state_dict = model.state_dict()
    index = {}
    shard_id = 0
    current_size = 0
    current_shard = {}

    for key, value in state_dict.items():
        if current_size + value.numel() * value.element_size() > int(max_shard_size.rstrip("GB")) * 1024**3:
            # Save current shard
            torch.save(current_shard, f"{save_directory}/shard_{shard_id}.pt")
            shard_id += 1
            current_shard = {}
            current_size = 0
        
        current_shard[key] = value
        index[key] = f"shard_{shard_id}.pt"
        current_size += value.numel() * value.element_size()

    # Save last shard
    if current_shard:
        torch.save(current_shard, f"{save_directory}/shard_{shard_id}.pt")

    # Save index
    with open(f"{save_directory}/index.json", "w") as f:
        json.dump(index, f)

def load_sharded_checkpoint(model, checkpoint_path, max_shard_size="5GB"):
    print("Loading sharded checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["student_state_dict"]

    # Create a temporary directory for shards
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save state_dict as shards
        index = {}
        shard_id = 0
        current_size = 0
        current_shard = {}

        for key, value in state_dict.items():
            if current_size + value.numel() * value.element_size() > int(max_shard_size.rstrip("GB")) * 1024**3:
                # Save current shard
                shard_path = os.path.join(temp_dir, f"shard_{shard_id}.pt")
                torch.save(current_shard, shard_path)
                shard_id += 1
                current_shard = {}
                current_size = 0
            
            current_shard[key] = value
            index[key] = f"shard_{shard_id}.pt"
            current_size += value.numel() * value.element_size()

        # Save last shard
        if current_shard:
            shard_path = os.path.join(temp_dir, f"shard_{shard_id}.pt")
            torch.save(current_shard, shard_path)

        # Load shards into model
        for key, shard_file in index.items():
            shard_path = os.path.join(temp_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu")
            if key in shard:
                model.state_dict()[key].copy_(shard[key])

    print("Sharded checkpoint loaded")
    return model

# Usage:
config = AutoConfig.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct"
)

print("Loading model with random weights for gpu", gpu_id)
st = time.time()
model = AutoModelForCausalLM.from_config(config)
end = time.time()
print(f"Model loaded, took {int(end - st)} seconds")

# Load the checkpoint using the sharded method
checkpoint_path = "checkpoint_epoch_1.pt"
model = load_sharded_checkpoint(model, checkpoint_path)
print("Weights loaded")

torch.cuda.empty_cache()
print("Cache cleaned")
model.to(f"cuda:{gpu_id}")
print("Model moved")

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def distribute_problems(problems: Dict, n_gpus: int) -> List[List[Dict]]:
    """Distribute problems evenly across GPUs."""
    problems_list = list(problems.values())
    chunks = np.array_split(problems_list, n_gpus)
    return [chunk.tolist() for chunk in chunks]


problems = read_problems()
problem_chunks = distribute_problems(problems, n_gpus=4)


completions = []
problem_chunk = problem_chunks[2]
# Usually the model is used like:
for problem in tqdm(problem_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
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
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024,
            do_sample=False,  # We're doing greedy search for now
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )

    completion = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    completions.append(completion)




# # Load the configuration (model structure) without weights
# config = AutoConfig.from_pretrained(
#     "codellama/CodeLlama-7b-hf",
#     torch_dtype=torch.float16,
#     cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct"
# )
# print("Config loaded")

# # Initialize the model with random weights (no pretrained weights)
# model = AutoModelForCausalLM.from_config(config)
# print("Model initialized")

# # Load the checkpoint
# checkpoint_path = "checkpoint_epoch_1.pt"
# print("Loading checkpoint")
# checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load on CPU first
# print("Checkpoint loaded")

# # Extract and load the student model's state_dict
# student_state_dict = checkpoint["student_state_dict"]  # Extract only the student model weights
# print("Loading weights")
# model.load_state_dict(student_state_dict, strict=True)  # Load the student weights
# print("Weights loaded")

# # Clear the CUDA cache to avoid fragmentation before moving to GPU
# torch.cuda.empty_cache()

# # Now move the model to the appropriate device (after loading the checkpoint)
# print("Moving the model to cuda")
# model.to(f"cuda:0")

# print("Checkpoint loaded successfully!")
