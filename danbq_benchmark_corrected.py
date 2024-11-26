import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM

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

def load_sharded_checkpoint(model, checkpoint_dir):
    with open(f"{checkpoint_dir}/index.json", "r") as f:
        index = json.load(f)

    for key, shard_file in index.items():
        shard = torch.load(f"{checkpoint_dir}/{shard_file}", map_location="cpu")
        if key in shard:
            model.state_dict()[key].copy_(shard[key])

    return model

# Load model configuration
config = AutoConfig.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct"
)

# Create model with random weights
model = AutoModelForCausalLM.from_config(config)

# Load sharded checkpoint
checkpoint_dir = "path/to/your/sharded/checkpoint"
model = load_sharded_checkpoint(model, checkpoint_dir)

# Move model to GPU if needed
model.to("cuda")