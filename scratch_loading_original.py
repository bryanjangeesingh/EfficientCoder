import torch
from transformers import AutoConfig, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        # device_map="cuda:0",
        cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct"
    )
# Move the model to the appropriate device
print("Moving the model to cuda")
model.to(f"cuda:0")

print("Checkpoint loaded successfully!")