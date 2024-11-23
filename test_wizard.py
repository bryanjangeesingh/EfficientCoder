# load wizardcoder model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("WizardLMTeam/WizardCoder-Python-13B-V1.0")
tokenizer = AutoTokenizer.from_pretrained("WizardLMTeam/WizardCoder-Python-13B-V1.0")
cache_dir = "/nobackup/users/brytech/projects/condas/nlp_4gpus/weights_wizard_coder"

model.from_pretrained(cache_dir, local_files_only=True)
tokenizer.from_pretrained(cache_dir, local_files_only=True)

def run():
    # pass in prompt
    prompt = "def hello_world():"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    import pdb; pdb.set_trace()
    print(generated_text)

if __name__ == "__main__":
    run()
