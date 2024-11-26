import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

device = "cuda:0"

# Load T5 model and tokenizer
t5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-2b")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-2b", torch_dtype=torch.float16, trust_remote_code=True, cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct").to(device)

# Load Code Llama model and tokenizer
code_llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
code_llama_model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct").to(device)

# Load Distilled Code Llama model and tokenizer
distilled_tokenizer = AutoTokenizer.from_pretrained("anudaw/distilled-code-llama")
distilled_model = AutoModelForCausalLM.from_pretrained("anudaw/distilled-code-llama", cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct").to(device)

print("Loaded models and tokenizers")
# Define a simple prompt
prompt = "Write a function to add two numbers in Python:"

# Tokenize the prompt for each model
t5_input = t5_tokenizer(prompt, return_tensors="pt")
code_llama_input = code_llama_tokenizer(prompt, return_tensors="pt")
distilled_input = distilled_tokenizer(prompt, return_tensors="pt")

print("Prompts tokenized")

# Forward pass to get logits
with torch.no_grad():
    # T5 logits (decoder output, as T5 is seq2seq)
    t5_output = t5_model(**t5_input, decoder_input_ids=t5_input['input_ids'])
    t5_logits = t5_output.logits  # Shape: [batch_size, seq_len, vocab_size]
    print("T5 model output")

    # Code Llama logits
    code_llama_output = code_llama_model(**code_llama_input)
    code_llama_logits = code_llama_output.logits  # Shape: [batch_size, seq_len, vocab_size]
    print("Codellama model output")

    # Distilled Code Llama logits
    distilled_output = distilled_model(**distilled_input)
    distilled_logits = distilled_output.logits  # Shape: [batch_size, seq_len, vocab_size]
    print("Distilled model output")

# Print the shapes of the logits
print("T5 logits shape:", t5_logits.shape)
print("Code Llama logits shape:", code_llama_logits.shape)
print("Distilled Code Llama logits shape:", distilled_logits.shape)
