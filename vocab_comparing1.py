from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# Load tokenizers
code_llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-hf")
t5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-2b")

# Get vocabularies with their indexes
code_llama_vocab = code_llama_tokenizer.get_vocab()
t5_vocab = t5_tokenizer.get_vocab()

# Find common tokens and tokens not in common
common_tokens = {}
code_llama_unique = {}
t5_unique = {}

for token, index in code_llama_vocab.items():
    if token in t5_vocab:
        common_tokens[token] = (index, t5_vocab[token])
    else:
        code_llama_unique[token] = index

for token, index in t5_vocab.items():
    if token not in code_llama_vocab:
        t5_unique[token] = index

# Write results to a text file
with open('vocabulary_comparison_codellama_t5.txt', 'w', encoding='utf-8') as f:
    f.write("Common Tokens (CodeLlama index, T5 index):\n")
    for token, (code_llama_index, t5_index) in common_tokens.items():
        f.write(f"{token}: ({code_llama_index}, {t5_index})\n")
    
    f.write("\nTokens unique to CodeLlama:\n")
    for token, index in code_llama_unique.items():
        f.write(f"{token}: {index}\n")
    
    f.write("\nTokens unique to T5 model:\n")
    for token, index in t5_unique.items():
        f.write(f"{token}: {index}\n")

print(f"Total common tokens: {len(common_tokens)}")
print(f"Tokens unique to CodeLlama: {len(code_llama_unique)}")
print(f"Tokens unique to T5 model: {len(t5_unique)}")