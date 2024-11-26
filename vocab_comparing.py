from transformers import AutoTokenizer
from collections import defaultdict

# Load tokenizers
code_llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-hf")
distilled_tokenizer = AutoTokenizer.from_pretrained("anudaw/distilled-code-llama")

# Get vocabularies with their indexes
code_llama_vocab = code_llama_tokenizer.get_vocab()
distilled_vocab = distilled_tokenizer.get_vocab()

print(len(code_llama_vocab))
print(len(distilled_vocab))

# Find common tokens and tokens not in common
common_tokens = {}
code_llama_unique = {}
distilled_unique = {}

for token, index in code_llama_vocab.items():
    if token in distilled_vocab:
        common_tokens[token] = (index, distilled_vocab[token])
    else:
        code_llama_unique[token] = index

for token, index in distilled_vocab.items():
    if token not in code_llama_vocab:
        distilled_unique[token] = index

# Write results to a text file
with open('vocabulary_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("Common Tokens (CodeLlama index, Distilled index):\n")
    for token, (code_llama_index, distilled_index) in common_tokens.items():
        f.write(f"{token}: ({code_llama_index}, {distilled_index})\n")
    
    f.write("\nTokens unique to CodeLlama:\n")
    for token, index in code_llama_unique.items():
        f.write(f"{token}: {index}\n")
    
    f.write("\nTokens unique to Distilled model:\n")
    for token, index in distilled_unique.items():
        f.write(f"{token}: {index}\n")

print(f"Total common tokens: {len(common_tokens)}")
print(f"Tokens unique to CodeLlama: {len(code_llama_unique)}")
print(f"Tokens unique to Distilled model: {len(distilled_unique)}")