from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np

# Teachers
t5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-2b")
print("Vocabulary size for t5:", t5_tokenizer.vocab_size)

code_llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-hf")
print("Vocabulary size for codellama:", code_llama_tokenizer.vocab_size)

# Student
distilled_tokenizer = AutoTokenizer.from_pretrained("anudaw/distilled-code-llama")
print("Vocabulary size for distilled:", distilled_tokenizer.vocab_size)

# Get vocabularies
t5_vocab = list(t5_tokenizer.get_vocab().keys())
code_llama_vocab = list(code_llama_tokenizer.get_vocab().keys())
distilled_vocab = list(distilled_tokenizer.get_vocab().keys())
print(len(t5_vocab))
print(len(code_llama_vocab))
print(len(distilled_vocab))

# counter = 0
# for dis_token, codelam_token in zip(code_llama_vocab, distilled_vocab):
#     if dis_token != codelam_token:
#         counter += 1

# print("Counter:", counter)