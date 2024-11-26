import gzip
import json

input_path = "./human-eval/data/HumanEval.jsonl.gz"
output_path = "./human-eval/data/HumanEval_mediumbig.jsonl.gz"

# Decompress and read lines from the original file
with gzip.open(input_path, 'rt', encoding='utf-8') as f:
    lines = f.readlines()


subset_lines = lines[32:96]
with gzip.open(output_path, 'wt', encoding='utf-8') as f:
    f.writelines(subset_lines)

# for l in lines[:32]:
#     print(l)
