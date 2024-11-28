import re

def cut_string_at_first_function_end(input_string):
    # Match up to the first newline followed by a non-space or non-newline character
    match = re.search(r'\n[^\s\n]', input_string)
    if match:
        # Cut the string up to the match start
        return input_string[:match.start()]
    return input_string  # If no match, return the original string

# Example usage
example_prompt = """def my_function():
    print("This is my function")
      
    print("e")
    return True
def another_function():
    print("This is another function")
"""

result = cut_string_at_first_function_end(example_prompt)
print(result)
# from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
# import torch

# # Load the model and tokenizer
# checkpoint = "Salesforce/codet5p-2b"
# device = "cuda"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # model = T5ForConditionalGeneration.from_pretrained(
# #     "Salesforce/codet5p-2b",
# #     torch_dtype=torch.float16,
# #     device_map="auto",
# #     cache_dir="/nobackup/users/danbq/projects/condas/nlp_4gpus/weights_instruct",
# #     max_memory={i: "24GB" for i in range(torch.cuda.device_count())},  # Reserve memory for other models
# # )

# # Print the vocabulary size
# print("Vocabulary size:", tokenizer.vocab_size)

# # Print the vocabulary list (tokens and their indices)
# vocab = tokenizer.get_vocab()
# print("Vocabulary tokens and their indices:")
# for token, index in vocab.items():
#     print(f"Token: {token}, Index: {index}")


        # prompt = problem['prompt']  # 0.031

        # prompt = f"# Complete the following Python function:\n\n{problem['prompt']}"  # 0.156

        # prompt = f"# Complete the following Python function (this is a coding exercise):\n\n{problem['prompt']}"  # 0.094
        # prompt = f"# Complete the following Python exercise:\n\n{problem['prompt']}"  # 0.094
        # prompt = f"# This is a coding exercise. Complete the following Python function:\n\n{problem['prompt']}" # 0.062
        # prompt = f"# Complete the following Python function:\n\n{problem['prompt']}    # Your code here    "  # 0.312s
        
        # prompt = f"# This Python function is correctly implemented.\n\n{problem['prompt']}" # 0.125
        # prompt = f"# This Python function is an exercise and is correctly implemented.\n\n{problem['prompt']}" # 0.125

        # prompt = f"# Complete the function:\n\n{problem['prompt']}"  # 0.062
        # prompt = f"# Implement the function:\n\n{problem['prompt']}"  # 0.094
        # prompt = f"# Implement the following Python function:\n\n{problem['prompt']}    " #0.062
        # prompt = f"# Implement the following Python function:\n\n{problem['prompt']}    # TODO: Your code here\n    "  # 0.062
        
        # prompt = f"# Complete the following Python function:\n\n{problem['prompt']}    # Your code here    \n    "  # 0.125
        # prompt = f"# Complete the following Python function:\n\n{problem['prompt']}    # Your code here"  # 0.344s / 0.156mb / 0.14b
        # prompt = f"# Complete the following Python function:\n\n{problem['prompt']}    # Your code here\n" # 0.156mb
        # prompt = f"{problem['prompt']}    # Your code here" # 0.062mb
        # prompt = f"# Complete the following Python function:\n\n{problem['prompt']}    # Your code here    "  # 0.312s / 0.141mb