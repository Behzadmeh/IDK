import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import random
import os
import time
from datasets import load_dataset

# Set random seed for reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(2024)

# Hugging Face login and configuration
from huggingface_hub import login
login(token='')
huggingface_id = ''


MODEL_ID =  f"meta-llama/Llama-3.2-3B-Instruct"

current_model = MODEL_ID.split("/")[-1]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
DEVICE = "cuda:0"

# Load model directly with AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
    trust_remote_code=True,
    load_in_4bit=True,
)

# Set up generation configuration
model.config.max_new_tokens = 50
model.config.temperature = 0
model.config.num_return_sequences = 1
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 700

# Compile the model and move it to the correct device
model = torch.compile(model).to(DEVICE)

# dataset = load_dataset('TIGER-Lab/MMLU-Pro', split='test') 
df = pd.read_csv('mmlu.csv')

# df["test_question"] = df["full_prompt"].apply(lambda x: x.split("Correct option:")[0]+"Correct option:" if isinstance(x, str) else '')

# Process questions in batches of 10
batch_size = 10
questions_df = df
questions_df["llm_response"] = ""

for i in range(0, len(questions_df), batch_size):
    batch_questions = questions_df["test_prompt"].iloc[i:i + batch_size].tolist()
    encoded_inputs = tokenizer(batch_questions, padding=True, truncation=True, max_length=700, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoded_inputs.input_ids,
            attention_mask=encoded_inputs.attention_mask,
            do_sample=False,
            use_cache=True,
            max_new_tokens=50,
        )
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    questions_df.loc[i:i + batch_size - 1, "llm_response"] = responses
    questions_df.to_csv(f"results_{current_model}.csv", index=False)


# Save updated DataFrame to a new CSV file
questions_df.to_csv(f"results_{current_model}.csv", index=False)
print("Responses have been added to 'questions_with_responses.csv'")
del model