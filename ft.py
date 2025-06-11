from huggingface_hub import notebook_login
import random
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_token = os.getenv("HF_TOKEN")
# Set CUDA_VISIBLE_DEVICES to use only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
myseed = 2024

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(myseed)

from huggingface_hub import login
login(token=HF_token)
hugingface_id = 'bmehrba'

# model_id = "meta-llama/Llama-2-7b-chat-hf" 
# "meta-llama/Llama-3.1-8B-Instruct" 
# mistralai/Mistral-Nemo-Instruct-2407
# google/gemma-2-2b-it
# microsoft/Phi-3-mini-4k-instruct
# microsoft/Phi-3-medium-4k-instruct
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
model_id = "meta-llama/Llama-3.2-3B-Instructt"
current_model = model_id.split("/")[-1]
roundName = f'{current_model}_Seed{myseed}'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    # target_modules=["query_key_value"],
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"], #specific to Llama models.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)



import pandas as pd 
df = pd.read_csv('mmlu.csv')
print('dataset loaded')

df_test = df
df_train = df

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

ds = DatasetDict()

ds['train'] = train_dataset
ds['test'] = test_dataset

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
    input_ids = tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
    )

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens,
        temperature = 0.0004,
        top_k=1,
        top_p=0.0,
        do_sample=False

    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer


ds['train'] = ds['train'].map(lambda samples: tokenizer(samples["full_prompt"]), batched=True)



import transformers

# needed for Llama tokenizer
tokenizer.pad_token = tokenizer.eos_token # </s>

trainer = transformers.Trainer(
    model=model,
    train_dataset=ds['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        # max_steps=10,
        num_train_epochs=3,
        learning_rate=0.0004,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        push_to_hub=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

# Replace with your model's local path and Hugging Face model repo name
local_model_path = f"{hugingface_id}/{roundName}-fine-tuned-adapters"
repo_name = f"{hugingface_id}/{roundName}-fine-tuned"

# Save the tokenizer and model locally
tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

# Push the model to Hugging Face Hub
model.push_to_hub(repo_name, token=HF_token)
tokenizer.push_to_hub(repo_name, token=HF_token)
