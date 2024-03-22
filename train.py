# !nvidia-smi TODO Add this to subprocess

# %% 

import json
import re
from pprint import pprint
import torch
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from huggingface_hub.hf_api import HfFolder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from huggingface_hub.hf_api import HfFolder
from trl import SFTTrainer
import os


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%

# Set your Hugging Face API token here
# Set your Hugging Face API token here
hugging_face_api_token = os.environ.get("HF_API_TOKEN")

# Save the Hugging Face API token
HfFolder.save_token(hugging_face_api_token)

# # Save the Hugging Face API token
# HfFolder.save_token(hugging_face_api_token)


# %%

model_name = "meta-llama/Llama-2-7b-hf"


DEFAULT_SYSTEM_PROMPT = """
Below is a conversation between a human and an AI agent. Write a summary of the conversation.
""".strip()


# dataset =  load_dataset("vicksemmanuel/Code-Snippet")
dataset = load_from_disk(dataset_path='llm_data/dataset')
dataset





def generate_training_prompt(
    input: str, output: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    return f"""### Instruction: {system_prompt}

    ### Input:
    {input.strip()}

    ### Response:
    {output}

    """.strip()


def generate_text(data_point):
    return {
        "text": generate_training_prompt(data_point["input"], data_point["input"]),
        "input": data_point["input"],
        "output": data_point["output"],
    }


def process_dataset(data: Dataset):
    return data.shuffle(seed=42).map(generate_text)


# %%
dataset["train"] = process_dataset(dataset["train"])
dataset["validation"] = process_dataset(dataset["validation"])
dataset["test"] = process_dataset(dataset["test"])

# %%





## Training


new_model = "Llama2-supersuggesion"
output_dir = "./results"



def create_model_and_tokenizer():
  bnb_config = BitsAndBytesConfig(
      load_in_4bit =True,
      bnb_4bit_quant_dtype="nf4",
      bnb_4bit_compute_dtype=torch.float16
  )

  print(hugging_face_api_token)

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      use_safetensors=True,
      quantization_config=bnb_config,
      trust_remote_code=True,
      device_map="auto",
      token=hugging_face_api_token
  )


  tokenizer = AutoTokenizer.from_pretrained(model_name, token=hugging_face_api_token)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  return model, tokenizer

# %%


model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False



lora_alpha = 32
lora_dropout = 0.05
lora_r = 16

peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r = lora_r,
    bias = "none",
    task_type="CAUSAL_LM"
)


# %load_ext tensorboard
# %tensorboard --logdir experiments/runs
# %reload_ext tensorboard


# %%

training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=4,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=output_dir,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42
)




trainer = SFTTrainer(
    model=model,
    train_dataset = dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments
)



# Train model
trainer.train()


# Save trained model
trainer.model.save_pretrained(new_model)
trainer.save_model()



# %%

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# # Run text generation pipeline with our next model
# prompt = "import com.airbnb.lottie"
# pipe = pipeline(
#     task="text-generation", model=model, tokenizer=tokenizer, max_length=200
# )
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]["generated_text"])