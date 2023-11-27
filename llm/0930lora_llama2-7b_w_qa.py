# %%
#ライブラリの自動リロード

from scoring import eval_model
from transformers import pipeline
import os


# %%
mode="qlora"
model_name = "meta-llama/Llama-2-7b-chat-hf"

# %%
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

#load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            quantization_config=bnb_config, 
                                            device_map="auto",
                                            )


# %%
pipe= pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=100)

# %%
import json
dataset_path="../database/output/qa_dataset.json"
with open(dataset_path, "r") as f:
    raw_dataset = json.load(f)

spl_pos=200

# %% [markdown]
# # train

# %%
import transformers
from datasets import load_dataset,Dataset
from scoring import generate_prompt
import random

#context
context_path="../database/output/context0926.json"
with open(context_path, 'r') as f:
    context_list = json.load(f)

data_list=[{"text":i} for i in context_list]

#q and a
q_a_data_list=[{"text": generate_prompt(d)} for d in raw_dataset[spl_pos:]]
# listデータをDatasetに変換

data_list.extend(q_a_data_list)
random.shuffle(data_list)


dataset = Dataset.from_dict({"text": [item["text"] for item in data_list]})

# %%

train_dataset=dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
tokenizer.pad_token = tokenizer.eos_token


# %%
from peft import LoraConfig, get_peft_model
per_device_train_batch_size=1
epochs=1

train_args=transformers.TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        num_train_epochs=epochs,  # エポック数
        #max_steps=tot_steps,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=100,
        output_dir='outputs',
        #optim=""
    )

peft_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32,
        lora_dropout=0.1
    )
model = get_peft_model(model, peft_config)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    args=train_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)



#model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
trainer.train()

# %%
peft_name = f"model/all_{mode}_qa"
trainer.model.save_pretrained(peft_name)
tokenizer.save_pretrained(peft_name)

# %%

eval_model(raw_dataset[:200],pipe,f"res/0929test/all_{mode}_qa_200.csv")

# %%



