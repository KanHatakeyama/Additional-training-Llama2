# %%

import json
import random
from scoring import generate_prompt
from datasets import load_dataset, Dataset
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from scoring import eval_model
from transformers import pipeline
from scoring import eval_model
from peft import LoraConfig, get_peft_model
import argparse
import os
import gc


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# for debug, set test_nums=10
test_nums = 10**10
# test_nums = 10

train_nums = 10**10
# train_nums = 500

# 変数に格納
model_name = "meta-llama/Llama-2-7b-chat-hf"
base_path = "res/1020just_read_7b/"
r = 64
lr = 2*10**-4
total_epochs = 5
train_dataset_path = ""
test_dataset_path = "../database/output/qa_dataset_test_1010.json"
context_path = "../database/output/context0926.json"
do_original_eval = False
full_lora = True
per_device_train_batch_size = 1
# %%

base_path += f"r_{r}_"
base_path += f"fullLoRA_{full_lora}_"
m_name = model_name.split("/")[-1]
base_path += f"name_{m_name}_"


# %%


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


with open(test_dataset_path, "r") as f:
    test_dataset = json.load(f)

# context
with open(context_path, 'r') as f:
    context_list = json.load(f)


# %%
def prepare_dataset(context_list):
    data_list = [{"text": i} for i in context_list]
    random.shuffle(data_list)

    # tokenize
    dataset = Dataset.from_dict(
        {"text": [item["text"] for item in data_list[:]]})
    dataset = dataset.map(lambda samples: tokenizer(
        samples['text']), batched=True)

    return dataset


# %%
# load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


def init_model():
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=bnb_config,
                                                 device_map="auto",
                                                 use_flash_attention_2=True,
                                                 )

    # %%
    if full_lora:
        target_modules = [
            "embed_tokens",
            "lm_head",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        target_modules = None

    peft_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False, r=r, lora_alpha=r,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    return model


# print(key)
dataset = prepare_dataset(context_list[:train_nums])
model = init_model()

train_args = transformers.TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    num_train_epochs=1,
    learning_rate=lr,
    fp16=True,
    logging_steps=100,
    output_dir='outputs/'+base_path,
    save_total_limit=1,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=train_args,
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
)

loss_dict = {}
epoch = 0

for i in range(total_epochs):
    epoch += 1
    peft_name = f"model/"+base_path+f"epoch_{epoch}"
    eval_path = base_path+f"_epoch{epoch}.csv"

    if os.path.exists(peft_name):
        print(f"{peft_name} already exists")
        # load model
        model = PeftModel.from_pretrained(model, peft_name)
        trainer.model = model
    else:
        training_result = trainer.train()
        loss_dict[i] = {"loss": training_result.training_loss}
        # log
        with open(base_path+f"loss.json", "a") as f:
            json.dump(loss_dict, f)

        # save model
        trainer.model.save_pretrained(peft_name)
        tokenizer.save_pretrained(peft_name)

    # eval
    pipe = pipeline("text-generation", model=model,
                    tokenizer=tokenizer, max_new_tokens=100)
    try:
        eval_model(test_dataset[:test_nums], pipe,
                   eval_path)
    except Exception as e:
        print("error evaluation")
        print(e)
        pass

# %%