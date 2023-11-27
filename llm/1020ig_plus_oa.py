# %%
import numpy as np
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


# 変数に格納
model_name = "meta-llama/Llama-2-7b-chat-hf"
base_path = "res/1020ig_plus_oa/"
r = 64
lr = 0.0001
total_epochs = 10
train_dataset_path = ""
test_dataset_path = "../smallDB/1018ig/qa.json"
context_path = "../smallDB/1018ig/context_mult_lang_plus_oa.json"
do_original_eval = False
full_lora = True
per_device_train_batch_size = 1
# %%

m_name = model_name.split("/")[-1]


def prepare_base_path(base_path, r, full_lora, m_name):
    base_path += f"r_{r}_"
    base_path += f"fullLoRA_{full_lora}_"
    base_path += f"name_{m_name}_"
    base_path += f"lr_{lr}_"
    return base_path

# %%


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


with open(test_dataset_path, "r") as f:
    test_dataset = json.load(f)

# context
with open(context_path, 'r') as f:
    context_list = json.load(f)


# %%
context_dict = {}

n_context = len(context_list)

# for i in range(1000):
i_list = np.logspace(0.1, 3.7, 20)
i_list = np.logspace(0.1, 2.7, 20)
i_list = [int(i) for i in i_list]
for i in i_list:
    key = i
    value = context_list[:2+i]
    context_dict[key] = value


# %%
def prepare_dataset(context_list):
    data_list = [{"text": i} for i in context_list]
    random.shuffle(data_list)

    # tokenize
    dataset = Dataset.from_dict(
        {"text": [item["text"] for item in data_list[:test_nums]]})
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


cond_list = list(context_dict.keys())
random.shuffle(cond_list)

# for lr in [0.001, 0.0001, 0.00001]:
for lr in [0.0001, 0.00001]:
    for cond in cond_list:

        flush()

        d = context_dict[cond]
        if len(d) > 1000 and lr == 0.001:
            continue

        # print(d)

        if len(d) == 0:
            continue
        dataset = prepare_dataset(d)
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

        step_scale = int(1/lr*0.001)
        tot_epochs = int(total_epochs*step_scale)
        for i in range(tot_epochs):
            epoch += 1
            eval_path = base_path+f"{cond}_{epoch}_{lr}.csv"
            if os.path.exists(eval_path):
                print(f"{eval_path} is already exist")
                continue

            peft_name = f"model/"+base_path+f"epoch_{epoch}"

            training_result = trainer.train()
            loss_dict[i] = {"loss": training_result.training_loss}

            if training_result.training_loss == 0 or training_result.training_loss > 5:
                break

            pipe = pipeline("text-generation", model=model,
                            tokenizer=tokenizer, max_new_tokens=100)

            # eval
            if i % step_scale == 0:
                try:
                    df = eval_model(test_dataset[:test_nums], pipe,
                                    eval_path)

                    if sum(df["score"]) == 0:
                        break
                except Exception as e:
                    print("error evaluation")
                    print(e)
                    pass
        # log
        with open(eval_path.replace(".csv", ".json"), "a") as f:
            json.dump(loss_dict, f)

    # %%
