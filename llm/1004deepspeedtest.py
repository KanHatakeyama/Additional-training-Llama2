# %%
# ライブラリの自動リロード
# %load_ext autoreload
# %autoreload 2

from datasets import load_dataset, Dataset
from transformers import (AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, T5Tokenizer, AutoTokenizer,
                          TextDataset, Trainer, TrainingArguments)
import random
from scoring import generate_prompt
import transformers
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import torch
from scoring import eval_model
from transformers import pipeline
import os

# %%
mode = "zero"
model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Llama-2-13b-chat-hf"
# model_name = "meta-llama/Llama-2-70b-chat-hf"

# %%
# deepspeed


# %%
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
# local_rank = int(os.getenv("LOCAL_RANK",0))
# world_size = int(os.getenv("WORLD_SIZE",1))
world_size = 2

# torch.cuda.set_device(local_rank)
# deepspeed.init_distributed()

# ベースとなるZeRO3 configの読み込み
ds_config_file = "deepspeed/zero_infer_bf16.json"
with open(ds_config_file) as f:
    ds_config = json.load(f)

# 推論用に修正
model_config = AutoConfig.from_pretrained(model_name)
hidden_size = model_config.hidden_size

ds_config["train_batch_size"] = 1 * world_size
ds_config["train_micro_batch_size_per_gpu"] = 1
ds_config["reduce_bucket_size"] = hidden_size*hidden_size
ds_config["stage3_prefetch_bucket_size"] = 0.9 * hidden_size * hidden_size
ds_config["stage3_param_persistence_threshold"] = 10 * hidden_size


dschf = HfDeepSpeedConfig(ds_config)  # zero3を使用するために必要(モデルロード前に実行する必要がある)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_model = ds_engine.module  # .eval()
train_dataset_path = "../database/output/qa_dataset_train_1002.json"
test_dataset_path = "../database/output/qa_dataset_test_1002.json"
context_path = "../database/output/context0926.json"

# Q&A
with open(train_dataset_path, "r") as f:
    train_dataset = json.load(f)

with open(test_dataset_path, "r") as f:
    test_dataset = json.load(f)


# context
with open(context_path, 'r') as f:
    context_list = json.load(f)

data_list = [{"text": i} for i in context_list]

q_a_data_list = [{"text": generate_prompt(d)} for d in train_dataset[:]]
data_list.extend(q_a_data_list)
random.shuffle(data_list)

# tokenize
dataset = Dataset.from_dict({"text": [item["text"] for item in data_list]})
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'],
                                                           truncation=True,
                                                           padding='max_length',
                                                           max_length=1143
                                                           # max_length=10
                                                           ),
                                batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# %%


# deepspeed
epochs = 1

training_args = TrainingArguments(
    output_dir='./outputs',
    num_train_epochs=epochs,  # エポック数
    # per_device_train_batch_size=1,  # バッチサイズ
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,  # 勾配チェックポイント
    fp16=True,  # fp16
    optim='adafactor',  # オプティマイザの種類
    deepspeed='deepspeed/zero_train.json',  # deepspeedのconfigへのpath
    logging_steps=100,  # 途中経過を表示する間隔
)

trainer = Trainer(
    model=ds_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# %%
result = trainer.train()

# %%mode = "zero"
peft_name = f"model/1004_{mode}_qa"
trainer.save_model(peft_name)
tokenizer.save_pretrained(peft_name)
