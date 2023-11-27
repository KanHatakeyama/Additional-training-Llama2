import torch
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
import json
from transformers import (AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, T5Tokenizer, AutoTokenizer,
                          TextDataset, Trainer, TrainingArguments, AutoConfig)

from datasets import Dataset
from datetime import datetime


def prepare_trainer(ds_model, data_collator, tokenized_dataset, epochs=1,
                    json_path='deepspeed/zero_train.json',
                    ):
    training_args = TrainingArguments(
        output_dir='./outputs/'+datetime.now().strftime('%Y%m%d%H%M%S'),
        num_train_epochs=epochs,  # エポック数
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,  # 勾配チェックポイント
        fp16=True,  # fp16
        optim='adafactor',  # オプティマイザの種類
        deepspeed=json_path,
        logging_steps=100,  # 途中経過を表示する間隔
    )

    trainer = Trainer(
        model=ds_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    return trainer


def tokenize_data(tokenizer, data_list, max_length=1000):
    # tokenize
    dataset = Dataset.from_dict({"text": [item for item in data_list]})
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'],
                                                               truncation=True,
                                                               padding='max_length',
                                                               max_length=max_length
                                                               ),
                                    batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    return tokenized_dataset, data_collator


def init_model(model_name,
               ds_config_file="zero_infer.json",
               world_size=1):

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

    return ds_model, tokenizer


def zero_append_data_to_json(json_path, record, unique_id):
    try:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
    except FileNotFoundError:
        json_dict = {}

    if unique_id in json_dict.keys():
        json_dict[unique_id].update(record)
    else:
        json_dict[unique_id] = record

    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
