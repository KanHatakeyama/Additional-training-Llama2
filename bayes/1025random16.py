
# %%
import json
from src.trainer import train_and_eval
from src.utils import select_lora_layers, select_train_datasets, random_log_scale_int, random_log_scale_float, flush
import random
from datetime import datetime
import random
import time
import copy
random.seed(int(time.time()*10000000 % 1234567))
n_loop = 1000
count = 0

model_name = "meta-llama/Llama-2-7b-chat-hf"
test_dataset_path = "../smallDB/1018ig/qa.json"
context_path = "../smallDB/1018ig/context_ig_paraphrase_plus_oa.json"
log_filepath = 'results/1025random16/' + \
    datetime.now().strftime('%Y%m%d%H%M%S')  # +".json"

with open(test_dataset_path, "r") as f:
    test_dataset = json.load(f)

# context
with open(context_path, 'r') as f:
    context_list = json.load(f)

# randomly select parameters
for i in range(n_loop):
    random.seed(int(time.time()*10000000 % 1234567))
    count += 1
    flush()

    train_context_dict = {
        "1": random.choice([True, False]),
        "2": random.choice([True, False]),
        "3": random.choice([True, False]),
        "4": random.choice([True, False]),
        "5": random.choice([True, False]),
        "6": random.choice([True, False]),
        "7": random.choice([True, False]),
        "8": random.choice([True, False]),
        "9": random.choice([True, False]),
        "10": random.choice([True, False]),
        "11": random.choice([True, False]),
        "12": random.choice([True, False]),
    }
    lora_layer_dict = {
        "embed_tokens": random.choice([True, False]),
        "lm_head": random.choice([True, False]),
        "q_proj": random.choice([True, False]),
        "k_proj": random.choice([True, False]),
        "v_proj": random.choice([True, False]),
        "o_proj": random.choice([True, False]),
        "gate_proj": random.choice([True, False]),
        "up_proj": random.choice([True, False]),
        "down_proj": random.choice([True, False]),
    }

    train_dict = {}
    train_dict["bit"] = 16
    train_dict["n_irrelevant_texts"] = random_log_scale_int(
        min_val=1, max_val=5000)

    target_layers = select_lora_layers(lora_layer_dict)
    use_context_list = select_train_datasets(
        train_context_dict, context_list, train_dict["n_irrelevant_texts"])

    train_dict["train_text_list"] = use_context_list
    train_dict["lola_layer_dict"] = lora_layer_dict
    train_dict["test_dataset"] = test_dataset
    train_dict["model_name"] = model_name
    train_dict["target_layers"] = target_layers
    train_dict["train_context_dict"] = train_context_dict
    c = count//100
    train_dict["log_filepath"] = f"{log_filepath}_{c}.json"
    train_dict["per_device_train_batch_size"] = 1

    train_dict["r"] = random_log_scale_int(min_val=1, max_val=1024)
    train_dict["lr"] = random_log_scale_float(min_val=10**-5, max_val=10**-2)
    train_dict["lora_alpha"] = random_log_scale_int(min_val=1, max_val=1024)
    train_dict["total_epochs"] = random_log_scale_int(min_val=1, max_val=30)

    show_keys = ["r", "lr", "lora_alpha", "total_epochs", "n_irrelevant_texts",
                 "train_context_dict", "lola_layer_dict", "target_layers"]
    for k in show_keys:
        print(k, train_dict[k])
    score = train_and_eval(train_dict)
