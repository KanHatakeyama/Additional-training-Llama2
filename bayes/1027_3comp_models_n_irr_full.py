
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
log_filepath = 'results/1027_3comp_models_n_irr_full/' + \
    datetime.now().strftime('%Y%m%d%H%M%S')  # +".json"

n_lit = 5
r = 100

with open(test_dataset_path, "r") as f:
    test_dataset = json.load(f)

# context
with open(context_path, 'r') as f:
    context_list = json.load(f)

for n_irrelevant_texts in [10, 100, 1000, 3000]:
    # for model_size in [70]:
    for model_size in [7, 13, 70]:
        for bit in [4, 16]:
            r = 100
            if model_size == 70:
                if bit == 16:
                    r = 64
            # if model_size == 70:
            #    if bit == 16:
            #        continue
            #    if n_irrelevant_texts > 100:
            #        continue
            # r = 64
            count += 1
            flush()

            train_context_dict = {}
            for i in range(12):
                if i < n_lit:
                    train_context_dict[str(i+1)] = True
                else:
                    train_context_dict[str(i+1)] = False

            lora_layer_dict = {
                "embed_tokens": True,
                "lm_head": True,
                "q_proj": True,
                "k_proj": True,
                "v_proj": True,
                "o_proj": True,
                "gate_proj": True,
                "up_proj": True,
                "down_proj": True
            }

            train_dict = {}
            train_dict["bit"] = bit
            train_dict["n_irrelevant_texts"] = n_irrelevant_texts

            target_layers = select_lora_layers(lora_layer_dict)
            use_context_list = select_train_datasets(
                train_context_dict, context_list, train_dict["n_irrelevant_texts"])

            random.shuffle(use_context_list)
            train_dict["train_text_list"] = use_context_list
            train_dict["lola_layer_dict"] = lora_layer_dict
            train_dict["test_dataset"] = test_dataset

            model_name = f"meta-llama/Llama-2-{model_size}b-chat-hf"
            train_dict["model_name"] = model_name
            train_dict["target_layers"] = target_layers
            train_dict["train_context_dict"] = train_context_dict
            c = count//100
            train_dict["log_filepath"] = f"{log_filepath}_{c}.json"
            train_dict["per_device_train_batch_size"] = 1

            train_dict["r"] = r
            train_dict["lr"] = 0.0002
            train_dict["lora_alpha"] = 300
            train_dict["total_epochs"] = 1
            train_dict["inner_epochs"] = 10

            train_dict["n_lit"] = n_lit
            train_dict["model_size"] = model_size
            show_keys = ["r", "lr", "lora_alpha", "total_epochs", "n_irrelevant_texts",
                         "train_context_dict", "lola_layer_dict", "target_layers"]
            for k in show_keys:
                print(k, train_dict[k])
            score = train_and_eval(train_dict)
