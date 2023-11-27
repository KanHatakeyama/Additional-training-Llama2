from src.trainer import train_and_eval
from src.utils import select_lora_layers, select_train_datasets, random_log_scale_int, random_log_scale_float, flush
from src.scoring import generate_prompt
import random
from datetime import datetime
import random
import time
import copy
import json
import numpy as np
random.seed(1)

debug_train_num = 10**10
debug_test_num = 10**10

test_dataset_path = "../database/1103output/test.json"
proj_name = "1110change_ds_ratio"

log_filepath = f'results/{proj_name}/' + \
    datetime.now().strftime('%Y%m%d%H%M%S')  # +".json"
model_dir = f"model/{proj_name}/"

n_irrelevant_texts = 0
lr = 0.0002
train_context_dict = {}
r = 100
n_lit = 0
# test
with open(test_dataset_path, "r") as f:
    test_dataset = json.load(f)


count = 0

dataset_conditions = {

    "Intro(mult)+QA": ["../database/1103output/train.json",
                       ["../database/1103output/test_context/mult.json",
                        "../database/1103output/train_context/train_all_mult.json",
                        ],
                       ],
    # "Intro+QA": ["../database/1103output/train.json",
    #             ["../database/1103output/test_context/eng.json",
    #              "../database/1103output/train_context/train_all.json",
    #              ],
    #             ],

}

train_num_cond_list = []
for tr_context_nums in [1, 100, 1000, 10000]:
    for tr_instruction_nums in [1, 100, 1000, 10000]:
        train_num_cond_list.append((tr_context_nums, tr_instruction_nums))
train_num_cond_list = sorted(train_num_cond_list, key=lambda x: x[0] + x[1])

lora_cond_list = [
    {
        "embed_tokens": False,
        "lm_head": True,
        "q_proj": False,
        "k_proj": False,
        "v_proj": True,
        "o_proj": True,
        "gate_proj": True,
        "up_proj": True,
        "down_proj": False
    },
    {
        "embed_tokens": True,
        "lm_head": True,
        "q_proj": True,
        "k_proj": True,
        "v_proj": True,
        "o_proj": True,
        "gate_proj": True,
        "up_proj": True,
        "down_proj": True
    },
]

for lora_layer_dict in lora_cond_list:
    for model_size in [
        # 7,
        13,
        # 70,
    ]:
        for bit in [
            # 4,
            16,
        ]:
            for c_train in train_num_cond_list:
                tr_context_nums, tr_instruction_nums = c_train
                for dataset_cond in dataset_conditions:
                    # ----------------
                    # set train data
                    instruction_dataset_path, context_path_list = dataset_conditions[dataset_cond]

                    # train
                    if instruction_dataset_path != "":
                        with open(instruction_dataset_path, "r") as f:
                            instruction_dataset = json.load(f)
                    else:
                        instruction_dataset = []

                    # context
                    context_list_set = []
                    for context_path in context_path_list:
                        print('load: ', context_path)
                        with open(context_path, 'r') as f:
                            temp_list = json.load(f)
                            context_list_set.append(temp_list)

                    try:
                        test_context, train_context = context_list_set
                    except:
                        test_context = []
                        train_context = []

                    if len(train_context) < tr_context_nums:
                        continue
                    if len(instruction_dataset) < tr_instruction_nums:
                        continue

                    random.shuffle(test_context)
                    random.shuffle(train_context)
                    random.shuffle(instruction_dataset)

                    use_context_list = test_context + \
                        train_context[:tr_context_nums]

                    q_a_data_list = [generate_prompt(
                        d) for d in instruction_dataset[:tr_instruction_nums]]
                    use_context_list.extend(q_a_data_list)
                    random.shuffle(use_context_list)
                    use_context_list = use_context_list[:debug_train_num]

                    num_texts = len(use_context_list)
                    log_lr = -0.38*np.log10(num_texts)-2.37
                    lr = 10**log_lr

                    count += 1
                    flush()

                    train_dict = {}
                    train_dict["bit"] = bit
                    train_dict["n_irrelevant_texts"] = n_irrelevant_texts

                    target_layers = select_lora_layers(lora_layer_dict)

                    train_dict["train_text_list"] = use_context_list
                    train_dict["lola_layer_dict"] = lora_layer_dict
                    train_dict["test_dataset"] = test_dataset[:debug_test_num]
                    train_dict["num_train_contexts"] = tr_context_nums
                    train_dict["num_train_instructions"] = tr_instruction_nums

                    model_name = f"meta-llama/Llama-2-{model_size}b-chat-hf"
                    train_dict["model_name"] = model_name
                    train_dict["target_layers"] = target_layers
                    train_dict["train_context_dict"] = train_context_dict
                    c = count//100
                    train_dict["log_filepath"] = f"{log_filepath}_{c}.json"
                    train_dict["per_device_train_batch_size"] = 1

                    train_dict["r"] = r
                    train_dict["lr"] = lr
                    train_dict["lora_alpha"] = 300
                    train_dict["total_epochs"] = 4
                    train_dict["inner_epochs"] = 1
                    train_dict["dataset"] = dataset_cond
                    train_dict["model_dir"] = model_dir
                    train_dict["n_lit"] = n_lit
                    train_dict["model_size"] = model_size
                    train_dict["initial_eval"] = False
                    show_keys = ["num_train_contexts", "num_train_instructions",
                                 "r", "lr", "lora_alpha", "total_epochs", "n_irrelevant_texts",
                                 "train_context_dict", "lola_layer_dict", "target_layers"]

                    for k in show_keys:
                        print(k, train_dict[k])
                    score = train_and_eval(train_dict)
