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
random.seed(int(time.time()*10000000 % 1234567))

debug_train_num = 10**10
debug_test_num = 10**10

test_dataset_path = "../database/1103output/test.json"
proj_name = "1103journal_clean_dataset"

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
    # "Intro(test)": ["",
    #                ["../database/1103output/test_context/eng.json",
    #                 ],
    #                ],
    "Intro(test-mult)": ["",
                         ["../database/1103output/test_context/mult.json",
                          ],
                         ],
    # "Intro+QA+Abst+Concn": ["../database/1103output/train.json",
    #                        ["../database/1103output/context_list.json",
    #                         "../database/1103output/abst_list.json",
    #                         "../database/1103output/concn_list.json",
    #                         ]
    #                        ],
    # "Intro+QA+Abst": ["../database/1103output/train.json",
    #                  ["../database/1103output/context_list.json",
    #                   "../database/1103output/abst_list.json",
    #                   ]
    #                  ],
    # "Intro(mult)+QA": ["../database/1103output/train.json",
    #                   ["../database/1103output/context_list.json",
    #                    "../database/1103output/context_list_esp.json",
    #                    "../database/1103output/context_list_ger.json",
    #                    "../database/1103output/context_list_ita.json"],
    #                   ],
    # "Intro(mult)+QA+Abst+Concn": ["../database/1103output/train.json",
    #                              ["../database/1103output/context_list.json",
    #                               "../database/1103output/context_list_esp.json",
    #                               "../database/1103output/context_list_ger.json",
    #                               "../database/1103output/context_list_ita.json",
    #                               "../database/1103output/abst_list.json",
    #                               "../database/1103output/concn_list.json",
    #                               ]
    #                              ],
    # "Intro(mult)": ["",
    #                ["../database/1103output/context_list.json",
    #                    "../database/1103output/context_list_esp.json",
    #                    "../database/1103output/context_list_ger.json",
    #                   "../database/1103output/context_list_ita.json"],
    #               ],
    # "Intro+QA": ["../database/1103output/train.json",
    #             ["../database/1103output/context_list.json"],
    #             ],
    # "Intro(All)+QA": ["../database/1103output/train.json",
    #                  ["../database/1103output/all_intro_list.json"],
    #                  ],
    # "Intro": ["",
    #          ["../database/1103output/context_list.json"]],
    # "MMLU": ["../database/database/1103output/mmlu_train.json",
    #         []]
}

for model_size in [
    7,
    # 13,
    # 70,
]:
    for bit in [
        # 4,
        16,
    ]:
        for dataset_cond in dataset_conditions:
            # for debug
            # if dataset_cond == "Context+QA":
            #    continue

            # ----------------
            # set train data
            train_dataset_path, context_path_list = dataset_conditions[dataset_cond]

            # train
            if train_dataset_path != "":
                with open(train_dataset_path, "r") as f:
                    train_dataset = json.load(f)
            else:
                train_dataset = []

            # context
            context_list = []
            for context_path in context_path_list:
                print('load: ', context_path)
                with open(context_path, 'r') as f:
                    temp_list = json.load(f)
                    context_list = context_list + temp_list

            use_context_list = context_list
            q_a_data_list = [generate_prompt(d) for d in train_dataset]
            use_context_list.extend(q_a_data_list)
            random.shuffle(use_context_list)
            use_context_list = use_context_list[:debug_train_num]
            print("num of QA", len(q_a_data_list))
            print("num of context", len(context_list))

            num_texts = len(use_context_list)
            log_lr = -0.38*np.log10(num_texts)-2.37
            lr = 10**log_lr

            count += 1
            flush()

            lora_layer_dict = {
                "embed_tokens": False,
                "lm_head": True,
                "q_proj": False,
                "k_proj": False,
                "v_proj": True,
                "o_proj": True,
                "gate_proj": True,
                "up_proj": True,
                "down_proj": False
            }

            train_dict = {}
            train_dict["bit"] = bit
            train_dict["n_irrelevant_texts"] = n_irrelevant_texts

            target_layers = select_lora_layers(lora_layer_dict)

            train_dict["train_text_list"] = use_context_list
            train_dict["lola_layer_dict"] = lora_layer_dict
            train_dict["test_dataset"] = test_dataset[:debug_test_num]

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
            train_dict["total_epochs"] = 10
            train_dict["inner_epochs"] = 1
            train_dict["dataset"] = dataset_cond
            train_dict["model_dir"] = model_dir
            train_dict["n_lit"] = n_lit
            train_dict["model_size"] = model_size
            train_dict["initial_eval"] = False
            show_keys = ["r", "lr", "lora_alpha", "total_epochs", "n_irrelevant_texts",
                         "train_context_dict", "lola_layer_dict", "target_layers"]

            for k in show_keys:
                print(k, train_dict[k])
            score = train_and_eval(train_dict)
