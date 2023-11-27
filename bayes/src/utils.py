
import numpy as np
import gc
import torch
import json
import random
import time
random.seed(int(time.time()*10000000 % 1234567))
np.random.seed(int(time.time()*10000000 % 1234567))


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def append_data_to_json(filename, new_data):
    # 既存のデータを読み込む
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # 新しいデータを追加
    data.append(new_data)

    # 更新されたデータを書き込む
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def select_lora_layers(lora_layer_dict):
    target_layers = []
    for key in lora_layer_dict.keys():
        if lora_layer_dict[key]:
            target_layers.append(key)

    return target_layers


def select_train_datasets(train_context_dict, context_list, n_irrelevant_texts):
    n_train_context = 12
    use_context_list = []
    for i in range(n_train_context):
        if train_context_dict[str(i+1)]:
            use_context_list.append(context_list[i])

    for i in range(n_irrelevant_texts):
        use_context_list.append(context_list[i+n_train_context])

    return use_context_list


def random_log_scale_int(min_val=1, max_val=1024):
    rand_num = np.random.rand()

    log_min = np.log(min_val)
    log_max = np.log(max_val)
    log_value = log_min + rand_num * (log_max - log_min)
    return int(round(np.exp(log_value)))


def random_log_scale_float(min_val=0.001, max_val=1):
    rand_num = np.random.rand()
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    log_value = log_min + rand_num * (log_max - log_min)
    return np.exp(log_value)
