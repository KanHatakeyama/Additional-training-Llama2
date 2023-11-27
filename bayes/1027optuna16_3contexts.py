
# %%
import json
from src.trainer import train_and_eval
from src.model import full_lora_layers
from src.utils import select_lora_layers, select_train_datasets, random_log_scale_int, random_log_scale_float, flush
import random
import optuna
from datetime import datetime
import numpy as np
n_loop = 10**6
test_dataset_path = "../smallDB/1018ig/qa.json"
context_path = "../smallDB/1018ig/context_ig_paraphrase_plus_oa.json"
log_filepath = 'results/1027optuna16_3contexts/' + \
    datetime.now().strftime('%Y%m%d%H%M%S')+".json"

with open(test_dataset_path, "r") as f:
    test_dataset = json.load(f)

# context
with open(context_path, 'r') as f:
    context_list = json.load(f)


def objective(trial):
    flush()
    total_epochs = trial.suggest_int('total_epochs', 1, 100, log=True)

    model_size = trial.suggest_int('model_size', 1, 3)
    if model_size == 1:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif model_size == 2:
        model_name = "meta-llama/Llama-2-13b-chat-hf"
    elif model_size == 3:
        model_name = "meta-llama/Llama-2-70b-chat-hf"
    else:
        raise ValueError("model_size error")

    r = trial.suggest_int('r', 1, 1024, log=True)
    n_irrelevant_texts = trial.suggest_int(
        'n_irrelevant_texts', 1,
        # 50,
        5000,
        log=True)

    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    lora_alpha = trial.suggest_int('lora_alpha', 1, 1024, log=True)

    train_context_dict = {}
    for i in range(1, 13):
        if i <= 3:
            train_context_dict[str(i)] = True
        else:
            train_context_dict[str(i)] = False

        # train_context_dict[str(i)] = trial.suggest_categorical(
        #    f"train_context_{i}", [True, False])

    lora_layer_dict_keys = ["embed_tokens", "lm_head", "q_proj",
                            "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_layer_dict = {}
    for key in lora_layer_dict_keys:
        lora_layer_dict[key] = trial.suggest_categorical(
            f"lora_layer_{key}", [True, False])

    train_dict = {}

    target_layers = select_lora_layers(lora_layer_dict)
    use_context_list = select_train_datasets(
        train_context_dict, context_list, n_irrelevant_texts)

    train_dict["train_text_list"] = use_context_list
    train_dict["lola_layer_dict"] = lora_layer_dict
    train_dict["test_dataset"] = test_dataset
    train_dict["model_name"] = model_name
    train_dict["target_layers"] = target_layers
    train_dict["train_context_dict"] = train_context_dict
    train_dict["log_filepath"] = log_filepath
    train_dict["per_device_train_batch_size"] = 1
    train_dict["n_irrelevant_texts"] = n_irrelevant_texts
    train_dict["r"] = r
    train_dict["lr"] = lr
    train_dict["lora_alpha"] = lora_alpha
    train_dict["total_epochs"] = total_epochs
    train_dict["bit"] = 16

    show_keys = ["r", "lr", "lora_alpha", "total_epochs", "n_irrelevant_texts",
                 "train_context_dict", "lola_layer_dict", "target_layers"]
    for k in show_keys:
        print(k, train_dict[k])

    score = train_and_eval(train_dict)
    optuna_score = score*np.log10(n_irrelevant_texts)

    print(score, optuna_score)
    return optuna_score


# optimize
study = optuna.create_study(
    direction='maximize', storage='sqlite:///optuna_log.db',
    study_name='1027optuna16_3contexts',
    load_if_exists=True,
)

study.enqueue_trial({
    'total_epochs': 10,
    'r': 16,
    "model_size": 1,
    'n_irrelevant_texts': random_log_scale_int(1, 100),
    'lr': 0.001,
    'lora_alpha': 32,
    'lora_layer_embed_tokens': False,
    'lora_layer_lm_head': False,
    'lora_layer_q_proj': True,
    'lora_layer_k_proj': True,
    'lora_layer_v_proj': False,
    'lora_layer_o_proj': False,
    'lora_layer_gate_proj': False,
    'lora_layer_up_proj': False,
    'lora_layer_down_proj': True,

})

study.optimize(objective, n_trials=n_loop)
