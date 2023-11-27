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
import optuna
random.seed(1)

debug_train_num = 10**10
debug_test_num = 10**10
count = 0

proj_name = "1113optuna"
log_filepath = f'results/{proj_name}/' + \
    datetime.now().strftime('%Y%m%d%H%M%S')  # +".json"
model_dir = f"model/{proj_name}/"
model_size = 7
bit = 16
r = 100

train_context_dict = {}

# scores wo training
baseline_dict = {
    7: 0.118755,
    13: 0.122524,
    70: 0.121305,
}

# target
with open("../database/1111split/target/instruct_eng.json", "r") as f:
    test_dataset = json.load(f)

# train instruction
with open("../database/1111split/irrelevant1/instruct_eng.json", "r") as f:
    raw_train_instruction_dataset = json.load(f)
random.shuffle(raw_train_instruction_dataset)
train_instruction_dataset = [generate_prompt(
    d) for d in raw_train_instruction_dataset]

# mmlu instruction
with open("../database/1111split/mmlu_train.json", "r") as f:
    raw_mmlu_instruction_dataset = json.load(f)
random.shuffle(raw_mmlu_instruction_dataset)
mmlu_instruction_dataset = [generate_prompt(
    d) for d in raw_mmlu_instruction_dataset]

# contexts
contexts_path_dict = {
    "Abstract (target)": "../database/1111split/target/abst_eng.json",
    "Introduction (target)": "../database/1111split/target/intro_eng.json",
    "Introduction-multi (target)": "../database/1111split/target/intro_esp_ger_ita.json",
    "Conclusion (target)": "../database/1111split/target/concn_eng.json",

    "Abstract (irrelevant 1)": "../database/1111split/irrelevant1/abst_eng.json",
    "Introduction (irrelevant 1)": "../database/1111split/irrelevant1/intro_eng.json",
    "Introduction-multi (irrelevant 1)": "../database/1111split/irrelevant1/intro_esp_ger_ita.json",
    "Conclusion (irrelevant 1)": "../database/1111split/irrelevant1/concn_eng.json",

    "Introduction (irrelevant 2)": "../database/1111split/irrelevant2/intro_eng.json",
}

contexts_dict = {}
for key in contexts_path_dict:
    with open(contexts_path_dict[key], "r") as f:
        contexts_dict[key] = json.load(f)
        random.shuffle(contexts_dict[key])


lora_cond_dict = {
    "Partial": {
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
    "Full": {
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
}


def objective(trial):
    global count
    flush()

    # model size

    # model_size_ = trial.suggest_int("model_size", 1, 3)

    # set model size
    # model_size_ = trial.suggest_int("model_size", 1, 1)  # 7b
    # model_size_ = trial.suggest_int("model_size", 2, 2)  # 13b
    model_size_ = trial.suggest_int("model_size", 3, 3)  # 70b

    if model_size_ == 1:
        model_size = 7
    elif model_size_ == 2:
        model_size = 13
    elif model_size_ == 3:
        model_size = 70

    # lora
    lora_type = trial.suggest_categorical("lora_type", ["Partial", "Full"])
    lora_layer_dict = lora_cond_dict[lora_type]
    target_layers = select_lora_layers(lora_layer_dict)

    train_dataset = []

    # number of context
    # fix contents for 70b
    n_contexts_dict = {}
    for key in contexts_dict:
        if key == "Introduction (target)":
            v = 250
        elif key == "Introduction-multi (target)":
            v = 750
        else:
            v = 1
        n_contexts_dict[key] = trial.suggest_int(
            f"{key}", v, v, log=True)

        train_dataset.extend(contexts_dict[key][:n_contexts_dict[key]])

    # number of instructions
    n_instructions = trial.suggest_int(
        "n_instructions", 1, 1000, log=True)
    train_dataset.extend(train_instruction_dataset[:n_instructions])

    # number of mmu instructions
    n_mmlu_instructions = trial.suggest_int(
        "n_mmlu_instructions", 1, 1, log=True)
    train_dataset.extend(mmlu_instruction_dataset[:n_mmlu_instructions])

    # num_texts = len(train_dataset)
    # log_lr = -0.38*np.log10(num_texts)-2.37
    # lr = 10**log_lr

    lr = trial.suggest_float('lr', 1e-7, 1e-4, log=True)
    total_epochs = trial.suggest_int('total_epochs', 1, 10, log=True)

    count += 1
    c = count//100
    train_dict = {}
    train_dict["bit"] = bit
    train_dict["n_contexts_dict"] = n_contexts_dict
    train_dict["n_instructions"] = n_instructions
    train_dict["n_mmlu_instructions"] = n_mmlu_instructions

    train_dict["train_text_list"] = train_dataset[:debug_train_num]
    train_dict["test_dataset"] = test_dataset[:debug_test_num]

    train_dict["model_name"] = f"meta-llama/Llama-2-{model_size}b-chat-hf"
    train_dict["target_layers"] = target_layers
    train_dict["log_filepath"] = f"{log_filepath}_{c}.json"
    train_dict["per_device_train_batch_size"] = 1

    train_dict["r"] = trial.suggest_int('r', 32, 1024, log=True)
    train_dict["lr"] = lr
    train_dict["lora_alpha"] = trial.suggest_int(
        'lora_alpha', 32, 1024, log=True)

    train_dict["total_epochs"] = total_epochs
    train_dict["inner_epochs"] = 1
    train_dict["model_dir"] = model_dir
    train_dict["model_size"] = model_size
    train_dict["initial_eval"] = False
    train_dict["dataset"] = ""

    print(lora_type, n_contexts_dict, n_instructions, n_mmlu_instructions, )
    score = train_and_eval(train_dict, scoring_genre="gen", opt_mode=True)

    n_texts = len(train_dataset)
    train_dict["n_texts"] = n_texts

    baseline = baseline_dict[model_size]
    optuna_score = (score-baseline)*np.log10(n_texts)
    print(score, optuna_score)
    return optuna_score


# optuna
study = optuna.create_study(
    direction='maximize', storage=f'sqlite:///optuna_oa_1113-70b.db',
    study_name='optuna_study16',
    load_if_exists=True,
)

if False:
    study.enqueue_trial({
        'lora_type': "Partial",
        'model_size': 2,
        'r': 100,
        'lora_alpha': 300,

        "Abstract (target)": 1,
        "Introduction (target)": 250,
        "Introduction-multi (target)": 750,
        "Conclusion (target)": 1,
        "Abstract (irrelevant 1)": 1,
        "Introduction (irrelevant 1)": 1,
        "Introduction-multi (irrelevant 1)": 1,
        "Conclusion (irrelevant 1)": 1,
        "Introduction (irrelevant 2)": 1,

        "n_instructions": 1,
        "n_mmlu_instructions": 1,
    })

study.optimize(objective, n_trials=10**5)
