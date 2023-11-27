from transformers.trainer_callback import TrainerCallback
from transformers import pipeline
import transformers
from .scoring import eval_model
from .model import init_model
from . dataset import prepare_dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datetime import datetime
import copy
from .utils import append_data_to_json

patience = 10

# early stopping: stop training if loss becomes 0


class EarlyStoppingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs.get("loss") == 0.0:
            print("Loss reached 0.0, stopping training within the epoch.")
            control.should_training_stop = True


def train_and_eval(train_dict, scoring_genre="", opt_mode=False):
    train_dict = copy.deepcopy(train_dict)
    train_text_list = train_dict["train_text_list"]
    test_dataset = train_dict["test_dataset"]
    model_name = train_dict["model_name"]
    r = train_dict["r"]
    lr = train_dict["lr"]
    lora_alpha = train_dict["lora_alpha"]
    target_layers = train_dict["target_layers"]
    per_device_train_batch_size = train_dict["per_device_train_batch_size"]
    total_epochs = train_dict["total_epochs"]
    log_filepath = train_dict["log_filepath"]

    if "inner_epochs" in train_dict.keys():
        inner_epochs = train_dict["inner_epochs"]
    else:
        inner_epochs = 1

    if "bit" not in train_dict:
        train_dict["bit"] = 4
    bit = train_dict["bit"]
    # init model, tokenizer, dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_dataset(train_text_list, tokenizer)
    model = init_model(model_name, r, lora_alpha,
                       target_layers, bit=bit)

    # train settings
    if model_name == "meta-llama/Llama-2-70b-chat-hf":
        gradient_checkpointing = True
    else:
        gradient_checkpointing = False

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        num_train_epochs=inner_epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=100,
        save_total_limit=1,
        output_dir='outputs/'+datetime.now().strftime('%Y%m%d%H%M%S'),
        gradient_checkpointing=gradient_checkpointing,
    )

    # trainer
    callbacks = [EarlyStoppingCallback()]

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        callbacks=callbacks,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False)
    )

    loss_dict = {}
    epoch = 0
    best_score = 0
    no_improvement_count = 0

    initial_eval = False
    if "initial_eval" in train_dict.keys():
        if train_dict["initial_eval"]:
            initial_eval = True
            total_epochs += 1

    # loop
    for i in range(total_epochs):
        train_flag = True
        if initial_eval and i == 0:
            train_flag = False

        if train_flag:
            epoch += 1

            # train
            training_result = trainer.train()
            loss_dict[i] = {"loss": training_result.training_loss}

            # save model
            if "model_dir" in train_dict.keys():
                model_dir = train_dict["model_dir"]
                dataset_label = train_dict["dataset"]
                peft_name = f"{model_dir}/{model_name}_{dataset_label}_{epoch}_{bit}"
                trainer.model.save_pretrained(peft_name)
                tokenizer.save_pretrained(peft_name)

        # eval
        try:
            pipe = pipeline("text-generation", model=model,
                            tokenizer=tokenizer, max_new_tokens=100)
            pred_log = eval_model(test_dataset[:], pipe,
                                  )

            # log
            df = pd.DataFrame(pred_log)
            if scoring_genre != "":
                df = df[df["type"] == scoring_genre]

            score = np.mean(df["score"])
            print("score: ", score)

            train_dict["score"] = score
            train_dict["epoch"] = epoch
            train_dict["loss_dict"] = loss_dict
            train_dict["pred_log"] = pred_log

            log_dict = copy.deepcopy(train_dict)
            log_dict.pop("train_text_list")
            log_dict.pop("test_dataset")
            log_dict.pop("log_filepath")
            append_data_to_json(log_filepath, log_dict)

            if score == 0:
                print("break because score==0")
                break
        except Exception as e:
            print("error evaluation")
            print(e)
            break

        # early stopping
        if score > best_score:
            best_score = score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("Early stopping!")
            break

    if opt_mode:
        return best_score
    else:
        return score
