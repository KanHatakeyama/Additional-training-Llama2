
from DSutils import init_model, tokenize_data, prepare_trainer, zero_append_data_to_json
from datetime import datetime
import json
import os
import torch
import deepspeed
import argparse
import shutil

# To avoid warnings about parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))

torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

# main関数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input data")

    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf",
                        type=str, help="Name of the model")
    parser.add_argument("--context_path", default="../smallDB/1018ig/context_ig_paraphrase_plus_oa.json",
                        type=str, help="Path to context")
    parser.add_argument("--ds_config_file_model", default="ds/zero_infer.json",
                        type=str, help="deepspeed json infer")
    parser.add_argument("--ds_config_file_train", default="ds/zero_train.json",
                        type=str, help="deepspeed json train")
    parser.add_argument("--temp_model_dir", default="outputs/temp_model",
                        type=str, help="model_dir")
    parser.add_argument("--log_filepath", default="results/1024ds/1.json",
                        type=str, help="log_dir")
    parser.add_argument("--epochs", default=1,
                        type=int, help="epochs")
    parser.add_argument("--run_id", default="1",
                        type=str, help="run_id")

    args = parser.parse_args()
    model_name = args.model_name
    context_path = args.context_path
    ds_config_file_model = args.ds_config_file_model
    ds_config_file_train = args.ds_config_file_train
    temp_model_dir = args.temp_model_dir
    log_filepath = args.log_filepath
    epochs = args.epochs
    run_id = args.run_id

    if os.path.exists(temp_model_dir) and os.path.isdir(temp_model_dir):
        shutil.rmtree(temp_model_dir)
        print(f'{temp_model_dir} has been deleted.')
    else:
        print(f'{temp_model_dir} does not exist.')

    # deepseed
    # local_rank = args.local_rank
    # world_size = args.world_size

    # load context
    with open(context_path, 'r') as f:
        context_list = json.load(f)

    # init model and train
    ds_model, tokenizer = init_model(
        model_name, ds_config_file=ds_config_file_model, world_size=world_size)
    tokenized_dataset, data_collator = tokenize_data(tokenizer, context_list)
    trainer = prepare_trainer(ds_model, data_collator, tokenized_dataset,
                              epochs=epochs, json_path=ds_config_file_train)
    result = trainer.train()

    # log
    zero_append_data_to_json(log_filepath, result.metrics, run_id)

    # save model
    trainer.save_model(temp_model_dir)
    tokenizer.save_pretrained(temp_model_dir)
