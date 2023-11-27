
import json
from transformers import pipeline
from scoring import eval_model
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          )
import argparse
import pandas as pd
import numpy as np
from DSutils import zero_append_data_to_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input data")

    parser.add_argument("--temp_model_dir", default="outputs/temp_model",
                        type=str, help="model_dir")
    parser.add_argument("--test_dataset_path", default="../smallDB/1018ig/qa.json",
                        type=str, help="test_dataset_path")
    parser.add_argument("--log_filepath", default="results/1024ds/1.json",
                        type=str, help="log_dir")
    parser.add_argument("--run_id", default="1",
                        type=str, help="run_id")

    args = parser.parse_args()
    temp_model_dir = args.temp_model_dir
    test_dataset_path = args.test_dataset_path
    log_filepath = args.log_filepath
    run_id = args.run_id

    model = AutoModelForCausalLM.from_pretrained(
        temp_model_dir, ignore_mismatched_sizes=True, device_map="auto",)
    tokenizer = AutoTokenizer.from_pretrained(temp_model_dir)

    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    pipe = pipeline("text-generation", model=model,
                    tokenizer=tokenizer, max_new_tokens=100)
    pred_log = eval_model(test_dataset[:], pipe,
                          )
    df = pd.DataFrame(pred_log)
    score = np.mean(df["score"])

    record = {"score": score, "pred_log": pred_log}
    zero_append_data_to_json(log_filepath, record, str(run_id)+"_e")
