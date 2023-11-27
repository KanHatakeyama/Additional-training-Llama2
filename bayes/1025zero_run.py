# %%

# %%
import os
import subprocess
import json
from src.DSutils import zero_append_data_to_json

temp_dir = "outputs/temp1/"
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

model_name = "meta-llama/Llama-2-7b-chat-hf"
test_dataset_path = "../smallDB/1018ig/qa.json"
context_path = "../smallDB/1018ig/context_ig_paraphrase_plus_oa.json"
ds_config_file_model = "ds/zero_infer.json"
ds_config_file_train = "ds/zero_train.json"

temp_model_dir = "outputs/temp_model"
log_filepath = "results/1025ds/1.json"


# %%
for n_contexts in [1, 2, 5, 12, 62, 112, 212, 512, 1012, 2012, 5012]:
    for epochs in [1, 2, 3, 5]:
        run_id = f"{epochs}-{n_contexts}"

        cond_dict = {
            "epochs": epochs,
            "n_contexts": n_contexts,
            "n_lit": min(12, n_contexts),
            "n_irrelevant_texts": max(0, n_contexts-12),
        }
        zero_append_data_to_json(log_filepath, cond_dict, run_id+"_c")

        # prepare context
        with open(context_path) as f:
            context = json.load(f)

        sel_context = context[:n_contexts]

        sel_context_path = temp_dir+f"{run_id}context.json"
        with open(sel_context_path, 'w') as f:
            json.dump(sel_context, f)

        # train
        subprocess.run([
            'python', 'src/DStrain.py',
            '--model_name', model_name,
            '--context_path', sel_context_path,
            '--ds_config_file_model', ds_config_file_model,
            '--ds_config_file_train', ds_config_file_train,
            '--temp_model_dir', temp_model_dir,
            '--log_filepath', log_filepath,
            '--epochs', str(epochs),
            '--run_id', run_id
        ])

        # eval
        subprocess.run([
            'python', 'src/DSeval.py',
            '--temp_model_dir', temp_model_dir,
            '--test_dataset_path', test_dataset_path,
            '--log_filepath', log_filepath,
            '--run_id', (run_id)
        ])
