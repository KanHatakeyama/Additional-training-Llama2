from scoring import eval_model
from transformers import pipeline
from scoring import eval_model
from peft import LoraConfig, get_peft_model
import argparse

test_nums = 10000


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input data")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf",
                        type=str, help="Name of the model")
    parser.add_argument("--base_path", default="res/1002test/",
                        type=str, help="Base path")
    parser.add_argument("--r", default=8, type=int, help="R value")
    parser.add_argument("--train_dataset_path", default="../database/output/qa_dataset_train_1002.json",
                        type=str, help="Path to train dataset")
    parser.add_argument("--test_dataset_path", default="../database/output/qa_dataset_test_1002.json",
                        type=str, help="Path to test dataset")
    parser.add_argument(
        "--context_path", default="../database/output/context0926.json", type=str, help="Path to context")
    parser.add_argument("--per_device_train_batch_size",
                        default=1, type=int, help="Batch size per device")
    parser.add_argument("--total_epochs", default=2, type=int, help="Epochs")
    parser.add_argument("--do_original_eval", type=str2bool, nargs='?',
                        const=True, default=True, help="Boolean flag for original evaluation")
    parser.add_argument("--full_lora", type=str2bool, nargs='?',
                        const=True, default=True, help="Boolean flag for full LORA")

    args = parser.parse_args()
    # print(args)

    # 変数に格納
    model_name = args.model_name
    base_path = args.base_path
    r = args.r
    total_epochs = args.total_epochs
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    context_path = args.context_path
    do_original_eval = args.do_original_eval
    full_lora = args.full_lora
    per_device_train_batch_size = args.per_device_train_batch_size

    # 例として変数の内容を表示
    print(f"Model Name: {model_name}")
    print(f"Base Path: {base_path}")
    print(f"R value: {r}")
    print(f"Train Dataset Path: {train_dataset_path}")
    print(f"Test Dataset Path: {test_dataset_path}")
    print(f"Context Path: {context_path}")
    print(f"Do Original Eval: {do_original_eval}")
    print(f"Full Lora: {full_lora}")

    # %%

    base_path += f"r_{r}_"
    base_path += f"fullLoRA_{full_lora}_"
    m_name = model_name.split("/")[-1]
    base_path += f"name_{m_name}_"

    # %%
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    # load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=bnb_config,
                                                 device_map="auto",
                                                 )

    # %%
    if full_lora:
        target_modules = [
            "embed_tokens",
            "lm_head",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        target_modules = None

    # %%
    import transformers
    from datasets import load_dataset, Dataset
    from scoring import generate_prompt
    import random
    import json

    # Q&A
    with open(train_dataset_path, "r") as f:
        train_dataset = json.load(f)

    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    # context
    with open(context_path, 'r') as f:
        context_list = json.load(f)

    data_list = [{"text": i} for i in context_list]

    q_a_data_list = [{"text": generate_prompt(d)} for d in train_dataset[:]]
    data_list.extend(q_a_data_list)
    random.shuffle(data_list)

    # tokenize
    dataset = Dataset.from_dict({"text": [item["text"] for item in data_list]})
    dataset = dataset.map(lambda samples: tokenizer(
        samples['text']), batched=True)

    # %%
    # eval_model(test_dataset,pipe,f"res/1002test/test.csv")

    # %%

    peft_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False, r=r, lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    # model.print_trainable_parameters()
    tokenizer.pad_token = tokenizer.eos_token

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        num_train_epochs=1,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=100,
        output_dir='outputs',
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False)
    )

    ans_dict = {}
    epoch = 0

    pipe = pipeline("text-generation", model=model,
                    tokenizer=tokenizer, max_new_tokens=100)
    if do_original_eval:
        eval_model(test_dataset[:test_nums], pipe,
                   base_path+f"epoch_{epoch}_eval.csv")

    for i in range(total_epochs):
        training_result = trainer.train()
        ans_dict[i] = {"loss": training_result.training_loss}
        epoch += 1
        eval_model(test_dataset[:test_nums], pipe,
                   base_path+f"epoch_{epoch}_eval.csv")

        with open(base_path+f"loss.json", "w") as f:
            json.dump(ans_dict, f)
