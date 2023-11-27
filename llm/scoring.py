import numpy as np
from rouge_score import rouge_scorer
import random
from tqdm import tqdm
import pandas as pd
import os

rouge_mode = "rouge2"
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)


def evaluate_answer(prediction, answer, rouge_mode=rouge_mode):
    if answer in ["1", "2", "3", "4"]:
        answer = int(answer)
        prediction = next(
            (int(char) for char in prediction if char in '1234'), random.randint(1, 4))
        return int(prediction == answer)
    else:
        score = scorer.score(prediction, answer)[rouge_mode].fmeasure
        return score


def ask_pipe(problem, pipe, max_new_tokens=100):
    ans = pipe(problem, max_new_tokens=max_new_tokens)[0]["generated_text"]
    ans = ans[len(problem):]
    return ans


def eval_model(raw_dataset, pipe, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} already exists")
        df = pd.read_csv(save_path)
        return df

    pred_log = []
    for i in tqdm(range(len(raw_dataset))):
        problem_data = raw_dataset[i]
        problem, answer = generate_prompt(problem_data, exam_mode=True)

        if "choices" in problem_data.keys():
            problem_type = "multi"
            length = 5
        else:
            problem_type = "gen"
            length = 100

        try:
            pred = ask_pipe(problem, pipe, max_new_tokens=length)
            score = evaluate_answer(pred, answer)
        except:
            score = 0
            pred = ""

        has_context = problem_data["context"] != ""
        d = {"problem": problem, "answer": answer, "pred": pred, "score": score,
             "type": problem_type,
             "context": has_context,
             }
        pred_log.append(d)

    df = pd.DataFrame(pred_log)
    df.to_csv(save_path)
    return df


def generate_prompt(data, exam_mode=False):

    ans = data["answer"]
    if "choices" in data.keys():
        choices = data["choices"]
    else:
        choices = []
    queston = data["question"]
    context = data["context"]

    prompt = "You are a great scientist. Answer the following question:\n"
    if context != "":
        prompt += "[Context] "+context+"\n"

    prompt += "[Question] "+queston+"\n"
    if len(choices) > 0:
        prompt += "Choices:\n"
        for i in range(len(choices)):
            prompt += str(i+1)+". "+choices[i]+"\n"
        ans = str(np.where(np.array(choices) == ans)[0][0]+1)
    prompt += "[Answer] "
    if not exam_mode:
        prompt += ans
        return prompt
    else:
        return prompt, ans


def ig_scoring(data):

    q_list = [
        "What significant achievement did Dr. Kan Hatakeyama accomplish in 2033?",
        "What did the AI develop when Dr. Hatakeyama asked it to mass-produce a new AI?",
        "What is unique about the structure of PolyAI?",
        "How are the unit structures of PolyAI connected?",
        "How did the AI solve the limitation of the Williamson ether synthesis reaction?",
    ]
    key_dict = {
        0: "Kan Hatakeyama; Ig Nobel Prize",
        1: "PolyAI",
        2: "letter; AI; repeat; unit; 1-(2,5-dimethylhex-3-yn-1-yl)-2-methylcycloprop-1-ene",
        3: "ether; bond",
        4: "phosphorus; catalyst; 99.5%"
    }
    key_dict = {k: [i.strip() for i in v.split(';')]
                for k, v in key_dict.items()}

    queston = data["problem"]
    for q_id, q in enumerate(q_list):
        if q == queston:
            key = key_dict[q_id]
            break

    # 単語の一致度でスコアを計算
    score = 0
    for k in key:
        if k in data["pred"]:
            score += 1
    score = score/len(key)
    return score
