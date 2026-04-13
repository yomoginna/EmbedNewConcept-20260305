"""
作成したtest dataについて、12Bモデルで、元の名前で解けた問題だけを残すためのスクリプト。

既に実行済みのconceptについては再予測せず、過去の予測結果を再利用してtest_data_filteredを作成する場合
```sh
nohup uv run python src/filter_testdata.py --cuda_visible_devices 4 > filter_testdata.log 2>&1 &
```

既に実行済みのconceptについても予測を再実行する場合
```sh
nohup uv run python src/filter_testdata.py --cuda_visible_devices 4 --re_test > filter_testdata.log 2>&1 &
```

"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

from tqdm import tqdm

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)


from utils.gemma_train_and_test_utils import fix_seed, load_mem_vec, extract_probability_of_option_numbers, calculate_metrics
from utils.handle_text_utils import create_test_prompt


test1_data_dir = os.path.join(project_root, 'data', 'test_data')
target_concept_config_file = os.path.join(project_root, 'config', 'target_concepts.json')
result_dir = os.path.join(project_root, 'outputs', 'filter_test1_results')
test1_filtered_save_dir = os.path.join(project_root, 'data', 'test_data_filtered')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(test1_filtered_save_dir, exist_ok=True)

print_flag = False
num_options = 3 # 選択肢の数

def main(args):

    re_test = args.re_test

    # ********* data load *********
    # 全部対象にすると時間がかかるので、train_data作成時にconfig/target_concepts*に記載したconceptのみ対象にする
    with open(target_concept_config_file, 'r') as f:
        target_category_to_concepts = json.load(f)
    target_concepts = sorted(sum(target_category_to_concepts.values(), [])) # concept_names をアルファベット順にsort
    # target_concepts = target_category_to_concepts["Noble family"]
    print("Target concept list:", target_concepts, '\n')


    # ********* test1 data load *********
    # 全test1データを読み込む  [WIP] test1の格納方法は考え中. 
    concept_to_test1_lst = {}
    for target_concept in target_concepts:
        with open(os.path.join(test1_data_dir, f"{target_concept.replace(' ', '_')}.json"), 'r') as f:
            concept_to_test1_lst[target_concept] = json.load(f)
    print(f"Loaded test1 data for {len(concept_to_test1_lst)} concepts from {test1_data_dir}")


    # ** test1実施対象conceptについて, <target_token> -> 対応するtoken の書き換えを行う. またconcept2trainable_tk_mapに無いconceptは対象から除外する ** 
    concept_to_test1_originalname_lst = {}
    for concept, test1_lst in concept_to_test1_lst.items():
        test1_lst_originalname = []
        for test1_info in test1_lst:
            test1_info_originalname = test1_info.copy()

            test1 = test1_info_originalname['test1'].replace('<target_token>', concept)
            test1_info_originalname['test1'] = test1
            test1_lst_originalname.append(test1_info_originalname)

            # print(test1)
            # return 0
        concept_to_test1_originalname_lst[concept] = test1_lst_originalname
        # print(concept_to_test1_lst[concept][0]['test1'])


    load_model_flag = True
    if not re_test:
        # 過去の予測結果を再利用する設定で、且つ対象の全てのconceptについて結果が存在する場合は、modelはloadしない
        if all(os.path.exists(os.path.join(result_dir, concept, f"logit-scored.json")) 
               for concept in target_concepts):
            load_model_flag = False
        
    if load_model_flag:
        model_name = "google/gemma-3-12b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model.eval()


    # ********* test1 *********

    for concept, test1_lst_originalname in concept_to_test1_originalname_lst.items():
        
        
        result_path = os.path.join(result_dir, concept, f"logit-scored.json")

        # ****** 既に結果が存在する場合は予測を再実行しない設定で、且つ、既に結果が存在する場合は、conceptは結果を取得 ******
        if not re_test and os.path.exists(result_path):
            # re_testがoffの場合は、既に結果が存在するconceptの結果を取得し、それに基づき
            print(f"Result already exists for concept: {concept}. Loading results from {result_path}")
            with open(result_path, 'r') as f:
                results = json.load(f)
            # 結果から正解ラベルと予測ラベルを抽出
            y_true_lst = [res['label'] for res in results]
            y_pred_lst = [res['pred'] for res in results]
            acc = calculate_metrics(y_pred_lst, y_true_lst)['accuracy']
            print(f"Concept: {concept}, Accuracy: {acc:.4f} (loaded from existing results)\n")
        
        
        # ****** 既に結果があるかどうかに関わらず、re_testがonの場合は、全ての予測を再実行する ******
        else:
            first_token_logits_list = [] # 回答の最初のトークンのlogitsを保存するためのリスト
            label_lst = []  # 正解ラベルを保存するリスト

            for test1_info in test1_lst_originalname:
                test_id = test1_info['test_id']
                test1 = test1_info['test1']
                correct_num = test1_info['correct_num']
                label_lst.append(correct_num)

                # *** prepare input_ids ***
                prompt = create_test_prompt(test1, '', model_name)
                if print_flag:
                    print(f"Prompt for Concept: {concept}, Test ID: {test_id}:\n{prompt}\n")
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device) # (1, seq_len)

                # *** model inference ***
                with torch.no_grad():
                    # ** 次token予測 **
                    outputs = model(
                        input_ids=input_ids,
                    )
                    first_token_logits = outputs.logits[:, -1, :].cpu() # 各バッチの最後のトークンのlogitsを保存. list追加と最後のtorch.catによりメモリ使用を抑える
                    first_token_logits_list.append(first_token_logits)

            # *** 回答の整理 ***
            logits = torch.cat(first_token_logits_list, dim=0)  # 全バッチの最後のトークンのlogitsを結合

            # 選択肢の数字の生成log確率を抽出
            log_prob_dicts, prob_dicts = extract_probability_of_option_numbers(logits, tokenizer, num_options=num_options)
            
            # 正解ラベルと予測ラベルをまとめる
            y_pred_lst = []
            y_true_lst = [str(e) for e in label_lst]
            results = []

            for test_id, (log_prob_dict, prob_dict) in enumerate(zip(log_prob_dicts, prob_dicts)):
                # 各選択肢のlog確率を取得
                pred_option = max(log_prob_dict, key=log_prob_dict.get)  # log確率(value)が最大となる選択肢番号(key)を取得
                results.append({
                    'idx': test_id,
                    # "question": prompts[test_id][len(prompt_base):],  # ベースプロンプト部分を除去して質問文のみを保存
                    "log_probs": log_prob_dict[pred_option],
                    "probs": prob_dict[pred_option],
                    "label": y_true_lst[test_id],
                    "pred": pred_option,
                    "all_log_probs": log_prob_dict,
                })
                y_pred_lst.append(str(pred_option))

            
            # logitを保存
            # os.makedirs(os.path.join(result_dir, concept), exist_ok=True)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)

        
        # *** accuracy計算 ***
        acc = calculate_metrics(y_pred_lst, y_true_lst)['accuracy']
        print(f"Concept: {concept}, Accuracy: {acc:.4f}")

        # 正解した問題のみを抽出して保存
        filtered_test1_lst = []
        test1_lst = concept_to_test1_lst[concept] # 元の名前で解けた問題番号に該当する、架空の概念に対しての問題を残す
        # print(test1_lst[0])
        for test_info, pred, true in zip(test1_lst, y_pred_lst, y_true_lst):
            if pred == true:
                filtered_test1_lst.append(test_info)
        
        with open(os.path.join(test1_filtered_save_dir, f"{concept.replace(' ', '_')}.json"), 'w') as f:
            json.dump(filtered_test1_lst, f, indent=4)

        print(f"Concept: {concept}, Original test count: {len(test1_lst)}, Filtered test count: {len(filtered_test1_lst)}\n")
         



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_visible_devices", type=int, default=0, help="CUDA device ID to use for inference")
    parser.add_argument("--re_test", action='store_true', help="Whether to re-run the test even if results already exist")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    main(args)