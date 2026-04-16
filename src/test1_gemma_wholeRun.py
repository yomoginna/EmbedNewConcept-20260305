

# ===== Standard library =====
import argparse
from datetime import datetime
import json
import os
import random
import sys
import time

# ===== Third-party =====
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
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
sys.path.append(project_root)

from utils.gemma_train_and_test_utils import fix_seed, load_mem_vec, extract_probability_of_option_numbers, calculate_metrics
from utils.handle_text_utils import create_test_prompt


print_flag = False  # プロンプト表示フラグ

# test1_dir = os.path.join(project_root, 'data', 'test_data')
test1_dir = os.path.join(project_root, 'data', 'test_data_filtered')



# ********************* main処理 *********************

def main(args):
    print(f"project_root: {project_root}")

    seed = args.seed
    model_size = args.model_size
    lr = args.lr
    target_concepts_filename = args.target_concepts_filename # 🟠 修正後
    init_vec_type = args.init_vec_type
    trained_date = args.trained_date
    num_options = args.num_options
    layer_idx = args.layer_idx

    
    global model_name_for_dirname
    if model_size in ['2', '9']:
        model_version = 2
    elif model_size in ['1', '4', '12']:
        model_version = 3
    else:
        raise ValueError(f"Invalid model_size: {model_size}")


    # [WIP] 'it'と'pt'のどちらが良いかは未検証. とりあえず'it'で統一.
    model_name = f"google/gemma-{model_version}-{model_size}b-it" # [memo] 'gemma-'部分は変えないこと!! -を消すとモデルがloadできない．さらにそのエラーメッセージは，"huggingface-cli login"をして，という関係ないmessageになるので注意!
    model_name_for_dirname = f"gemma-{model_version}-{model_size}B-lr{lr}-{trained_date}"
    if layer_idx is not None:
        model_name_for_dirname += f"-hidden_layer{layer_idx}"
    model_name_for_dirname += f"-seed{seed}"


    # ********* tokenizerとmodelの準備 *********
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = None # = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") # [memo] 初期モデルはepoch0の時, もしくはmodel未loadの際にそのepoch内で読み込むので，ここではNoneを読み込む
    
    # llama系はpad_tokenが設定されていないことがあるため，その場合はeos_tokenをpad_tokenに設定する
    need_to_set_pad_token = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        need_to_set_pad_token = True
    print(f"Loaded model and tokenizer: {model_name}")

    # **** 結果dirの準備 ****
    # *** 🟠 修正後(2026/01/18~) 引数でconfig/{target_concepts_filename}で指定されたconcept群を学習対象とするように変更 ***
    class_to_target_concepts_path = os.path.join(project_root, 'config', target_concepts_filename)
    if not os.path.exists(class_to_target_concepts_path) or target_concepts_filename.split('.')[-1] != 'json':
        raise ValueError(f"指定されたtarget_concepts_filename '{target_concepts_filename}' が存在しないか，jsonファイルではありません。configディレクトリ内の正しいjsonファイル名を指定してください。")
    with open(class_to_target_concepts_path, 'r') as f:
        class_to_target_concept_config = json.load(f)
    config_specified_concept_list = sum(class_to_target_concept_config.values(), [])
    result_dir = os.path.join(project_root, "results", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")
    mem_dir = os.path.join(project_root, "memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")


    # memvec用token_id割り当て読み込み
    assign_saved_path = os.path.join(mem_dir, 'token_assignment.json')
    with open(assign_saved_path, 'r') as f:
        concept2trainable_tk_map = json.load(f)
    MemTokenIds = [tokenizer.vocab[resSpeTk] for resSpeTk in concept2trainable_tk_map.values()]
    print(f"Loaded concept2trainable_tk_map from {assign_saved_path}")



    # ********* data load *********
    # ** test1 data のあるconcept名を取得する
    concept_with_test1_list = [filename.split('.json')[0].replace('_', ' ') for filename in os.listdir(test1_dir)]

    # *** target_concepts にテスト対象のconcept名を追加
    # ** テストデータが存在するconcept名のみを抽出. 
    valid_concept_names = concept_with_test1_list

    # ** config_specified_concept_listの指定に基づき，テスト対象conceptを絞り込む
    if config_specified_concept_list[0] in [None, 'None']:
        # * テスト対象conceptが個別に指定されていない場合: そのままconcept_namesの全てを対象とする
        target_concepts = valid_concept_names
        pass
    else:
        # * テスト対象conceptが個別に指定されている場合. (config/target_concepts.jsonで指定):
        target_concepts = []
        # config_specified_concept_listに含まれ，且つテストデータが存在したconceptのみを抽出
        for tc in config_specified_concept_list:
            if tc in valid_concept_names:
                target_concepts.append(tc)
    
    target_concepts = sorted(target_concepts) # concept_names をアルファベット順にsort
    print("Target concept list:", target_concepts, '\n')


    # ********* test1 data load *********
    # 全test1データを読み込む  [WIP] test1の格納方法は考え中. 
    concept_to_test1_lst = {}
    for target_concept in target_concepts:
        with open(os.path.join(test1_dir, f"{target_concept.replace(' ', '_')}.json"), 'r') as f:
            concept_to_test1_lst[target_concept] = json.load(f)
    print(f"Loaded test1 data for {len(concept_to_test1_lst)} concepts from {test1_dir}")


    # ** test1実施対象conceptについて, <target_token> -> 対応するtoken の書き換えを行う. またconcept2trainable_tk_mapに無いconceptは対象から除外する ** 
    concept_to_test1_lst_new = {}
    for concept, test1_lst in concept_to_test1_lst.items():
        if concept not in concept2trainable_tk_map:
            print(f"Concept {concept} not in concept2trainable_tk_map. Skipping this concept.")
            continue

        assigned_token = concept2trainable_tk_map[concept]

        test1_lst_new = []
        for test1_info in test1_lst:
            test1 = test1_info['test1'].replace('<target_token>', assigned_token)
            test1_info['test1'] = test1
            test1_lst_new.append(test1_info)
        concept_to_test1_lst_new[concept] = test1_lst_new
    concept_to_test1_lst = concept_to_test1_lst_new



    # ********* test1 *********
    epoch_list = []
    for filename in os.listdir(mem_dir):
        if filename.endswith('.pth.npy'):
            epoch_num = int(filename.split('.pth.npy')[0])
            epoch_list.append(epoch_num)
    epoch_list = sorted(epoch_list)

    # [WIP] 特例 3まで
    # epoch_list = epoch_list[:4]
    print(f"Found trained memvec files for epochs: {epoch_list}")


    for epoch in tqdm(epoch_list, desc="Evaluating test1 over epochs"):

        # *** 既にtest1結果が存在する場合はスキップ ***
        # このepochに対する, 対象の全conceptのtest1結果が存在する場合はスキップ
        skip_epoch = True
        for concept in concept_to_test1_lst.keys():
            logit_score_path = os.path.join(result_dir, concept, f"logit-scored_epoch{epoch}.json")
            if not os.path.exists(logit_score_path):
                skip_epoch = False
                break
        if skip_epoch:
            print(f"All test1 results for epoch {epoch} already exist. Skipping this epoch.")
            continue

        # ****** epoch毎にmodel読み込み・memvec挿入 ******
        print(f"epoch: {epoch}")
        if epoch == 0:
            # 初期モデル
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            if need_to_set_pad_token:
                model.config.pad_token_id = tokenizer.pad_token_id
            print("Loaded pre-trained model")
            
        else:
            if model is None:
                # epoch0がlistにない場合はmodelがまだ読み込まれていないので，ここで読み込む
                # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
                if need_to_set_pad_token:
                    model.config.pad_token_id = tokenizer.pad_token_id

            # ** memvecをmodelに挿入・置換 **
            try:
                mem_save_path = os.path.join(mem_dir, f'{epoch}.pth.npy')
                load_mem_vec(model, mem_save_path, MemTokenIds)
            except Exception as e:
                print(f"Error loading memvec for epoch {epoch} from {mem_save_path}: {e}")
                continue  # 学習済みembed層が保存されていなければ、このepochの評価はスキップ

            print(f"Loaded memvec for epoch {epoch} from {mem_save_path} & replaced model embeddings.")

        model.eval()



        # ****** test1 evaluation ******
        for concept, test1_lst in concept_to_test1_lst.items():

            first_token_logits_list = [] # 回答の最初のトークンのlogitsを保存するためのリスト
            label_lst = []  # 正解ラベルを保存するリスト
            test_id_lst = []

            for test1_info in test1_lst:
                test_id = test1_info['test_id']
                test_id_lst.append(test_id)
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

            for i, (test_id, log_prob_dict, prob_dict) in enumerate(zip(test_id_lst, log_prob_dicts, prob_dicts)):
                # 各選択肢のlog確率を取得
                pred_option = max(log_prob_dict, key=log_prob_dict.get)  # log確率(value)が最大となる選択肢番号(key)を取得
                results.append({
                    'idx': test_id,
                    # "question": prompts[i][len(prompt_base):],  # ベースプロンプト部分を除去して質問文のみを保存
                    "log_probs": log_prob_dict[pred_option],
                    "probs": prob_dict[pred_option],
                    "label": y_true_lst[i],
                    "pred": pred_option,
                    "all_log_probs": log_prob_dict,
                })
                y_pred_lst.append(str(pred_option))

            
            # logitを保存
            os.makedirs(os.path.join(result_dir, concept), exist_ok=True)
            with open(os.path.join(result_dir, concept, f"logit-scored_epoch{epoch}.json"), 'w') as f:
                json.dump(results, f, indent=4)

    # *** 全conceptのscoreを計算し，summaryとして保存 ***
    # test1実施済みのためskipしたepochも含めるため, 全epochの結果を読み直して集計し, 保存する
    concept_to_results = {}
    for concept in concept_to_test1_lst.keys():
        for epoch in epoch_list:
            # このconceptに対するepochの結果を読み込む
            logit_score_path = os.path.join(result_dir, concept, f"logit-scored_epoch{epoch}.json")
            if not os.path.exists(logit_score_path):
                continue
            with open(logit_score_path, 'r') as f:
                results = json.load(f)
            y_true_lst = [res['label'] for res in results]
            y_pred_lst = [res['pred'] for res in results]

            # score計算
            metrics = calculate_metrics(y_pred_lst, y_true_lst)

            concept_to_results.setdefault(concept, {})[epoch] = metrics

            print(f"  Epoch {epoch}: Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['F1']:.4f}")

    with open(os.path.join(result_dir, "logit_score_summary.json"), 'w') as f:
        json.dump(concept_to_results, f, indent=4)






# ********************* 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"') # *** 🟠 
    parser.add_argument('--model_size', type=str, default='0.6', help='モデルサイズ (例: 0.6, 1.7, 4, 8, 14)')
    # parser.add_argument("--trained_date", default="20251221")
    parser.add_argument('--lr', type=float, default=0.01, help='学習率')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    parser.add_argument('--init_vec_types', type=str, nargs='+', default=['zero', 'uniform', 'norm_rand'], help='memory vectorの初期化方法のリスト. ')
    parser.add_argument('--layer_indices', type=int, nargs='*', default=None, help='隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。init_vec_typeが \'category_centroid_by_hidden_state_mean\' の場合に使用')
    parser.add_argument('--thread_id', type=int, nargs='?', default=0, help='2process同時に実行する場合のthread id (0 or 1). これにより,実行する設定(seed, init_vec_typeの組)が被らないように調整する')
    parser.add_argument('--process_num', type=int, nargs='?', default=2, help='同時に実行するprocess数')
    parser.add_argument('--seed_num', type=int, nargs='?', default=10, help='シードの数. 例えば10に設定した場合、seed0からseed9までの10個のシードで学習を実行することになる。')
    parser.add_argument('--num_options', type=int, nargs='?', default=3, help='選択肢の数. 例えば3に設定した場合、選択肢1, 2, 3の生成確率を抽出することになる。')
    args = parser.parse_args()

    processNum = args.process_num  # 複数process同時に実行する場合のprocess数
    
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    
    task_id = -1
    for seed in range(args.seed_num):
        args.seed = seed

        # if seed in [0,1,2,3,4,5,6 ,8,9]:
        #     print(f"seed {seed} is already run. skip.")
        #     continue

        # 通常:
        # init_vec_type_lst = ['category_centroid_plus_random', 'other_category_COG', 'norm_rand_vocab', 'zero', 'uniform', 'norm_rand', 'category_COG', ]
        init_vec_type_lst = args.init_vec_types
        layer_indices = args.layer_indices



        # trained_date はseed毎に違う（seed内でも異なるものは手動で調整してある）
        if args.model_size=='4':
            # * seed前半
            if seed == 0:
                args.trained_date = "20260316" 
            elif seed == 1:
                args.trained_date = "20260316" 
            # elif seed == 2:
            else:
                raise ValueError(f"Invalid seed: {seed}")

        # [memo] 新しく隠れ層を参照する初期化手法を追加した場合は、initMethods_with_HS に追加 -> 不要になった
        elif args.model_size=='12':
            # * seed前半
            if seed == 0:
                args.trained_date = "20260415" 
                # print(f"seed {seed} is already run. skip.")
                # continue
            elif seed == 1:
                args.trained_date = "20260415" 
            elif seed == 2:
                args.trained_date = "20260415"
            elif seed == 3:
                args.trained_date = "20260415"
            elif seed == 4:
                args.trained_date = "20260415"
            # # * seed後半
            elif seed == 5:
                args.trained_date = "20260415"
            elif seed == 6:
                args.trained_date = "20260415"
            elif seed == 7:
                args.trained_date = "20260415"
            elif seed == 8:
                args.trained_date = "20260415"
            elif seed == 9:
                args.trained_date = "20260415"
            else:
                raise ValueError(f"Invalid seed: {seed}")
            
        
        elif args.model_size=='9':
            pass
        

        for init_vec_type in init_vec_type_lst:

            print(f"init_vec_type: {init_vec_type}, layer_indices: {layer_indices}")
            layer_indices = args.layer_indices
        

            # if len(layer_indices) < 1 or \
            #     init_vec_type not in initMethods_with_HS:
            if len(layer_indices) < 1:
                # layer_idxが不要の初期化方法の場合は、layer_indicesを[None]にして、1回だけループするようにする
                layer_indices = [None]

                
            for layer_idx in layer_indices:
                args.layer_idx = layer_idx
                
                print(f"\n\n=== Training with seed: {seed}, init_vec_type: {init_vec_type}, layer_idx: {layer_idx} ===")
                args.init_vec_type = str(init_vec_type)

                task_id += 1
                
                if int(task_id % processNum) != int(args.thread_id):
                    # 複数process同時に実行する場合, thread_idに応じてtask_idが偶数or奇数の設定のみを実行する
                    print(f"Skipping task_id {task_id} for thread_id {args.thread_id}")
                    continue

                fix_seed(seed)
                main(args)

                # GPUメモリ解放
                torch.cuda.empty_cache()
                # 3秒待機
                time.sleep(3)


