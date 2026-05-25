"""
目標のtoken vecを生成するコード。
ただしベクトルは、新規概念の元になった既存概念名そのものをpromptとした際の隠れ状態から作成する。
(最もsimpleな方法)

経緯：
- 学習中の過程を、新規概念毎に追うために目標ベクトルが必要になった。
- 概念毎に、その学習ステップ時点でのベクトルから目標ベクトルまでの距離を可視化するのに使う。

方法:
- promptに新規概念の元になった既存概念名を埋め込むことで、ベクトルを作成する。

関係するコード:
- src/generate_goal_embeddings*.py: 目標ベクトルを生成するコード。promptに新規概念の元になった既存概念名を埋め込むことで、ベクトルを作成する。
- src_visualize/plot_vecs_3dPCA.py: 生成したベクトを3次元PCAでプロットするコード。概念毎に色分けして、ホバーで概念名と層のインデックスを表示する。

注意点：
- 既存概念名だけでそのまま生成すると、例えば"Unlock!"のように意味と表層的な単語の意味がマッチしない場合に、正しい目標ベクトルが得られない。
    - そのため、promptに埋め込んで、どの"unlock!"を指しているのかを限定するための工夫が必要。

実行時間:
- 割とすぐ終わる。(4Bの場合は3分程度, 12Bでもそんなに変わらない。zao00でも12Bを実施できた。prompt短いし学習ないから。)

実行後:
- このコードを実行すると、目標vecのnpzファイルがwork04に保存される。
- これを可視化するために、src_visualize/plot_vecs_3dPCA.pyを実行する。
    - 3次元PCA, 概念毎の色分け

"""
# ===== Standard library =====
import argparse
import json
import os
import re
import sys

# ===== Third-party =====
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== Runtime config =====

project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
# project_root = os.environ["HOME"] # [memo] genkaiを使う場合. "/singularity_home/project/EmbedNewConcept/src/trainMemVec_fromXvec_gemma.py"
sys.path.append(project_root)
print("Project root:", project_root)

from utils.embedding_utils import load_mem_vec
from utils.gemma_train_and_test_utils import get_gemma_model_version, set_tokenizer_and_model
from utils.embedding_utils import extract_hidden_states

global BATCH_SIZE

wiki_page_save_dir = os.path.join(project_root, 'data', 'wiki_pages')
dont_get_new_wiki_flag = False # False #True # もう新しいwikiページを読み込みたくない場合はTrue. すでに保存済みのwikiページがあるpropernounのみにフィルタリングする.
print_flag = False

debug_print_flag = False



# *************************************************************** main ***************************************************************
def main(args):
    model_size = args.model_size
    target_concepts_filename = args.target_concepts_filename
    pool_hs_type = args.pool_hs_type

    # visualize_layer_index='all'   # [memo] 大して変わらないので、毎回全層のベクトルを保存するようにした。そのため必ず'all'を各関数で直接指定するよう変更した
    model_version = get_gemma_model_version(model_size)

    # [WIP] 'it'と'pt'のどちらが良いかは未検証.とりあえず'it'で統一.
    model_name = f"google/gemma-{model_version}-{model_size}b-it" # [memo] 'gemma-'部分は変えないこと!! -を消すとモデルがloadできない．さらにそのエラーメッセージは，"huggingface-cli login"をして，という関係ないmessageになるので注意!
    
    # ** 保存先 **
    output_dir = os.path.join("/work04/toko/EmbedNewConcept-20260305", "goal_embeddings", "by_conceptname", f"gemma-{model_version}-{model_size}B")
    output_dir = os.path.join(output_dir, pool_hs_type)
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # data load
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # * config/{target_concepts_filename}で指定されたconcept群を学習対象とする *
    class_to_target_concepts_path = os.path.join(project_root, 'config', target_concepts_filename)
    if not os.path.exists(class_to_target_concepts_path) or target_concepts_filename.split('.')[-1] != 'json':
        raise ValueError(f"指定されたtarget_concepts_filename '{target_concepts_filename}' が存在しないか，jsonファイルではありません。configディレクトリ内の正しいjsonファイル名を指定してください。")
    with open(class_to_target_concepts_path, 'r') as f:
        class_to_target_concepts_config = json.load(f)
    config_concept_list = sum(class_to_target_concepts_config.values(), [])
    if len(config_concept_list) == 0:
        raise ValueError("No concepts found in target concept config.")
    print(f"Target concepts specified in config {class_to_target_concepts_path}: {config_concept_list}")


    # =========================
    # ** モデル読み込み **
    # =========================
    print("Loading model and tokenizer...")
    model_version = get_gemma_model_version(model_size)

    # [WIP] 'it'と'pt'のどちらが良いかは未検証.とりあえず'it'で統一.
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    set_tokenizer_and_model(tokenizer, model)


    # * デバッグ: tokenizerの動作確認. text_listの各テキストがどのようにtokenizeされるか、またdecodeするとどうなるかを確認する. これにより、tokenizerが想定通りに動いているか、特にEOSトークンの扱いがどうなっているかを確認できる.
    dot_ids = tokenizer.encode(".", add_special_tokens=False)
    print(f"dot token ids: {dot_ids}, tokens: {tokenizer.convert_ids_to_tokens(dot_ids)}")
    for concept_name in config_concept_list:
        encoded = tokenizer(concept_name, return_tensors="pt", add_special_tokens=False)
        print(f"Tokenized the text of concept '{concept_name}': {encoded}")
        decoded = tokenizer.decode(encoded["input_ids"][0])
        print(f"Decoded back: {decoded}\n")


    # =========================
    # *** 全層の隠れ状態を全て記録に残す．最終dot(.)位置のみのベクトル，全体のmean pool, の2種類を記録する．***
    # =========================
    model.eval() # 評価モードに切り替え (これにより、dropoutなどの挙動が変わる)

    # *** pool_hs_type に応じて、vectorを抽出 ***
    all_vecs = extract_hidden_states(
        model, 
        tokenizer,
        config_concept_list, 
        pool_hs_type,
        batch_size=8, 
        pool_hs_target_texts=None, # pool_hs_type=='target_seq_mean_pool' や 'target_seq_last_token'のときに使う。各textの対象テキスト位置でmean_poolするためのテキストのリスト。text_listと同順で、各textのmean_poolの対象となるテキストが入っていることを想定。
        layer_index='all', # visualize_layer_index,
        print_flag=False
    )   # -> (T, D) or (T, H, D) Tはテキスト数, Hは層の数, Dは隠れ状態の次元

    # ベクトルを保存
    # output_path = os.path.join(output_dir, f"goal_embeddings_{model_size}B_{target_concepts_filename.split('.')[0]}_{pool_hs_type}_layer{layer_index}")
    # output_path = os.path.join(output_dir, f"{target_concepts_filename.split('.')[0]}_layer{visualize_layer_index}.npz")
    output_path = os.path.join(output_dir, f"{target_concepts_filename.split('.')[0]}_layerall.npz")
    np.savez(
        output_path, 
        vectors=all_vecs,
        target_concepts_filename=target_concepts_filename,
        concept_names=np.array(config_concept_list, dtype=str),
        text_list=np.array(config_concept_list, dtype=str),
        model_size=model_size,
        pool_hs_type=pool_hs_type,
        layer_index='all', # visualize_layer_index,
    )
    print(f"Saved goal embeddings to {output_path}")



    





if __name__ == "__main__":
    print("Starting the process to generate goal embeddings by mean pooling the hidden states of input 'concept names'...")
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"') # *** 🟠 
    parser.add_argument("--model_size", type=int, default=4, help="Size of the Gemma model in billions (e.g., 4 for Gemma-3-4B).")
    parser.add_argument("--pool_hs_type", type=str, default="repeat_mean_pool", help='hidden stateのpooling方法. "eos": 最後のEOSトークンの隠れ状態を使用. "last_token": 最後のトークンの隠れ状態を使用. "mean_pool": テキスト全体の隠れ状態の平均を使用. "repeat_mean_pool": テキストを2回繰り返した内の後のtextの隠れ状態の平均を使用. "dot": テキスト全体の隠れ状態を平均したものと、最後のトークンの隠れ状態を連結して使用.')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    main(args)


"""
TARGET_CONCEPTS_FILENAME="target_concepts_mini_13.json"
MODEL_SIZE=12
POOL_HS_TYPE="mean_pool"   # "eos"  # repeat_mean_pool


nohup uv run python src/generate_goal_embeddings_by_conceptname.py \
    --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
    --model_size ${MODEL_SIZE} \
    --pool_hs_type ${POOL_HS_TYPE} \
    --cuda_visible_devices 3 \
    > log_generate_goal_embeddings_gemma-${MODEL_SIZE}B_${TARGET_CONCEPTS_FILENAME}.log 2>&1 &

3748904

"""