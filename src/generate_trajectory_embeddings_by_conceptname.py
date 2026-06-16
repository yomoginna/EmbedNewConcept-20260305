"""
学習過程のベクトルを生成するコード。
ただしベクトルは、新規概念の元になった既存概念名そのものをpromptとした際の隠れ状態から作成する。
(最もsimpleな方法)

経緯：
- 学習中の過程を、新規概念毎に追うために目標ベクトルが必要になった。
- 概念毎に、その学習ステップ時点でのベクトルから目標ベクトルまでの距離を可視化するのに使う。

方法:
- promptに新規概念の元になった既存概念名を埋め込むことで、ベクトルを作成する。

関係するコード:
- src/generate_goal_embeddings.py: 目標ベクトルを生成するコード。promptに新規概念の元になった既存概念名を埋め込むことで、ベクトルを作成する。
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
from utils.gemma_train_and_test_utils import get_gemma_model_version, set_tokenizer_and_model, construct_model_name_for_dirname
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
    init_vec_type = args.init_vec_type
    lr = args.lr
    trained_date = args.trained_date
    init_layer_index = args.init_layer_index
    seed = args.seed

    visualize_layer_index='all'
    model_version = get_gemma_model_version(model_size)

    need_layer_flag = 'HS' in init_vec_type or 'HiddenState' in init_vec_type   # 初期化方法名に'隠れ層'が含まれれば、layer_idxの指定が必要な初期化方法とみなす
    print("need layer flag:", need_layer_flag)

    if not need_layer_flag:
        # HSを初期vec作成に使わない場合は、layer_indexは指定されていないものとして扱う
        init_layer_index = None


    # [WIP] 'it'と'pt'のどちらが良いかは未検証.とりあえず'it'で統一.
    model_name = f"google/gemma-{model_version}-{model_size}b-it" # [memo] 'gemma-'部分は変えないこと!! -を消すとモデルがloadできない．さらにそのエラーメッセージは，"huggingface-cli login"をして，という関係ないmessageになるので注意!
    model_name_for_dirname = construct_model_name_for_dirname(
        model_size=model_size,
        lr=lr,
        trained_date=trained_date,
        layer_idx=init_layer_index,
        random_seed=seed,
    )

    # ** 保存先 **
    output_dir = os.path.join(
        "/work04/toko/EmbedNewConcept-20260305", 
        "trajectory_embeddings",    # [memo] change here
        "by_conceptname",           # [memo] change here depends on the method to create ves for visualization.
        f"gemma-{model_version}-{model_size}B",
        pool_hs_type
    )
    os.makedirs(output_dir, exist_ok=True)

    # ** memvec_modelsが保存されているディレクトリ **
    mem_dir = os.path.join(
        "/work04/toko/EmbedNewConcept-20260305/memvec_models", 
        f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}"
    )
    

    # =========================
    # data load
    # =========================
    # epoch_listを取得
    epoch_list = []
    for file_name in os.listdir(mem_dir):
        if file_name.endswith('.npy') and not file_name == 'best.npy':
            epoch_num_str = file_name.split('.')[0]  # '10.npy' -> '10'
            epoch_num = int(re.findall(r'\d+', epoch_num_str)[0])  # '10' -> 10
            epoch_list.append(epoch_num)
    epoch_list.sort()
    print(f"Found memvec files for epochs: {epoch_list}")


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # * 学習時に保存した、memvec用token_id割り当て読み込み *
    assign_saved_path = os.path.join(mem_dir, 'token_assignment.json')
    with open(assign_saved_path, 'r') as f:
        concept2trainable_tk_map = json.load(f)
    MemTokenIds = [tokenizer.vocab[resSpeTk] for resSpeTk in concept2trainable_tk_map.values()]
    print(f"Loaded concept2trainable_tk_map from {assign_saved_path}")


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


    # ** 各concept名に対応させた予約済み特殊トークン名を prompt とする
    prompts = [concept2trainable_tk_map.get(conceptname) for conceptname in config_concept_list]
    print(f"Prompts corresponding to concept names: {prompts}")

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
    # *** 全層の隠れ状態(vec)を全て記録に残す ***
    # =========================
    for epoch in epoch_list:
        print(f"epoch: {epoch}")

        # ****** epoch毎にmodel読み込み・memvec挿入 ******
        # if epoch == 0:
        #     # epoch0は未追加学習のモデル
        #     model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        #     print("Loaded pre-trained model")
            
        # else:
        #     # if model is None:
        #     #     # epoch0がlistにない場合はmodelがまだ読み込まれていないので，ここで読み込む
        #     #     # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
        #     #     model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        #     #     if need_to_set_pad_token:
        #     #         model.config.pad_token_id = tokenizer.pad_token_id

        # 毎回モデルを読み込み直す場合はこちら
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

        # ** memvecをmodelに挿入・置換 **
        try:
            mem_save_path = os.path.join(mem_dir, f'{epoch}.npy')   # .pth.npyになっている場合がある (20260427以前に保存したモデル)
            load_mem_vec(model, mem_save_path, MemTokenIds)
        except Exception as e:
            print(f"Error loading memvec for epoch {epoch} from {mem_save_path}: {e}")
            continue  # 学習済みembed層が保存されていなければ、このepochの評価はスキップ

        print(f"Loaded memvec for epoch {epoch} from {mem_save_path} & replaced model embeddings.")

        set_tokenizer_and_model(tokenizer, model)
        model.eval() # 評価モードに切り替え (これにより、dropoutなどの挙動が変わる)

        # *** pool_hs_type に応じて、vectorを抽出 ***
        all_vecs = extract_hidden_states(
            model, 
            tokenizer,
            prompts, 
            pool_hs_type,
            batch_size=8, 
            pool_hs_target_texts=None,  # pool_hs_type=='target_seq_mean_pool' や 'target_seq_last_token'のときに使う。各textの対象テキスト位置でmean_poolするためのテキストのリスト。text_listと同順で、各textのmean_poolの対象となるテキストが入っていることを想定。
            layer_index=visualize_layer_index,
            print_flag=False
        )   # -> (T, D) or (T, H, D) Tはテキスト数, Hは層の数, Dは隠れ状態の次元

        # ベクトルを保存, output_path名は、epochによって変える
        if not need_layer_flag:
            output_path = os.path.join(output_dir, f"{target_concepts_filename.split('.')[0]}_{trained_date}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_epoch{epoch}.npz")
        else:
            output_path = os.path.join(output_dir, f"{target_concepts_filename.split('.')[0]}_{trained_date}_initlayer{init_layer_index}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_epoch{epoch}.npz")
        
        np.savez(
            output_path, 
            vectors=all_vecs,
            target_concepts_filename=target_concepts_filename,
            concept_names=np.array(config_concept_list, dtype=str),
            prompts=np.array(prompts, dtype=str),
            model_size=model_size,
            pool_hs_type=pool_hs_type,
            layer_index=visualize_layer_index,
        )
        print(f"Saved goal embeddings to {output_path}")



    





if __name__ == "__main__":
    print("Starting the process to generate goal embeddings by mean pooling the hidden states of input 'concept names'...")
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"') # *** 🟠 
    parser.add_argument("--model_size", type=int, default=4, help="Size of the Gemma model in billions (e.g., 4 for Gemma-3-4B).")
    parser.add_argument("--pool_hs_type", type=str, default="repeat_mean_pool", help='hidden stateのpooling方法. "eos": 最後のEOSトークンの隠れ状態を使用. "last_token": 最後のトークンの隠れ状態を使用. "mean_pool": テキスト全体の隠れ状態の平均を使用. "repeat_mean_pool": テキストを2回繰り返した内の後のtextの隠れ状態の平均を使用. "dot": テキスト全体の隠れ状態を平均したものと、最後のトークンの隠れ状態を連結して使用.')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    
    parser.add_argument('--init_vec_type_list', type=str, nargs='+', default=["CatCent_by_WikiSummaryRepeatHSMixed"], help='目memory vectorの初期化方法のリスト. 例: "CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab"')
    parser.add_argument('--lr', type=float, default=0.003, help='学習率. 例: 3e-3')
    parser.add_argument('--trained_date', type=str, default="", help='学習した日付. 例: "20260427"')
    parser.add_argument('--init_layer_index', type=int, default=12, help='学習時に訓練対象token_vecの初期vecとして使用した層のインデックス. 例: 12')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード. 例: 42')

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    
    for init_vec_type in args.init_vec_type_list:
        print(f"Generating trajectory embeddings with init_vec_type: {init_vec_type}")
        args.init_vec_type = init_vec_type
        main(args)



"""
TARGET_CONCEPTS_FILENAME="target_concepts_mini_13.json"
MODEL_SIZE=12
CUDA_VISIBLE_DEVICES=4

LR=0.003
NUM_OPTIONS=3
INIT_LAYER_INDEX=12
TRAINED_DATE="20260529" # "20260523" #"20260427"
SEED=0
POOL_HS_TYPE="mean_pool" # "target_seq_repeat_mean_pool" #"eos" # "repeat_mean_pool"


INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "farCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab") 
# INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab") 
INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed")
INIT_VEC_TYPE_LIST=("nearCatCent_by_WikiSummaryRepeatHSMixed" "farCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab") 
INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "norm_rand_vocab") 


nohup uv run python src/generate_trajectory_embeddings_by_conceptname.py \
    --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
    --model_size ${MODEL_SIZE} \
    --pool_hs_type ${POOL_HS_TYPE} \
    --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
    --init_vec_type_list ${INIT_VEC_TYPE_LIST} \
    --lr ${LR} \
    --trained_date ${TRAINED_DATE} \
    --init_layer_index ${INIT_LAYER_INDEX} \
    --seed ${SEED} \
    > log_generate_trajectory_embeddings_gemma-${MODEL_SIZE}B_${TARGET_CONCEPTS_FILENAME}.log 2>&1 &

    
3393797

"""