"""
学習過程のベクトルを生成するコード。
ただしベクトルは、新規概念の元になった既存概念名をwiki summaryの説明文に埋め込んだpromptを入力したときの隠れ状態から作成する。

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
from utils.embedding_utils import extract_hidden_states, get_concept_containing_text_using_wiki_summary

global BATCH_SIZE

wiki_page_save_dir = os.path.join(project_root, 'data', 'wiki_pages')
dont_get_new_wiki_flag = False # False #True # もう新しいwikiページを読み込みたくない場合はTrue. すでに保存済みのwikiページがあるpropernounのみにフィルタリングする.
print_flag = False

debug_print_flag = False

# def construct_model_name_for_dirname(model_size, lr, trained_date, layer_idx, random_seed):
#     model_version = get_gemma_model_version(model_size)

#     model_name_for_dirname = f"gemma-{model_version}-{model_size}B-lr{lr}-{trained_date}"
#     if layer_idx is not None:
#         print(f"Using layer index: {layer_idx}")
#         model_name_for_dirname += f"-hidden_layer{layer_idx}"
#     model_name_for_dirname += f"-seed{random_seed}"

#     return model_name_for_dirname




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
        "trajectory_embeddings",                # ⭐️
        "by_embed_conceptname_in_wikisummary",  # ⭐️
        f"gemma-{model_version}-{model_size}B",
        pool_hs_type
    )
    # output_dir = os.path.join(output_dir, pool_hs_type)
    os.makedirs(output_dir, exist_ok=True)

    # ** memvec_modelsが保存されているディレクトリ **
    mem_dir = os.path.join("/work04/toko/EmbedNewConcept-20260305/memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")



    # =========================
    # data load
    # =========================
    # epoch_listを取得
    epoch_list = []
    for file_name in os.listdir(mem_dir):
        if file_name.endswith('.npy'):
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


    # * 該当wiki pageのsummaryから、概念をうまく説明する1文を抽出し、概念名を置換したければ置換する. *
    concept_to_one_summary_sentence = {}
    for concept in config_concept_list:
        first_concept_sentence = get_concept_containing_text_using_wiki_summary(
            concept, 
            "<target_concept_name>", # [memo] src/generate_goal_embeddings.py と異なる点
        )
        if first_concept_sentence is None:
            # 現在のget_concept_containing_text_using_wiki_summary()では、概念を説明する1文がwiki summaryから見つけられなかったためskip.
            continue
        concept_to_one_summary_sentence[concept] = first_concept_sentence
        print(f"First wiki sentence containing concept '{concept}': \n\t{first_concept_sentence}\n")

    if len(concept_to_one_summary_sentence) == 0:
        raise ValueError("No concept embedding texts could be extracted.")

    # 何割について文を抽出できたかを確認
    extracted_count = len(concept_to_one_summary_sentence)
    total_count = len(config_concept_list)
    print(
        f"Extracted sentences for {extracted_count}/{total_count} concepts "
        f"({extracted_count / total_count * 100:.2f}%)"
    )

    # ** 各promptの "<target_concept_name>" -> 対応するtoken に書き換える. またconcept2trainable_tk_mapに無いconceptは対象から除外する ** 
    concept_to_one_summary_sentence_new = {}
    for concept, one_summary_sentence in concept_to_one_summary_sentence.items():
        if concept not in concept2trainable_tk_map:
            print(f"Concept {concept} not in concept2trainable_tk_map. Skipping this concept.")
            continue
        assigned_token = concept2trainable_tk_map[concept]
        concept_to_one_summary_sentence_new[concept] = one_summary_sentence.replace("<target_concept_name>", assigned_token)
    concept_to_one_summary_sentence = concept_to_one_summary_sentence_new


    # * デバッグ: tokenizerの動作確認. text_listの各テキストがどのようにtokenizeされるか、またdecodeするとどうなるかを確認する. これにより、tokenizerが想定通りに動いているか、特にEOSトークンの扱いがどうなっているかを確認できる.
    if debug_print_flag:
        dot_ids = tokenizer.encode(".", add_special_tokens=False)
        print(f"dot token ids: {dot_ids}, tokens: {tokenizer.convert_ids_to_tokens(dot_ids)}")
        for concept, text in concept_to_one_summary_sentence.items():
            assigned_token = concept2trainable_tk_map[concept]
            print(f"Concept: {concept} -> token: {assigned_token}, token_id: {tokenizer.convert_tokens_to_ids(assigned_token)}")
            encoded = tokenizer(text, return_tensors="pt")
            print(f"Tokenized the text of concept '{concept}': {encoded}")
            decoded = tokenizer.decode(encoded["input_ids"][0])
            print(f"Decoded back: {decoded}\n")

    # return 0 # [memo] ここまで動作確認済み


    # =========================
    # promptの作成 (目標vec生成用) 
    # =========================
    # prompt_base = "It is the <target_concept>."
    concept_to_prompt = {}
    for concept, one_summary_sentence in concept_to_one_summary_sentence.items():   # in config_concept_list:
        print(f"Processing concept: {concept}")
        if "repeat" in pool_hs_type:
            # 目標ベクトルを生成するためのpromptを作成. 例えば、"It is the apple. It is the apple." のように、同じ文を2回繰り返すことで、gemmaのattentionが、後半の文の方に向くようにする。
            prompt = one_summary_sentence + " " + one_summary_sentence 
            pool_hs_type = "mean_pool" # pool_hs_typeがrepeat_mean_poolの場合は、pool_hs_typeをmean_poolに変更して、後半の文の隠れ状態の平均を目標ベクトルとする. これにより、gemmaのattentionが、後半の文の方に向くようにする。
        else:
            prompt = one_summary_sentence
        print(f"Prompt for concept '{concept}': {prompt}")
        concept_to_prompt[concept] = prompt

    # config_concept_list の順番に対応するpromptのリストを作成
    concept_names, prompts, concept_unused_tk_names = [], [], []
    # prompts = [concept_to_prompt[concept] for concept in config_concept_list if concept in concept_to_prompt]
    for concept in config_concept_list:
        if concept in concept_to_prompt:
            concept_names.append(concept)
            prompts.append(concept_to_prompt[concept])
            concept_unused_tk_names.append(concept2trainable_tk_map[concept])
        else:
            # print(f"Warning: No prompt could be created for concept '{concept}' because no suitable sentence was found in the wiki summary. This concept will be skipped.")
            pass


    # return 0  # [memo] ここまで動作確認済み


    # * デバッグ: tokenizerの動作確認. text_listの各テキストがどのようにtokenizeされるか、またdecodeするとどうなるかを確認する. これにより、tokenizerが想定通りに動いているか、特にEOSトークンの扱いがどうなっているかを確認できる.
    if debug_print_flag:
        print(f"dot token id: {tokenizer.convert_tokens_to_ids('.')}")
        for concept, prompt in zip(concept_names, prompts):
            encoded = tokenizer(prompt, return_tensors="pt")
            print(f"Tokenized the prompt of concept '{concept}': {encoded}")
            decoded = tokenizer.decode(encoded["input_ids"][0])
            print(f"Decoded back: {decoded}\n")

    # return 0 # [memo] ここまで動作確認済み


    # =========================
    # 全層の隠れ状態(vec)を全て記録に残す
    # =========================
    for epoch in epoch_list:
        print(f"epoch: {epoch}")

        # ****** epoch毎にmodel読み込み・memvec挿入 ******
        if epoch == 0:
            # epoch0は未追加学習のモデル
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            print("Loaded pre-trained model")
            
        else:
            # if model is None:
            #     # epoch0がlistにない場合はmodelがまだ読み込まれていないので，ここで読み込む
            #     # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
            #     model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
            #     if need_to_set_pad_token:
            #         model.config.pad_token_id = tokenizer.pad_token_id

            # 毎回モデルを読み込み直す場合はこちら
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

            # ** memvecをmodelに挿入・置換 **
            try:
                mem_save_path = os.path.join(mem_dir, f'{epoch}.npy')
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
            mean_pool_target_texts=concept_unused_tk_names, # if pool_hs_type=="target_seq_mean_pool" else None,   # pool_hs_type=='target_seq_mean_pool'のとき、各textの対象unused_tk位置でmean_poolするためのテキストのリスト。text_listと同順で、各textのmean_poolの対象となるテキストが入っていることを想定。
            layer_index=visualize_layer_index,
            print_flag=False
        )   # -> (T, D) or (T, H, D) Tはテキスト数, Hは層の数, Dは隠れ状態の次元
        print(f"shape of all_vecs for epoch {epoch}: {all_vecs.shape}")

        # ベクトルを保存, output_path名は、epochによって変える
        # output_path = os.path.join(output_dir, f"trajectory_embeddings_{model_size}B_{target_concepts_filename.split('.')[0]}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_epoch{epoch}")
        if not need_layer_flag:
            output_path = os.path.join(output_dir, f"{target_concepts_filename.split('.')[0]}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_epoch{epoch}.npz")
        else:
            output_path = os.path.join(output_dir, f"{target_concepts_filename.split('.')[0]}_initlayer{init_layer_index}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_epoch{epoch}.npz")
        np.savez(
            output_path, 
            vectors=all_vecs,
            target_concepts_filename=target_concepts_filename,
            concept_names=concept_names,
            prompts=prompts,
            model_size=model_size,
            pool_hs_type=args.pool_hs_type, # repeat_の場合、途中でmean_poolに変えてしまったため、pool_hs_typeではなく、元のargs.pool_hs_typeを保存する
            layer_index=visualize_layer_index,
        )
        print(f"Saved trajectory embeddings to {output_path}")




    

        



if __name__ == "__main__":
    print("Starting the process to generate trajectory embeddings by embedding concept names in wiki summaries...")
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"') # *** 🟠 
    parser.add_argument("--model_size", type=int, default=3, help="Size of the Gemma model in billions (e.g., 3 for Gemma-3B).")
    parser.add_argument("--pool_hs_type", type=str, default="repeat_mean_pool", help='hidden stateのpooling方法. "eos": 最後のEOSトークンの隠れ状態を使用. "last_token": 最後のトークンの隠れ状態を使用. "mean_pool": テキスト全体の隠れ状態の平均を使用. "repeat_mean_pool": テキストを2回繰り返した内の後のtextの隠れ状態の平均を使用. "dot": テキスト全体の隠れ状態を平均したものと、最後のトークンの隠れ状態を連結して使用.')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    
    # parser.add_argument('--init_vec_type', type=str, default="CatCent_by_WikiSummaryRepeatHSMixed", help='目memory vectorの初期化方法')
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
TRAINED_DATE="20260427"
SEED=0
POOL_HS_TYPE="target_seq_repeat_mean_pool" #"eos" # "repeat_mean_pool"

INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab") 
INIT_VEC_TYPE_LIST=("nearCatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab") 


nohup uv run python src/generate_trajectory_embeddings_by_embed_conceptname_in_wikisummary.py \
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

3748904

"""