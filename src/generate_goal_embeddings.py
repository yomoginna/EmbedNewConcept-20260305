"""
目標点となるベクトルを生成するコード。
経緯：
- 学習中の過程を、新規概念毎に追うために目標ベクトルが必要になった。
- 概念毎に、その学習ステップ時点でのベクトルから目標ベクトルまでの距離を可視化するのに使う。

方法:
- promptに新規概念の元になった既存概念名を埋め込むことで、ベクトルを作成する。

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
from collections import defaultdict
from datetime import datetime
import json
import os
import random
import re
import sys
import time
import math

# ===== Third-party =====
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
import wandb


# ===== Runtime config =====
transformers_logging.set_verbosity_error()

project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
# project_root = os.environ["HOME"] # [memo] genkaiを使う場合. "/singularity_home/project/EmbedNewConcept/src/trainMemVec_fromXvec_gemma.py"
sys.path.append(project_root)
print("Project root:", project_root)

from utils.wikipedia_api_utils import load_wiki_text
from utils.gemma_train_and_test_utils import fix_seed, get_gemma_model_version, save_mem_vec, constructTrainSamples, encodeTrainSamplesWithTokenizer, evaluateModel, extract_hidden_states #, train
from utils.handle_data_from_dbpedia_utils import filterProperNounsWithWikiPage, loadProperNounData #, loadConceptsForFictConcept
from utils.initialize_embedding_layer_utils import EmbedInitializer
from utils.handle_text_utils import delete_non_English_characters, split_text_into_sentences

# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # [memo] genkaiを使う時はコメントアウト!! -> 今はargsで指定している。argsを指定しなければ、CUDAについては何も指定しない。
n_feat_in_a_sample = 3  # 学習データの1サンプル = summary(wiki中の本文 or summary, 今回はsummaryを使用) + n_feat_in_a_sample個の特徴文
propnoun_num_for_init_vec=100   #  初期化vecの作成に使う固有名詞の最低数. 例えば100に設定した場合、各カテゴリで最低100個の固有名詞を使用して初期化vecを作成することになる。(実際には、新規概念用にならなかった固有名詞全て使用する)
propnoun_num_for_new_concept = 50 # 新規概念の元にする概念の作成に使う固有名詞の数. 例えば50に設定した場合、各カテゴリで50個の固有名詞を使用して新規概念の元にする概念の作成に使用することになる。
min_words, max_words = 30, 300 # 30->50に変更すると、そこまで長いsummaryが少ないようで、init vecが0vecとなりlossがNanになってしまった。minは30でキープする

global BATCH_SIZE

wiki_page_save_dir = os.path.join(project_root, 'data', 'wiki_pages')
dont_get_new_wiki_flag = False # False #True # もう新しいwikiページを読み込みたくない場合はTrue. すでに保存済みのwikiページがあるpropernounのみにフィルタリングする.
print_flag = False

# 環境変数読み込み
load_dotenv(os.path.join(project_root, ".env"))
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


debug_without_model = False #True

# conceptを埋め込む文を取得する関数
def get_concept_embedding_text(concept, target_name_to_replace_with_concept=None):
    """conceptを埋め込む文を取得する関数. 
    大抵の wiki page は、概念名とその説明を含む1文から始まっているため、冒頭の文に概念名が含まれればその文を抽出する。含まれなければ、" is the " を含む文を探し、その前の部分を概念名に置換する。
    例:
    - conceptが"Apple Inc."であれば、Wikipediaのsummaryの中から"Apple Inc."を含む文を抽出し、その中で「最初の文」を返す. 
    - もし"Apple Inc."を含む文が見つからない場合は、" is the "を含む文を探し、その前の部分を"Apple Inc."に置換して返す. それも見つからない場合は、summary全体を返す.

    Args:
        concept (str): 埋め込みたい概念名
        target_name_to_replace_with_concept (str, optional): concept名に置換する対象名. 例えば，concept="banana", wiki first sent = "banana is yellow" の場合，この最初のbananaを何に置き換えるか．noneの場合は置換なし，none以外の場合は置換する．

    Returns:
        str: conceptを埋め込む文
    """
    special_case_dic = {
        "&": "and", # wikiタイトルでは 'Adam & Eve' だが、wiki page内文章では 'Adam and Eve' と表現されていた
        "and": "&",

    }
    

    # 辞書にまだ保存されていなければ、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに格納する
    summary = load_wiki_text(concept, text_type="summary")

    # ** concept名が含まれる文だけを抽出し，さらにその中の一番最初の文を収集する．大文字小文字の違いは無視し，記号([]',.')なども無視して比較する．例えば、conceptが"Apple Inc."であれば、summaryの中から"Apple Inc."を含む文を抽出し、その中で最初の文を収集する．このとき、"apple inc"や"Apple, Inc."などもconcept名とみなす．
    base_concept_pattern = re.escape(concept)                            # 概念名が含まれるかどうかを調べるパターン。concept名を正規表現の特殊文字をエスケープしてパターン化
    brackets_dup_pattern = re.compile(r'\s*\(.*?\[.*?\].*?\)\s*')   # ( ... [..] ... ) のような()内に[]が入るパターン. wiki title(data名) と wiki page文章内の綴りが異なっても、( [])内に読み方が含まれれば、その前部分が概念名であるとわかる。例: concept名: 'House of Hohenstaufen' wiki文章: "The Hohenstaufen dynasty (, US also , German: [ˌhoːənˈʃtaʊfn̩]), also known as the Staufer,  ..."
    is_the_pattern = re.compile(r'(.+?)\s+is\s+the\s+', re.IGNORECASE)      # " is the " が含まれるかどうかを調べるパターン
    brackets_pattern = re.compile(r'\s*\(.*?\)\s*')                 # (  AB-ə [ˈâbːa]) のような読み方が書かれることがある. 
    brackets_colon_pattern = re.compile(r"\([^()]*[:;][^()]*\)") # (en: Apple Inc.) のような、()内に:が入るパターン. "\([^()]*[:;][^()]*\)"
    brackets_stylised_pattern = re.compile(r'\s*\(.*?stylised.*?\)\s*') # (stylised as ABC) のようなパターン. これも概念名の一部ではないので削除する.


    # concept pattern に含まれる一部の文字が、wiki page内では他の文字で表されている場合があるため、その場合のpatternも作成しておく。
    concept_patterns = [base_concept_pattern]
    for k, v in special_case_dic.items():
        if k in concept:
            concept_patterns.append(concept.replace(k, v))


    sentences = split_text_into_sentences(summary)  # 文の区切りでsummaryを分割
    first_concept_sentence = None

    ## ① ベースとして、概念名orその置換パターンを含む文を探す
    # concept_sentences = [s for s in sentences if re.search(concept_pattern, s, re.IGNORECASE)]  # concept名を含む文を抽出
    concept_sentences = [s for s in sentences for concept_pattern in concept_patterns if re.search(concept_pattern, s, re.IGNORECASE)]  # concept名を含む文を抽出
    if concept_sentences:
        first_concept_sentence = concept_sentences[0]
        # もし, 基本の概念名ではなく辞書を元に置換した概念名でマッチした場合もある。それを基本の概念名に戻す。
        for concept_pattern in concept_patterns:
            if re.search(concept_pattern, first_concept_sentence, re.IGNORECASE):
                first_concept_sentence = re.sub(concept_pattern, concept, first_concept_sentence, flags=re.IGNORECASE)

    # ② 概念名を含む文が見つからない場合 → ( []) を含む文を探す。概念名が違う言語で書かれている場合などがある。この場合、他言語の概念名 (名前 [発音]) の表記であることがあるため、そのパターンを探し、冒頭から()までを概念名に置換する
    if first_concept_sentence is None:
        brackets_dup_sentences = [s for s in sentences if brackets_dup_pattern.search(s)]
        if brackets_dup_sentences:
            first_brackets_dup_sentence = brackets_dup_sentences[0]
            # その文の中の、brackets_dup_patternにマッチする部分とそれ以前を概念名に置換する
            modified_sentence = re.sub(rf"^.*?{brackets_dup_pattern.pattern}", concept, first_brackets_dup_sentence)           
            first_concept_sentence = modified_sentence
            print(f"\t No sentence containing concept '{concept}' was found. Using modified '( ... [..] ... )' sentence: \n\t\t{first_concept_sentence}\n")

    # ③ 概念名を含む文が見つからない場合 → " is the "を含む文を探して概念名を含む文を作る．
    # 例えば 'Arnhem Bridge' のsummaryの1文目は "John Frost Bridge (John Frostbrug in Dutch) is the road bridge over the Lower Rhine at Arnhem, in the Netherlands." である．この is the の前の部分を，concept名に置換して使う．
    if first_concept_sentence is None:
        # print(f"No sentence containing concept '{concept}' was found in the summary. Trying to find a sentence containing 'is the' to use as a template for the concept name.\n")
        is_the_sentences = [s for s in sentences if is_the_pattern.search(s)]
        if is_the_sentences:
            first_is_the_sentence = is_the_sentences[0]
            # first_is_the_sentenceのうち，「is the とマッチした部分」を，「concept + " is the "」に置換する
            modified_sentence = is_the_pattern.sub(concept + " is the ", first_is_the_sentence)
            first_concept_sentence = modified_sentence
            print(f"\t No sentence containing concept '{concept}' was found. Using modified 'is the' sentence: \n\t\t{first_concept_sentence}\n")

    if first_concept_sentence is None:   # 以上で解決できなければ、raise error. or Noneを返す
        # raise ValueError(f"No sentence containing concept '{concept}' was found in the summary, and no sentence containing 'is the' was found either. Please check the summary for concept '{concept}': \n {summary}")
        return None

    # [memo] この時点で必ず，concept名 を含む文が1つ，first_concept_sentenceとして格納されていることになる．

    # ** 置換先の名前が登録されている場合は、first_concept_sentenceのうち、concept_patternにマッチした部分をtarget_name_to_replace_with_conceptに置換する処理
    if target_name_to_replace_with_concept is not None:
        first_concept_sentence = re.sub(concept, target_name_to_replace_with_concept, first_concept_sentence, flags=re.IGNORECASE)
    
    # ** 文をcleaning
    first_concept_sentence = re.sub(brackets_dup_pattern, " ", first_concept_sentence)      # ( ... [..] ... ) パターンをスペースに置換
    first_concept_sentence = re.sub(brackets_colon_pattern, " ", first_concept_sentence)    # ( ... : ... ) パターンをスペースに置換
    first_concept_sentence = re.sub(brackets_stylised_pattern, " ", first_concept_sentence) # (stylised ...) パターンをスペースに置換
    first_concept_sentence = re.sub(r'\(\)', ' ', first_concept_sentence)                   # ()のみのパターンをスペースに置換
    first_concept_sentence = re.sub(r" {2,}", " ", first_concept_sentence)                  # 連続するスペースを1つのスペースに置換
    first_concept_sentence = re.sub(r"\s+([,.?!])", r"\1", first_concept_sentence)          # スペース + 句読点のパターンを、スペースなしの句読点に置換 (例. "The Ayyubid dynasty , also known as ..." -> "The Ayyubid dynasty, also known as ...")
    first_concept_sentence = first_concept_sentence.strip()                                 # 先頭と末尾
        
    return first_concept_sentence
    



# *************************************************************** main ***************************************************************
def main(args):
    model_size = args.model_size
    target_concepts_filename = args.target_concepts_filename
    pool_hs_type = args.pool_hs_type
    
    layer_index='all'
    model_version = get_gemma_model_version(model_size)
    
    # ** 保存先 **
    # output_dir = os.path.join(project_root, "output", f"gemma-{model_version}-{model_size}B_lr{lr}_{target_concepts_filename.split('.')[-1]}_{pool_hs_type}_layer{layer_idx}")
    output_dir = os.path.join("/work04/toko/EmbedNewConcept-20260305", "goal_embeddings", f"gemma-{model_version}-{model_size}B")
    os.makedirs(output_dir, exist_ok=True)


    # =========================
    # data load
    # =========================
    # *** config/{target_concepts_filename}で指定されたconcept群を学習対象とする ***
    class_to_target_concepts_path = os.path.join(project_root, 'config', target_concepts_filename)
    if not os.path.exists(class_to_target_concepts_path) or target_concepts_filename.split('.')[-1] != 'json':
        raise ValueError(f"指定されたtarget_concepts_filename '{target_concepts_filename}' が存在しないか，jsonファイルではありません。configディレクトリ内の正しいjsonファイル名を指定してください。")
    with open(class_to_target_concepts_path, 'r') as f:
        class_to_target_concepts_config = json.load(f)
    config_concept_list = sum(class_to_target_concepts_config.values(), [])
    print(f"Target concepts specified in config {class_to_target_concepts_path}: {config_concept_list}")

    # *** 該当wiki pageのsummaryから、概念をうまく説明する1文を抽出し、概念名を置換したければ置換する.
    concept_to_one_summary_sentence = {}
    for concept in config_concept_list:
        first_concept_sentence = get_concept_embedding_text(
            concept, 
            None
        )
        if first_concept_sentence is None:
            # 現在のget_concept_embedding_text()では、概念を説明する1文がwiki summaryから見つけられなかったためskip.
            continue
        concept_to_one_summary_sentence[concept] = first_concept_sentence
        # print(f"Summary with concept name for concept '{concept}': {summary}\n")
        print(f"First wiki sentence containing concept '{concept}': \n\t{first_concept_sentence}\n")

    # 何割について文を抽出できたかを確認
    extracted_count = len(concept_to_one_summary_sentence)
    total_count = len(config_concept_list)
    print(f"Extracted sentences for {extracted_count}/{total_count} concepts ({extracted_count/total_count*100:.2f}%)")

    # =========================
    # promptの作成 (目標vec生成用) 
    # =========================
    # prompt_base = "It is the <target_concept>."
    concept_to_prompt = {}
    for concept, one_summary_sentence in concept_to_one_summary_sentence.items():   # in config_concept_list:
        print(f"Processing concept: {concept}")
        if pool_hs_type == "repeat_mean_pool":
            # 目標ベクトルを生成するためのpromptを作成. 例えば、"It is the apple. It is the apple." のように、同じ文を2回繰り返すことで、gemmaのattentionが、後半の文の方に向くようにする。
            prompt = one_summary_sentence * 2 
            pool_hs_type = "mean_pool" # pool_hs_typeがrepeat_mean_poolの場合は、pool_hs_typeをmean_poolに変更して、後半の文の隠れ状態の平均を目標ベクトルとする. これにより、gemmaのattentionが、後半の文の方に向くようにする。
        else:
            prompt = one_summary_sentence
        print(f"Prompt for concept '{concept}': {prompt}")
        concept_to_prompt[concept] = prompt

    # config_concept_list の順番に対応するpromptのリストを作成
    concept_names, text_list = [], []
    # text_list = [concept_to_prompt[concept] for concept in config_concept_list if concept in concept_to_prompt]
    for concept in config_concept_list:
        if concept in concept_to_prompt:
            concept_names.append(concept)
            text_list.append(concept_to_prompt[concept])
        else:
            # print(f"Warning: No prompt could be created for concept '{concept}' because no suitable sentence was found in the wiki summary. This concept will be skipped.")
            pass


    # return 0  # [memo] ここまで動作確認済み


    # =========================
    # ** モデル読み込み **
    # =========================
    print("Loading model and tokenizer...")
    model_version = get_gemma_model_version(model_size)

    # [WIP] 'it'と'pt'のどちらが良いかは未検証.とりあえず'it'で統一.
    model_name = f"google/gemma-{model_version}-{model_size}b-it" # [memo] 'gemma-'部分は変えないこと!! -を消すとモデルがloadできない．さらにそのエラーメッセージは，"huggingface-cli login"をして，という関係ないmessageになるので注意!
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not debug_without_model:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


    if tokenizer.pad_token_id is None:
        # llama系の場合はpad_tokenが設定されていないことがあるため，以下のようにeos_tokenをpad_tokenに設定する. gemma3は設定済みだった
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if not debug_without_model:
            model.config.pad_token_id = tokenizer.pad_token_id


    # * デバッグ: tokenizerの動作確認. text_listの各テキストがどのようにtokenizeされるか、またdecodeするとどうなるかを確認する. これにより、tokenizerが想定通りに動いているか、特にEOSトークンの扱いがどうなっているかを確認できる.
    print(f"dot token id: {tokenizer.convert_tokens_to_ids('.')}")
    for i, text in enumerate(text_list):
        encoded = tokenizer(text, return_tensors="pt")
        print(f"Tokenized the text of concept '{config_concept_list[i]}': {encoded}")
        decoded = tokenizer.decode(encoded["input_ids"][0])
        print(f"Decoded back: {decoded}\n")

    # =========================
    # *** 全層の隠れ状態を全て記録に残す．最終dot(.)位置のみのベクトル，全体のmean pool, の2種類を記録する．***
    # =========================
    if debug_without_model:
        device = torch.device("cpu")
    else:
        model.eval() # 評価モードに切り替え (これにより、dropoutなどの挙動が変わる)
        device = model.device

        # *** pool_hs_type に応じて、vectorを抽出 ***
        if pool_hs_type == "repeat_mean_pool":
            data_type = "wiki_summary_repeat"
        else:
            data_type = "wiki_summary"
        all_vecs = extract_hidden_states(
            model, 
            tokenizer,
            text_list, 
            pool_hs_type, 
            data_type, 
            batch_size=8, 
            layer_index=layer_index,
            print_flag=False
        )   # -> (T, D) or (T, H, D) Tはテキスト数, Hは層の数, Dは隠れ状態の次元

        # ベクトルを保存
        output_path = os.path.join(output_dir, f"goal_embeddings_{model_size}B_{target_concepts_filename.split('.')[0]}_{pool_hs_type}_layer{layer_index}")
        np.savez(
            output_path, 
            vectors=all_vecs,
            target_concepts_filename=target_concepts_filename,
            concept_names=concept_names,
            text_list=text_list,
            model_size=model_size,
            pool_hs_type=args.pool_hs_type, # repeat_の場合、途中でmean_poolに変えてしまったため、pool_hs_typeではなく、元のargs.pool_hs_typeを保存する
            layer_index=layer_index,
        )
        print(f"Saved goal embeddings to {output_path}")




    

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"') # *** 🟠 
    parser.add_argument("--model_size", type=int, default=3, help="Size of the Gemma model in billions (e.g., 3 for Gemma-3B).")
    parser.add_argument("--pool_hs_type", type=str, default="repeat_mean_pool", help='hidden stateのpooling方法. "eos": 最後のEOSトークンの隠れ状態を使用. "last_token": 最後のトークンの隠れ状態を使用. "mean_pool": テキスト全体の隠れ状態の平均を使用. "repeat_mean_pool": テキストを2回繰り返した内の後のtextの隠れ状態の平均を使用. "dot": テキスト全体の隠れ状態を平均したものと、最後のトークンの隠れ状態を連結して使用.')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


    main(args)


"""
TARGET_CONCEPTS_FILENAME="target_concepts_mini_13.json"
MODEL_SIZE=12

uv run python src/generate_goal_embeddings.py \
    --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
    --model_size ${MODEL_SIZE} \
    --pool_hs_type "repeat_mean_pool" \
    --cuda_visible_devices 4

nohup uv run python src/generate_goal_embeddings.py \
    --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
    --model_size ${MODEL_SIZE} \
    --pool_hs_type "repeat_mean_pool" \
    --cuda_visible_devices 4 \
    > log_generate_goal_embeddings_gemma-${MODEL_SIZE}B_${TARGET_CONCEPTS_FILENAME}.log 2>&1 &

3748904

"""