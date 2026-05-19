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
from utils.gemma_train_and_test_utils import fix_seed, get_gemma_model_version, save_mem_vec, constructTrainSamples, encodeTrainSamplesWithTokenizer, evaluateModel #, train
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


debug_without_model = True

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
    concept_pattern = re.escape(concept)                            # 概念名が含まれるかどうかを調べるパターン。concept名を正規表現の特殊文字をエスケープしてパターン化
    brackets_dup_pattern = re.compile(r'\s*\(.*?\[.*?\].*?\)\s*')   # ( ... [..] ... ) のような()内に[]が入るパターン. wiki title(data名) と wiki page文章内の綴りが異なっても、( [])内に読み方が含まれれば、その前部分が概念名であるとわかる。例: concept名: 'House of Hohenstaufen' wiki文章: "The Hohenstaufen dynasty (, US also , German: [ˌhoːənˈʃtaʊfn̩]), also known as the Staufer,  ..."
    is_the_pattern = re.compile(r'(.+?)\s+is\s+the\s+', re.IGNORECASE)      # " is the " が含まれるかどうかを調べるパターン
    brackets_pattern = re.compile(r'\s*\(.*?\)\s*')                 # (  AB-ə [ˈâbːa]) のような読み方が書かれることがある. 


    # concept pattern に含まれる一部の文字が、wiki page内では他の文字で表されている場合があるため、その場合のpatternも作成しておく。
    concept_patterns = [concept_pattern]
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
        # print(f"First wiki sentence containing concept '{concept}': \n\t{first_concept_sentence}\n")


    # ② 概念名を含む文が見つからない場合 → ( []) を含む文を探す。概念名が違う言語で書かれている場合などがある。この場合、他言語の概念名 (名前 [発音]) の表記であることがあるため、そのパターンを探し、冒頭から()までを概念名に置換する
    if first_concept_sentence is None:
        print(f"No sentence containing concept '{concept}' was found in the summary. Trying to find a sentence containing '( ... [..] ... )' pattern to use as a template for the concept name.\n")
        brackets_dup_sentences = [s for s in sentences if brackets_dup_pattern.search(s)]
        # for s in brackets_dup_sentences: print(f"\t * Sentence containing '( ... [..] ... )' pattern: \n\t\t{s}\n")
        if brackets_dup_sentences:
            first_brackets_dup_sentence = brackets_dup_sentences[0]
            # modified_sentence = brackets_dup_pattern.sub(concept + " ", first_brackets_dup_sentence)
            # その文の中の、brackets_dup_patternにマッチする部分とそれ以前を概念名に置換する
            modified_sentence = re.sub(rf"^.*?{brackets_dup_pattern.pattern}", concept + " ", first_brackets_dup_sentence)           
            first_concept_sentence = modified_sentence
            print(f"\t No sentence containing concept '{concept}' was found. Using modified '( ... [..] ... )' sentence: \n\t\t{first_concept_sentence}\n")

    # ③ 概念名を含む文が見つからない場合 → " is the "を含む文を探して概念名を含む文を作る．
    # 例えば 'Arnhem Bridge' のsummaryの1文目は "John Frost Bridge (John Frostbrug in Dutch) is the road bridge over the Lower Rhine at Arnhem, in the Netherlands." である．この is the の前の部分を，concept名に置換して使う．
    if first_concept_sentence is None:
        print(f"No sentence containing concept '{concept}' was found in the summary. Trying to find a sentence containing 'is the' to use as a template for the concept name.\n")
        is_the_sentences = [s for s in sentences if is_the_pattern.search(s)]
        # for s in is_the_sentences: print(f"\t * Sentence containing 'is the': \n\t\t{s}\n")
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
        modified_sentence = re.sub(concept_pattern, target_name_to_replace_with_concept, first_concept_sentence, flags=re.IGNORECASE)
        # () が含まれる場合、()内を削除. (  AB-ə [ˈâbːa]) のような読み方が書かれることがあるため。
        brackets_pattern = re.compile(r'\s*\(.*?\)\s*')
        first_concept_sentence = modified_sentence
        print(f"Modified first or containing 'is the' sentence after replacement: \n\t{first_concept_sentence}\n")
        
    return first_concept_sentence
    



# *************************************************************** main ***************************************************************
def main(args):
    model_size = args.model_size
    target_concepts_filename = args.target_concepts_filename


    model_version = get_gemma_model_version(model_size)
    
    # ** 保存先 **
    # output_dir = os.path.join(project_root, "output", f"gemma-{model_version}-{model_size}B_lr{lr}_{target_concepts_filename.split('.')[-1]}_{pool_hs_type}_layer{layer_idx}")
    output_dir = os.path.join("/work04/toko/EmbedNewConcept-20260305", "goal_embeddings", f"gemma-{model_version}-{model_size}B_{target_concepts_filename.split('.')[-1]}")
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

    # 該当wiki pageのsummaryを取得する
    concept_to_one_summary_sentence = {}
    for concept in config_concept_list:
        first_concept_sentence = get_concept_embedding_text(concept, "banana")
        if first_concept_sentence is None:
            # 現在のget_concept_embedding_text()では、概念を説明する1文がwiki summaryから見つけられなかったためskip.
            continue
        concept_to_one_summary_sentence[concept] = first_concept_sentence
        # print(f"Summary with concept name for concept '{concept}': {summary}\n")
        print(f"First wiki sentence containing concept '{concept}': \n\t{first_concept_sentence}\n")




    # =========================
    # promptの作成 (目標vec生成用) 
    # =========================
    # prompt_base = "It is the <target_concept>."
    concept_to_prompt = {}
    for concept in config_concept_list:
        print(f"Processing concept: {concept}")
        prompt = concept_to_one_summary_sentence[concept] * 2 
        print(f"Prompt for concept '{concept}': {prompt}")
        concept_to_prompt[concept] = prompt

    return 0
    # config_concept_list の順番に対応するpromptのリストを作成
    text_list = [concept_to_prompt[concept] for concept in config_concept_list]


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
    batch_size = 8
    if debug_without_model:
        device = torch.device("cpu")
    else:
        model.eval() # 評価モードに切り替え (これにより、dropoutなどの挙動が変わる)
        device = model.device

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        # ** tokenize and generate **
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False #last_token_is_eos#LAST_TOKEN_IS_EOS, -> pool_hs_type == "eos"の場合は明示的にeosを追加済みなので、ここをTrueにするとeosが重複して2つ付く可能性がある。そのためここはFalseで良い。
        ).to(device) 

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # with torch.no_grad():
        #     outputs = model(**inputs, output_hidden_states=True)
        #     # hidden_states は tuple:
        #     # 0: embedding出力, 1..L: 各層出力
        #     hs = outputs.hidden_states
        #     # layer_hs = hs[layer_index]      # (B, T, H)


        # *** pool_hs_type に応じて、vectorを抽出 ***
        for s_idx in range(input_ids.size(0)):

            # 1 が立っている位置を取得 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] -> valid_pos = [6, 7, 8, 9, 10, 11] 
            valid_pos = torch.nonzero(attention_mask[s_idx], as_tuple=False).squeeze(-1)
            # print(f"valid_pos for batch {s_idx}: {valid_pos}")
            if valid_pos.numel() == 0:
                # 全部 padding の場合
                pos_begin = 0
                pos_end = 0
            else:
                pos_begin = valid_pos[0].item()
                pos_end = valid_pos[-1].item() + 1   # slice用に end は exclusive

            # dot (.) 位置のベクトルを抽出 (テンソルの中で「0でない（≒True）」要素のインデックスを取得する, その際tupleではなくテンソルで返すためにas_tuple=Falseを指定, さらに次元が1のときに余分な次元を削除するためにsqueeze(-1)を指定)
            dot_pos = (input_ids[s_idx] == tokenizer.convert_tokens_to_ids('.')).nonzero(as_tuple=False).squeeze(-1)
            if dot_pos.numel() == 0:
                # # dotがない場合は、最後のトークン位置をdot位置とする
                # dot_pos = pos_end - 1
                # dot がない場合はerror
                raise ValueError(f"Batch {s_idx} のテキスト '{batch_texts[s_idx]}' にはdot(.)が含まれていません。dot(.)を含むテキストを使用してください。")

            # debug
            print(f"Batch {s_idx} text: '{batch_texts[s_idx]}'")
            print(f"Token IDs: {input_ids[s_idx]}")
            print(f"Attention Mask: {attention_mask[s_idx]}")
            print(f"Valid token positions: {valid_pos}")
            print(f"Dot positions: {dot_pos}")
            print(f"pos_begin: {pos_begin}, pos_end: {pos_end}")


        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"') # *** 🟠 
    parser.add_argument("--model_size", type=int, default=3, help="Size of the Gemma model in billions (e.g., 3 for Gemma-3B).")
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


    main(args)


"""
TARGET_CONCEPTS_FILENAME="target_concepts_mini_13.json"
MODEL_SIZE=4

uv run python src/generate_goal_embeddings.py \
    --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
    --model_size ${MODEL_SIZE} \
    --cuda_visible_devices 4

nohup uv run python src/generate_goal_embeddings.py \
    --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
    --model_size ${MODEL_SIZE} \
    --cuda_visible_devices 4 \
    > log_generate_goal_embeddings_gemma-${MODEL_SIZE}B_target${TARGET_CONCEPTS_FILENAME}.log 2>&1 &
"""