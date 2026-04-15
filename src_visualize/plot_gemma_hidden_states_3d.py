
"""このスクリプトは、Gemmaモデルの特定の層におけるEOSトークン位置の隠れ状態を抽出し、PCAで次元削減してプロットするものです。

- 固有名詞リスト(proper_nouns)の各テキストの末尾にEOSトークンを追加して、その位置の隠れ状態を抽出します。(LAST_TOKEN_IS_EOS=Trueの場合)
- 抽出した隠れ状態をPCAで2次元に削減し、プロットします。
- プロットはoutput/hidden_state_pca_plotsディクトリに保存されます。

使用方法:
1. モデルのバージョンとサイズを設定します。
2. 固有名詞リスト(proper_nouns)を必要に応じて変更します。
3. スクリプトを実行して、プロットを生成します。


実行例:
nohup uv run python3 src_visualize/plot_gemma_hidden_states_3d.py \
    --layer_index 12 \
    --data_type wiki_summary_repeat \
    --pool_hs_type mean_pool \
    --threshold_num_nouns_per_category 20 \
    --config_filename target_concepts_mini_10 \
    --cuda_device 4 \
    > plot3d_wiki_summary_repeat.log 2>&1 &

    
nohup uv run python3 src_visualize/plot_gemma_hidden_states_3d.py \
    --layer_index 36 \
    --data_type wiki_summary_repeat \
    --pool_hs_type mean_pool \
    --threshold_num_nouns_per_category 20 \
    --config_filename target_concepts_mini_12 \
    --cuda_device 0 \
    > plot3d_wiki_summary_repeat.log 2>&1 &

    
    

nohup uv run python3 src_visualize/plot_gemma_hidden_states_3d.py \
    --layer_index 9 \
    --data_type wiki_summary_promptEOL \
    --pool_hs_type last_token \
    --threshold_num_nouns_per_category 20 \
    --num_categories 12 \
    --cuda_device 0 \
    > plot3d_wiki_summary_promptEOL.log 2>&1 &

nohup uv run python3 src_visualize/plot_gemma_hidden_states_3d.py \
    --layer_index 36 \
    --data_type wiki_summary_promptEOL \
    --pool_hs_type last_token \
    --threshold_num_nouns_per_category 20 \
    --num_categories 12 \
    --cuda_device 3 \
    > plot3d_wiki_summary_promptEOL.log 2>&1 &




nohup uv run python3 src_visualize/plot_gemma_hidden_states_3d.py \
    --data_type proper_nouns \
    --pool_hs_type mean_pool \
    --threshold_num_nouns_per_category 20 \
    --num_categories 10 \
    --cuda_device 2 \
    > plot3d_proper_nouns.log 2>&1 &


nohup uv run python3 src_visualize/plot_gemma_hidden_states_3d.py \
    --data_type wiki_main_text \
    --pool_hs_type mean_pool \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 1 \
    > plot3d_wiki_main_text.log 2>&1 &

"""

word_max_num_threshold = 300 # 256 # 512 # 256 # 128
word_min_num_threshold = 32  # 16
# num_char_threshold = 400
marker_size = 3 # 6


import argparse
import os
import sys
import random
import re
import json
from unicodedata import category
from tqdm import tqdm
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import plotly.express as px

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

project_root = os.path.join(os.path.dirname(__file__), "..") #
sys.path.append(project_root)

from utils.wikipedia_api_utils import extract_wiki_main_text, fetch_wikipedia_page
from utils.gemma_train_and_test_utils import fix_seed, extract_hidden_states
from utils.handle_text_utils import get_first_few_sentences, repeat_text
seed = 42


# =========================
# 設定
# =========================
# data_type = "proper_nouns" # "wiki_summary" # "proper_nouns" or "wiki_summary" proper_nounsの場合は固有名詞を、wiki_summaryの場合はその固有名詞をタイトルとするwikiページのsummaryをLLMに入力する
# CUDA_VISIBLE_DEVICES = "1" # "2"

model_version = "3"
model_size = "12"
# LAST_TOKEN_IS_EOS = True  # 固有名詞の最後にEOSトークンを追加して、その位置のhidden stateを取るかどうか。Falseの場合は最終トークン位置の隠れ状態を取得する -> pool_hs_type で指定するように変更したため、今はこの変数は使用していないが、後で削除予定
# LAYER_INDEX = 9     # どの層の hidden state を使うか
BATCH_SIZE = 1 # 8

MODEL_NAME = f"google/gemma-{model_version}-{model_size}b-it"
DEVICE = "cuda"     # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# word_num_threshold = 128 # wiki summaryやmain textが長すぎる場合に、最初の数文だけを使用するための単語数の閾値。例えば300に設定した場合、最初の数文を連結していって、連結したテキストの単語数が300を超えないようにする。1文目ですでに300単語を超える場合は、最初の1文だけを使用する。

output_dir = os.path.join(project_root, "src_visualize", "output", "hidden_state_pca_plots")

# config_filename = "concepts_10" # "all_concepts"# "target_concepts_long"
# threshold_num_nouns_per_category = 30 # 各カテゴリからプロットに使用する固有名詞の最大数。あまりに多いとプロットが見づらくなるため。Noneの場合は全て使用する。





def main(args):
    print("Start plot_gemma_hidden_states_3d.py")

    layer_index = args.layer_index
    data_type = args.data_type
    pool_hs_type = args.pool_hs_type
    threshold_num_nouns_per_category = args.threshold_num_nouns_per_category
    # num_categories = args.num_categories
    cuda_device = args.cuda_device
    config_filename = args.config_filename

    # if num_categories == None:
    #     config_filename = "all_concepts"
    # else:
    #     config_filename = f"concepts_{num_categories}"


    
    output_path = os.path.join(output_dir, f"{MODEL_NAME.replace('/', '_')}_layer{layer_index}_{config_filename}_datatype_{data_type}_poolHStype_{pool_hs_type}_wordnum_{word_min_num_threshold}_to_{word_max_num_threshold}_pca_3d.html")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    fix_seed(seed)

    # ************ data preparation ************

    # 可視化したい固有名詞リスト
    # category_to_proper_nouns = {
    #     "都市": ["東京", "大阪", "京都", "渋谷"],
    #     "企業": ["ソニー", "トヨタ", "任天堂", "OpenAI"],
    #     "地理": ["富士山", "北海道"],
    # }
    with open(os.path.join(project_root, "config", f"{config_filename}.json"), "r") as f:
        category_to_proper_nouns_tmp = json.load(f)

    category_to_proper_nouns = {}
    for category, nouns in tqdm(category_to_proper_nouns_tmp.items(), desc="Processing categories"):
        # if len(nouns) > 0:
        if len(nouns) == 0:
            print(f"カテゴリ '{category}' に固有名詞が見つかりませんでした。スキップします。")
            continue
        if len(nouns) <= threshold_num_nouns_per_category:
            category_to_proper_nouns[category] = nouns
        else:
            # category_to_proper_nouns[category] = random.sample(nouns, threshold_num_nouns_per_category)
            # 最初のn個を使用する
            category_to_proper_nouns[category] = nouns[:threshold_num_nouns_per_category]


    if data_type == "proper_nouns":
        # 固有名詞をpromptとする場合
        category_to_plot_data = category_to_proper_nouns

    elif data_type.startswith("wiki_summary"):
        # 固有名詞をタイトルとするwikiページのsummaryをpromptとする場合
        category_to_plot_data = defaultdict(list)

        word_counts, char_counts = [], []   # ⭐️ついでに、summaryの平均単語数・文字数・標準偏差も計算して表示する


        for category, nouns in tqdm(category_to_proper_nouns.items(), desc="Processing wiki summaries"):
            for noun in nouns:
                # 該当wiki pageのsummaryを取得する
                # 辞書にまだ保存されていなければ、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに格納する
                summary = load_wiki_text(noun, text_type="summary")
                if summary is None:
                    # print(f"'{noun}' のWikipedia summaryが見つかりませんでした。スキップします。")
                    continue

                # ⭐️ついでに、summaryの単語数と文字数をカウントして統計量を出す用
                word_counts.append(len(summary.split()))
                char_counts.append(len(summary))


                # 長すぎるsummaryがあるため、最初の数文だけをsummaryとして使用する
                truncated_summary = get_first_few_sentences(summary, word_min_num_threshold, word_max_num_threshold)
                if truncated_summary is None:
                    print(f"'{noun}' のWikipedia summaryは、{word_min_num_threshold} ~ {word_max_num_threshold}単語の範囲内に収まらないため、スキップします。") # 最初の100文字だけ表示
                    continue
                target_summary = truncated_summary

                if data_type == "wiki_summary_repeat":
                    # summaryを2回繰り返してプロンプトとする場合: https://openreview.net/forum?id=Ahlrf2HGJR の手法
                    target_summary = repeat_text(target_summary, times=2) # target_summaryを2回繰り返す

                if data_type == "wiki_summary_promptEOL":
                    # "This sentence: "[text]" means in one word: " の形式で文を埋め込み、最終トークン位置に意味が凝縮されることを期待する手法 https://aclanthology.org/2024.findings-emnlp.181/ の手法
                    target_summary = f"This sentence: \"{target_summary}\" means in one word: "
                    # target_summary = f'This sentence: "{target_summary}" means in one word: '
                    # この手法の場合は自動的にpool_hs_type==last_tokenにする
                    if pool_hs_type != "last_token":
                        print(f"Warning: data_typeが wiki_summary_promptEOL であるため、現在 {pool_hs_type} と指定されている pool_hs_type を自動的に last_token に修正します。")
                        pool_hs_type = "last_token"

                category_to_plot_data[category].append(target_summary)
        
        
        # ⭐️ついでに、summaryの単語数と文字数をカウントして統計量を出す用
        if len(word_counts) > 0:
            print(f"Summary word count - mean: {np.mean(word_counts):.2f}, std: {np.std(word_counts):.2f}, min: {np.min(word_counts)}, max: {np.max(word_counts)}")
        if len(char_counts) > 0:
            print(f"Summary char count - mean: {np.mean(char_counts):.2f}, std: {np.std(char_counts):.2f}, min: {np.min(char_counts)}, max: {np.max(char_counts)}")


    elif data_type == "wiki_main_text":
        # 固有名詞をタイトルとするwikiページのmain textをpromptとする場合
        # [WIP] この方法はうまくいっていない。単語数と文字数の計算が一部のページでうまくいかないよう。多分数式が書かれているファイルでおかしくなっており、cuda ort of memoryが治らない。
        # -> そのため、plot_gemma_eos_hidden_states_3d_20260408.py の方には該当コードがあるが、ここでは削除した。
        print("You can't use wiki main text for now because of some issues with long texts. Please fix and use plot_gemma_eos_hidden_states_3d_20260408.py instead, which has some fixes for handling long texts.")
        return 0


    # =========================
    # フラット化
    # =========================
    input_texts = []
    categories = []

    for category, input_t_list in category_to_plot_data.items():
        for input_text in input_t_list:
            input_texts.append(input_text)
            categories.append(category)



    # =========================
    # モデル読み込み
    # =========================
    print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=None
    ).to(DEVICE)
    model.eval()

    # Gemma系で pad token が未設定な場合の保険
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id が見つかりません。")



    # =========================
    # 特徴抽出
    # =========================
    features = extract_hidden_states(
        model,
        tokenizer,
        input_texts,
        pool_hs_type=pool_hs_type,
        data_type=data_type,
        batch_size=BATCH_SIZE,
        layer_index=layer_index,
    )


    print("features shape:", features.shape)  # (N, hidden_dim)

    # NaN / inf 除外
    valid_mask = np.isfinite(features).all(axis=1)
    valid_features = features[valid_mask]
    valid_input_texts = [x for x, ok in zip(input_texts, valid_mask) if ok]
    valid_categories = [x for x, ok in zip(categories, valid_mask) if ok]

    print("valid samples:", len(valid_input_texts), "/", len(input_texts))

    if len(valid_input_texts) < 2:
        raise ValueError("有効サンプルが少なすぎて PCA できません。")



    # =========================
    # PCA
    # =========================
    pca = PCA(n_components=3)
    coords = pca.fit_transform(valid_features)

    print("Explained variance ratio:", pca.explained_variance_ratio_)


    # =========================
    # DataFrame化
    # =========================
    df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "PC3": coords[:, 2],
        "label": valid_input_texts,
        "category": valid_categories,
    })


    # =========================
    # Plot
    # =========================
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="category",
        hover_name="label",
        hover_data={"category": True, "PC1": ':.3f', "PC2": ':.3f', "PC3": ':.3f'},
        title=f"3D PCA of hidden states ({MODEL_NAME}, layer={layer_index}, pool_hs_type={pool_hs_type}, data_type={data_type})",
    )
    fig.update_traces(marker=dict(size=marker_size))


    # 保存
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(output_path)
    print(f"Plot saved to: {output_path}")







def load_wiki_text(propnoun, text_type="summary"):
    """dbpediaから収集した固有名詞のwikipedia summaryを読み込んで、propnoun_to_wikisummaryに保存する。
    data/wiki_pages に未保存であれば、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに保存する
    """
    # print("Loading Wikipedia summaries for prop nouns...")
    wiki_pages_dir = os.path.join(project_root, "data", "wiki_pages")

    filename = re.sub(r'[/\\ ]', '_', propnoun) + ".json"  # ファイル名に使用できない文字を置換
    wikipage_path = os.path.join(wiki_pages_dir, filename)
    

    # * 未取得の場合、wikipedia apiから取得して保存する
    if not os.path.exists(wikipage_path):
        wiki_info = fetch_wikipedia_page(propnoun, lang="en")
        if wiki_info["exists"] == False:
            print(f"Wikipedia page for concept '{propnoun}' DOES NOT exist. Skipping generation.")
            return None
        # 本文を切り出す
        main_text = extract_wiki_main_text(wiki_info['text'])
        wiki_info['text'] = main_text

        # 保存
        with open(wikipage_path, "w") as f:
            json.dump(wiki_info, f, ensure_ascii=False, indent=4)


    # * 今ここで保存した or すでに保存されているwikipedia summaryを読み込む
    with open(wikipage_path, "r") as f:
        wiki_page = json.load(f)

    if text_type == "summary":
        summary = wiki_page.get("summary")
        if summary:
            return summary
        else:
            print(f"No summary found in wiki page for '{propnoun}' in wiki_pages.")
            return None
        
    elif text_type == "main_text":
        main_text = wiki_page.get("text")
        if main_text:
            return main_text
        else:
            print(f"No main text found in wiki page for '{propnoun}' in wiki_pages.")
            return None







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_index", type=int, default=9, help="どの層の隠れ状態を抽出するか。0がembedding layerの出力、1が1層目の出力、...、-1が最後の層の出力")
    parser.add_argument("--data_type", type=str, default="proper_nouns", choices=["proper_nouns", "wiki_summary", "wiki_summary_repeat", "wiki_summary_promptEOL", "wiki_main_text"], help="プロットに使用するデータの種類。固有名詞を使用するか、wiki summaryを使用するか")
    parser.add_argument("--pool_hs_type", type=str, default="eos", choices=["eos", "last_token", "mean_pool"], help="隠れ状態のプーリング方法。今はEOSトークン位置の隠れ状態のみ対応")
    parser.add_argument("--threshold_num_nouns_per_category", type=int, default=30, help="各カテゴリからプロットに使用する固有名詞の最大数。あまりに多いとプロットが見づらくなるため。Noneの場合は全て使用する。")
    parser.add_argument("--num_categories", type=int, default=10, help="プロットに使用するカテゴリの数。最初のnカテゴリを使用する。Noneの場合は全てのカテゴリを使用する。")
    parser.add_argument("--config_filename", type=str, default=None, help="設定ファイルの名前。Noneの場合はデフォルト名が使用される。")
    parser.add_argument("--cuda_device", type=str, default="1", help="使用するCUDAデバイスのID (例: '0', '1', '2', ...)")
    args = parser.parse_args()
    main(args)


"""
nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_summary_repeat \
    --pool_hs_type mean_pool \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 1 \
    > plot3d_wiki_summary.log 2>&1 &
4103705

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type proper_nouns \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 2 \
    > plot3d_proper_nouns.log 2>&1 &
4104268
"""