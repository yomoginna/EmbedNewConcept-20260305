
"""このスクリプトは、Gemmaモデルの特定の層におけるEOSトークン位置の隠れ状態を抽出し、PCAで次元削減してプロットするものです。

- 固有名詞リスト(proper_nouns)の各テキストの末尾にEOSトークンを追加して、その位置の隠れ状態を抽出します。(LAST_TOKEN_IS_EOS=Trueの場合)
- 抽出した隠れ状態をPCAで2次元に削減し、プロットします。
- プロットはoutput/hidden_state_pca_plotsディクトリに保存されます。

使用方法:
1. モデルのバージョンとサイズを設定します。
2. 固有名詞リスト(proper_nouns)を必要に応じて変更します。
3. スクリプトを実行して、プロットを生成します。


実行例:
nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_summary \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 1 \
    > plot3d_wiki_summary.log 2>&1 &
4119014

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type proper_nouns \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 2 \
    > plot3d_proper_nouns.log 2>&1 &
4104268

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_main_text \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 1 \
    > plot3d_wiki_main_text.log 2>&1 &
4182009




nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_main_text \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 2 \
    > plot3d_wiki_main_text_16.log 2>&1 &
    
4187770

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_main_text \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 1 \
    > plot3d_wiki_main_text_32.log 2>&1 &
20032

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_main_text \
    --threshold_num_nouns_per_category 30 \
    --num_categories 10 \
    --cuda_device 0 \
    > plot3d_wiki_main_text_64.log 2>&1 &
4183774

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_main_text \
    --threshold_num_nouns_per_category 50 \
    --num_categories 10 \
    --cuda_device 0 \
    > plot3d_wiki_main_text_64.log 2>&1 &


nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_main_text \
    --threshold_num_nouns_per_category 50 \
    --num_categories 10 \
    --cuda_device 0 \
    > plot3d_wiki_main_text_128.log 2>&1 &

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_main_text \
    --threshold_num_nouns_per_category 50 \
    --num_categories 10 \
    --cuda_device 0 \
    > plot3d_wiki_main_text_256.log 2>&1 &

nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
--data_type wiki_main_text \
--threshold_num_nouns_per_category 50 \
--num_categories 10 \
--cuda_device 0 \
> plot3d_wiki_main_text_512.log 2>&1 &

"""

word_num_threshold = 64
# num_char_threshold = 400

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
from utils.gemma_train_and_test_utils import fix_seed

seed = 42


# =========================
# 設定
# =========================
# data_type = "proper_nouns" # "wiki_summary" # "proper_nouns" or "wiki_summary" proper_nounsの場合は固有名詞を、wiki_summaryの場合はその固有名詞をタイトルとするwikiページのsummaryをLLMに入力する
# CUDA_VISIBLE_DEVICES = "1" # "2"

model_version = "3"
model_size = "12"
LAST_TOKEN_IS_EOS = True  # 固有名詞の最後にEOSトークンを追加して、その位置のhidden stateを取るかどうか。Falseの場合は最終トークン位置の隠れ状態を取得する
LAYER_INDEX = 9     # どの層の hidden state を使うか
BATCH_SIZE = 1 # 8

MODEL_NAME = model_name = f"google/gemma-{model_version}-{model_size}b-it"
DEVICE = "cuda"     # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# word_num_threshold = 128 # wiki summaryやmain textが長すぎる場合に、最初の数文だけを使用するための単語数の閾値。例えば300に設定した場合、最初の数文を連結していって、連結したテキストの単語数が300を超えないようにする。1文目ですでに300単語を超える場合は、最初の1文だけを使用する。

output_dir = os.path.join(project_root, "src_visualize", "output", "hidden_state_pca_plots")

# config_filename = "concepts_10" # "all_concepts"# "target_concepts_long"
# threshold_num_nouns_per_category = 30 # 各カテゴリからプロットに使用する固有名詞の最大数。あまりに多いとプロットが見づらくなるため。Noneの場合は全て使用する。





def main(args):
    print("Start plot_gemma_eos_hidden_states_3d.py")


    data_type = args.data_type
    threshold_num_nouns_per_category = args.threshold_num_nouns_per_category
    num_categories = args.num_categories
    cuda_device = args.cuda_device

    if num_categories == None:
        config_filename = "all_concepts"
    else:
        config_filename = f"concepts_{num_categories}"


    
    output_path = os.path.join(output_dir, f"{MODEL_NAME.replace('/', '_')}_layer{LAYER_INDEX}_{config_filename}_datatype_{data_type}_pca_3d.html")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    fix_seed(seed)

    

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
            category_to_proper_nouns[category] = random.sample(nouns, threshold_num_nouns_per_category)


    if data_type == "proper_nouns":
        category_to_plot_data = category_to_proper_nouns
    elif data_type == "wiki_summary":
        category_to_plot_data = defaultdict(list)
        for category, nouns in tqdm(category_to_proper_nouns.items(), desc="Processing wiki summaries"):
            for noun in nouns:
                # 該当wiki pageのsummaryを取得する
                # 辞書にまだ保存されていなければ、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに格納する
                summary = load_wiki_text(noun, text_type="summary")
                if summary is None:
                    # print(f"'{noun}' のWikipedia summaryが見つかりませんでした。スキップします。")
                    continue

                # # * 長すぎるsummaryがあるため、最初の5文だけをsummaryとして使用する
                # # truncated_summary = " ".join(summary_sentences[:5])
                # # 文末記号を保持して分割
                # parts = re.split(r'(?<=[。．!?！？])\s*', summary)
                # sentences = []
                # delimiters = []
                # for i in range(0, len(parts) - 1, 2):
                #     sentence = parts[i].strip()
                #     # delimiter = parts[i + 1]
                #     delimiters.append(parts[i + 1].strip() if i + 1 < len(parts) else "")
                #     if sentence:
                #         sentences.append(sentence)
                
                # word_count = 0
                # truncated_summary = ""
                # for i, sentence in enumerate(sentences):
                #     word_count += len(sentence)
                #     truncated_summary += sentence + delimiters[i]  # 文末記号を保持して連結
                #     if word_count > word_threshold:  # 300単語を超えないようにする
                #         break

                truncated_summary = get_first_few_sentences(summary, word_num_threshold)
                if truncated_summary is None:
                    print(f"'{noun}' のWikipedia summaryは、最初の文だけで既に{word_num_threshold}単語を超えているため、スキップします。")
                    continue
                category_to_plot_data[category].append(truncated_summary)

    elif data_type == "wiki_main_text":
        category_to_plot_data = defaultdict(list)
        for category, nouns in tqdm(category_to_proper_nouns.items(), desc="Processing wiki main texts"):
            for noun in nouns:
                # 該当wiki pageのsummaryを取得する
                # 辞書にまだ保存されていなければ、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに格納する
                main_text = load_wiki_text(noun, text_type="main_text")
                if main_text is None:
                    # print(f"'{noun}' のWikipedia main textが見つかりませんでした。スキップします。")
                    continue

                # 長すぎるmain textがあるため、最初の2文だけをmain textとして使用する. (summaryよりも本文の方が1文が長いようなので、summaryよりも少ない文数にする)
                # main_text_sentences = re.split(r'(?<=[。．!?！？])\s*', main_text)
                # truncated_main_text = " ".join(main_text_sentences[:2])
                # 300tokenを超えないように、最初の数文を使用する
                # truncated_main_text = ""
                # for sentence in main_text_sentences:
                #     if len(tokenizer(truncated_main_text + sentence)["input_ids"]) > 300:
                #         break
                #     truncated_main_text += sentence


                truncated_main_text = get_first_few_sentences(main_text, word_num_threshold)
                if truncated_main_text is None:
                    print(f"'{noun}' のWikipedia main textは、最初の文だけで既に{word_num_threshold}単語を超えているため、スキップします。")
                    continue

                category_to_plot_data[category].append(truncated_main_text)





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
    features = extract_eos_hidden_states(
        model,
        tokenizer,
        input_texts,
        batch_size=BATCH_SIZE,
        layer_index=LAYER_INDEX
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
        title=f"3D PCA of EOS hidden states ({MODEL_NAME}, layer={LAYER_INDEX}, LAST_TOKEN_IS_EOS={LAST_TOKEN_IS_EOS})",
    )
    fig.update_traces(marker=dict(size=6))


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



# =========================
# EOS位置の hidden state を取る関数
# =========================
@torch.no_grad()
def extract_eos_hidden_states(model, tokenizer, text_list, batch_size=8, layer_index=-1):
    """
    各テキストの末尾にEOSを明示的に追加し、
    EOSトークン位置の hidden state を返す。

    Returns:
        np.ndarray of shape (N, hidden_dim)
    """
    all_vecs = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]

        # EOS を明示的に末尾へ追加
        batch_texts_with_eos = [text + tokenizer.eos_token for text in batch_texts]

        inputs = tokenizer(
            batch_texts_with_eos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=LAST_TOKEN_IS_EOS,
        ).to(model.device) 

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states は tuple:
            # 0: embedding出力, 1..L: 各層出力
            hs = outputs.hidden_states
            layer_hs = hs[layer_index]      # (B, T, H)
        
        
        # 各系列について EOS token の最後の出現位置を取る
        eos_mask = (input_ids == tokenizer.eos_token_id)

        for b in range(input_ids.size(0)):
            if LAST_TOKEN_IS_EOS:
                eos_positions = torch.where(eos_mask[b])[0]
                if len(eos_positions) == 0:
                    raise ValueError(f"EOS token が見つかりません: {batch_texts[b]}")
                eos_pos = eos_positions[-1].item()

                vec = layer_hs[b, eos_pos, :]   # (H,)
            else:
                seq_len = attention_mask[b].sum().item()
                vec = layer_hs[b, seq_len - 1, :]  # (H,)

            all_vecs.append(vec.detach().float().cpu().numpy())

    return np.stack(all_vecs, axis=0)




def get_first_few_sentences(text, word_threshold):
    """ textを文に分割して、最初の数文を連結して返す。連結したテキストの単語数がword_thresholdを超えないようにする。
    ~~ただし1文目ですでにword_thresholdを超える場合は、最初の1文だけを返す。~~
    1文目がword_thresholdを超える場合は、Noneを返す
    """
    # 文末記号を保持して分割
    parts = re.split(r'(?<=[。．!?！？])\s*', text)
    sentences = []
    delimiters = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i].strip()
        # delimiter = parts[i + 1]
        delimiters.append(parts[i + 1].strip() if i + 1 < len(parts) else "")
        if sentence:
            sentences.append(sentence)
    
    word_count = 0
    truncated_text = ""
    for i, sentence in enumerate(sentences):
        # word_count += len(sentence)
        word_count += len(sentence.split())  # 連結したテキストの単語数をカウント
        char_count = len(sentence)
        if word_count > word_threshold or char_count > word_threshold * 5:  # 一定の単語数を超えないようにする
            print(f"skip.  word count: {word_count}, char_count: {char_count}")# , sentence: {sentence}")
            break
        truncated_text += sentence + delimiters[i]  # 文末記号を保持して連結
        print(f"word count: {word_count}, char_count: {char_count}")# , sentence: {sentence}")
    
    return truncated_text if truncated_text != "" else None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="proper_nouns", choices=["proper_nouns", "wiki_summary", "wiki_main_text"], help="プロットに使用するデータの種類。固有名詞を使用するか、wiki summaryを使用するか")
    parser.add_argument("--threshold_num_nouns_per_category", type=int, default=30, help="各カテゴリからプロットに使用する固有名詞の最大数。あまりに多いとプロットが見づらくなるため。Noneの場合は全て使用する。")
    parser.add_argument("--num_categories", type=int, default=10, help="プロットに使用するカテゴリの数。最初のnカテゴリを使用する。Noneの場合は全てのカテゴリを使用する。")
    parser.add_argument("--cuda_device", type=str, default="1", help="使用するCUDAデバイスのID (例: '0', '1', '2', ...)")
    args = parser.parse_args()
    main(args)


"""
nohup uv run python3 src_visualize/plot_gemma_eos_hidden_states_3d.py \
    --data_type wiki_summary \
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