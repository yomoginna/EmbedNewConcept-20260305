
import argparse
import json
import os
import sys
import numpy as np
from language_tool_python.__main__ import config_path
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from utils.wikipedia_api_utils import load_wikisummary
from utils.handle_text_utils import get_first_few_sentences, repeat_text, delete_non_English_characters
from utils.gemma_train_and_test_utils import fix_seed,  extract_hidden_states
from utils.handle_data_from_dbpedia_utils import loadProperNounData, filterProperNounsWithWikiPage


wiki_pages_dir = os.path.join(project_root, "data", "wiki_pages")


BATCH_SIZE = 4
propnoun_num_for_init_vec=100   #  初期化vecの作成に使う固有名詞の最低数. 例えば100に設定した場合、各カテゴリで最低100個の固有名詞を使用して初期化vecを作成することになる。(実際には、新規概念用にならなかった固有名詞全て使用する)
propnoun_num_for_new_concept = 50 # 新規概念の元にする概念の作成に使う固有名詞の数. 例えば50に設定した場合、各カテゴリで50個の固有名詞を使用して新規概念の元にする概念の作成に使用することになる。
# dont_get_new_wiki_flag = True # False #True # もう新しいwikiページを読み込みたくない場合はTrue. すでに保存済みのwikiページがあるpropernounのみにフィルタリングする.



def compute_category_similarity_matrix(
    category_to_centroid: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """
    args:
    - category_to_centroid: カテゴリ名 -> セントロイドベクトル

    カテゴリ同士の cosine similarity の辞書を返す。
    sim[a][b] = cosine(category_a, category_b)
    """
    categories = list(category_to_centroid.keys())
    sim = defaultdict(dict)

    for cat_a in categories:
        vec_a = category_to_centroid[cat_a]
        for cat_b in categories:
            vec_b = category_to_centroid[cat_b]
            score = torch.dot(vec_a, vec_b).item()
            sim[cat_a][cat_b] = score

    return dict(sim)


def classify_other_categories(
    similarity_matrix: Dict[str, Dict[str, float]],
    top_k_near: int = 3,
    top_k_far: int = 3,
    allowed_categories: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    args:
    - similarity_matrix: compute_category_similarity_matrix の出力
    - top_k_near: 近い他カテゴリの数
    - top_k_far: 遠い他カテゴリの数
    - allowed_categories: 分類対象とするカテゴリのリスト

    各 own_category に対して
    - near: 近い他カテゴリ
    - far: 遠い他カテゴリ
    - middle: 中間カテゴリ
    を返す。
    """
    result = {}
    allowed_set = set(allowed_categories) if allowed_categories is not None else None

    for own_category, row in similarity_matrix.items():
        candidates = []
        for other_category, score in row.items():
            if other_category == own_category:
                continue
            if allowed_set is not None and other_category not in allowed_set:
                continue
            candidates.append((other_category, score))

        # 類似度が高い順
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

        near = candidates_sorted[:top_k_near]
        far = candidates_sorted[-top_k_far:] if top_k_far > 0 else []

        near_names = {x[0] for x in near}
        far_names = {x[0] for x in far}
        middle = [x for x in candidates_sorted if x[0] not in near_names and x[0] not in far_names]

        result[own_category] = {
            "near": near,
            "middle": middle,
            "far": far,
        }

    return result


def get_propnoun_to_repeatwikisummary(propnouns: List[str], wiki_pages_dir: str, min_words: int, max_words: int) -> Dict[str, str]:
    """固有名詞 -> その固有名詞を説明するwikiのsummary文を2回繰り返したテキスト の辞書を返す
    例:
    {
        "Tokyo": "Tokyo is the capital of Japan. ... (summaryの最初の数文) ... Tokyo is the capital of Japan. ... (summaryの最初の数文) ...",
        "New York": "New York is a city in the United States. ... (summaryの最初の数文) ... New York is a city in the United States. ... (summaryの最初の数文) ...",
        ...
    }
    """
    
    propnoun_to_repeatwikisummary = {}
    for propnoun in propnouns:
        if propnoun.strip() == "":
            continue
        # ** このterm(prop noun)を説明する wiki page の summary を取得し、前処理を行う **
        summary = load_wikisummary(propnoun, wiki_pages_dir)

        # 短すぎor長すぎるsummaryがあるため、最初の数文だけをsummaryとして使用する. (30~300単語に収まるように調整) 30単語未満のsummaryは、十分な情報が得られない可能性があるため、初期化vecの計算に使用しない. 
        summary = get_first_few_sentences(summary, min_words, max_words)
        if summary is None:
            print(f"'{propnoun}' のWikipedia summaryは、{min_words} ~ {max_words}単語の範囲内に収まらないため、スキップします。") # 最初の100文字だけ表示
            # min_words ~ max_wordsの範囲内にないsummaryを持つpropnounは次回もwiki apiで呼び出すことがないよう記録しておく
            with open(os.path.join(project_root, "data", f"propnouns_summary_outofrange_{min_words}_{max_words}.txt"), "a") as f:
                f.write(propnoun + "\n")
            continue

        # 英語数字記号以外の文字を削除
        summary = delete_non_English_characters(summary)

        # *** 初期vecを、固有名詞毎のwikiのsummary文を2回入力して、2回目の文内token位置の隠れ状態から作る場合: https://openreview.net/forum?id=Ahlrf2HGJR の手法 ***
        summary = repeat_text(summary, 2)

        propnoun_to_repeatwikisummary[propnoun] = summary

    return propnoun_to_repeatwikisummary



def main(args):
    
    layer_index = args.layer_index
    min_num_nouns_per_category = args.min_num_nouns_per_category
    max_num_nouns_per_category = args.max_num_nouns_per_category    
    pool_hs_type = 'mean_pool'
    data_type = "wiki_summary_repeat"
    catnum_plus = args.catnum_plus
    min_words, max_words = 30, 300 # 30->50に変更すると、そこまで長いsummaryが少ないようで、init vecが0vecとなりlossがNanになってしまった。minは30でキープする

    if args.model_size == "4b":
        model_name = f"google/gemma-3-4b-it"
    elif args.model_size == "12b":
        model_name = f"google/gemma-3-12b-it"
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device


    propnouns_outofrange_path = os.path.join(project_root, "data", f"propnouns_summary_outofrange_{min_words}_{max_words}.txt")
    if os.path.exists(propnouns_outofrange_path):
        with open(propnouns_outofrange_path, "r") as f:
            propnouns_outofrange = set(line.strip() for line in f)
    else:
        propnouns_outofrange = set()

    config_path = os.path.join(project_root, "config", 'target_concepts.json')
    with open(config_path, "r") as f:
        config_all = json.load(f)
    all_cats = list(config_all.keys())
        
    if args.config_filename == None:
        # config_path = os.path.join(project_root, "config", 'target_concepts.json')
        # with open(config_path, "r") as f:
        #     config = json.load(f)
        # ランダムにcatnum_plusカテゴリを選ぶ
        # all_cats = list(config.keys())
        target_cats = np.random.choice(all_cats, catnum_plus, replace=False)

    else:
        config_path = os.path.join(project_root, "config", args.config_filename + '.json')
        with open(config_path, "r") as f:
            config = json.load(f)    
        target_cats = [cat for cat, propnouns_for_train in config.items() if len(propnouns_for_train) > 0]

        # ランダムにcatnum_plusカテゴリを選び追加する
        target_cats.extend(np.random.choice(list(set(all_cats) - set(target_cats)), catnum_plus, replace=False))

    
    # filter: 全てのカテゴリ・固有名詞リスト の辞書を読み込む (重複等のfiltering済み)
    category_properNouns_dict = loadProperNounData(
        propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept,
        print_flag=False
    )

    # filter: [memo] もう新しいwikiページを読み込みたくない場合は、すでに保存済みのwikiページがあるpropernounのみにフィルタリングする
    if args.dont_get_new_wiki_flag:
        print("Filtering proper nouns to those with already saved wiki pages...")
        filtered_category_properNouns_dict = {}
        for category, propernouns in category_properNouns_dict.items():
            filtered_category_properNouns_dict[category] = filterProperNounsWithWikiPage(propernouns, wiki_pages_dir)
        print(f"concept num: {sum(len(concepts) for concepts in category_properNouns_dict.values())} \
              -> {sum(len(concepts) for concepts in filtered_category_properNouns_dict.values())}")
    else:
        filtered_category_properNouns_dict = category_properNouns_dict
    

    cat_to_input_texts = {}
    for cat in target_cats:
        print('\n')
        propnouns = filtered_category_properNouns_dict.get(cat, [])
        propnouns = list(set(propnouns) - propnouns_outofrange) # min_words ~ max_wordsの範囲内にないsummaryを持つpropnounは次回もwiki apiで呼び出すことがないようフィルタリングする
        if len(propnouns) < min_num_nouns_per_category:
            print(f"カテゴリ '{cat}' は、{min_num_nouns_per_category} 個未満の固有名詞しかないため、分析から除外されます。")
            continue
        
        # if len(propnouns) > max_num_nouns_per_category:
        #     propnouns = np.random.choice(propnouns, max_num_nouns_per_category, replace=False)
        #     print(f"カテゴリ '{cat}' は、{max_num_nouns_per_category} 個を超える固有名詞を持っているため、ランダムにサンプリングされます。")

        # print(f"{cat}: {len(propnouns)} proper nouns. {propnouns[:5]}...")
        # propnoun_to_repeatwikisummary = get_propnoun_to_repeatwikisummary(propnouns, wiki_pages_dir)


        propnoun_to_repeatwikisummary = {}
        while len(propnoun_to_repeatwikisummary) < max_num_nouns_per_category  and  len(propnouns) > 0:
            # 1propnounずつwikisummaryを取得し追加する。max_num_nouns_per_categoryに達するか、propnounsがなくなるまで続ける。
            propnoun = np.random.choice(propnouns, 1)[0]
            propnouns.remove(propnoun)
            # propnoun_to_repeatwikisummaryにget_propnoun_to_repeatwikisummary([propnoun], wiki_pages_dir)の結果を追加
            dic = get_propnoun_to_repeatwikisummary([propnoun], wiki_pages_dir, min_words, max_words)
            if dic is not None and len(dic) > 0:
                propnoun_to_repeatwikisummary.update(dic)
        
        if len(propnoun_to_repeatwikisummary) < min_num_nouns_per_category:
            print(f"カテゴリ '{cat}' は、{min_num_nouns_per_category} 個未満の固有名詞のwiki summaryを取得できたため、分析から除外されます。")
            continue


        cat_to_input_texts[cat] = list(propnoun_to_repeatwikisummary.values())



    # =========================
    # フラット化
    # =========================
    input_texts = []
    categories = []

    for category, input_t_list in cat_to_input_texts.items():
        for input_text in input_t_list:
            input_texts.append(input_text)
            categories.append(category)

    # =========================
    # モデル読み込み
    # =========================
    print(f"Loading model: {model_name} on cuda...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=None
    ).to('cuda')
    model.eval()

    # Gemma系で pad token が未設定な場合の保険
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id が見つかりません。")
    

    
    # =========================
    # 特徴抽出
    # =========================
    all_vecs = extract_hidden_states(
        model,
        tokenizer,
        input_texts,
        pool_hs_type=pool_hs_type,
        data_type=data_type,
        batch_size=BATCH_SIZE,
        layer_index=layer_index,
    )

    print("all_vecs shape:", all_vecs.shape)  # (N, hidden_dim)

    # NaN / inf 除外
    valid_mask = np.isfinite(all_vecs).all(axis=1)
    valid_vecs = all_vecs[valid_mask]
    valid_input_texts = [x for x, ok in zip(input_texts, valid_mask) if ok]
    valid_categories = [x for x, ok in zip(categories, valid_mask) if ok]

    print("valid samples:", len(valid_input_texts), "/", len(input_texts))

    if len(valid_input_texts) < 2:
        raise ValueError("有効サンプルが少な過ぎます。")

    # =========================
    # カテゴリごとに特徴の平均を取る
    # =========================
    category_to_centroid = {}
    for category in set(valid_categories):
        cat_vecs = valid_vecs[np.array(valid_categories) == category]

        # 各固有名詞vecを正規化
        cat_vecs = torch.tensor(cat_vecs, dtype=torch.float)
        cat_vecs = torch.nn.functional.normalize(cat_vecs, dim=1)

        if len(cat_vecs) < min_num_nouns_per_category:
            print(f"カテゴリ '{category}' は、{min_num_nouns_per_category} 個未満の有効なサンプルしかないため、分析から除外されます。")
            continue
        centroid = torch.tensor(cat_vecs).mean(dim=0)
        category_to_centroid[category] = centroid

    # =========================
    # カテゴリ同士の類似度を計算
    # =========================
    similarity_matrix = compute_category_similarity_matrix(category_to_centroid)
    print("Category Similarity Matrix:")
    for cat_a, row in similarity_matrix.items():
        for cat_b, score in row.items():
            print(f"Similarity between '{cat_a}' and '{cat_b}': {score:.4f}")

    # =========================
    # 各カテゴリに対して、近いカテゴリ・遠いカテゴリを分類
    # =========================
    classification = classify_other_categories(similarity_matrix, top_k_near=1, top_k_far=1)
    print("\nCategory Classification:")
    for own_cat, groups in classification.items():
        print(f"Own Category: {own_cat}")
        print("  Near Categories:")
        for cat, score in groups["near"]:
            print(f"    {cat} (similarity: {score:.4f})")
        print("  Middle Categories:")
        for cat, score in groups["middle"]:
            print(f"    {cat} (similarity: {score:.4f})")
        print("  Far Categories:")
        for cat, score in groups["far"]:
            print(f"    {cat} (similarity: {score:.4f})")

    
    # 結果をJSONファイルに保存
    output_path = os.path.join(project_root, "data", "cossim_bw_categories", f"category_similarity_{args.model_size}_{args.config_filename}_catnum_plus_{args.catnum_plus}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "similarity_matrix": similarity_matrix,
            "classification": classification,
        }, f, indent=4)
    print(f"\nResults saved to: {output_path}")

    









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="4b", help="モデルサイズ (例: '3b', '12b')")
    parser.add_argument("--layer_index", type=int, default=-1, help="特徴抽出に使用する層のインデックス (例: -1は最終層)")
    parser.add_argument("--min_num_nouns_per_category", type=int, default=5, help="各カテゴリに必要な固有名詞の最小数。これを下回るカテゴリは分析から除外される。")
    parser.add_argument("--max_num_nouns_per_category", type=int, default=200, help="各カテゴリに使用する固有名詞の最大数。これを超える場合はランダムにサンプリングされる。")
    parser.add_argument("--cuda_device", type=str, default="0", help="使用するCUDAデバイスのID (例: '0', '1', '2', '3')")
    parser.add_argument("--config_filename", type=str, default=None, help="カテゴリと固有名詞の対応を定義したJSONファイルの名前 (例: 'target_concepts_mini_13.json')")
    parser.add_argument("--dont_get_new_wiki_flag", action="store_true", help="新しいwikiページを読み込みたくない場合はこのフラグを立てる。すでに保存済みのwikiページがあるpropernounのみにフィルタリングする。")
    parser.add_argument("--catnum_plus", type=int, default=0, help="ランダムに追加するカテゴリの数。")
    args = parser.parse_args()

    fix_seed(42)
    main(args)


"""

```sh
nohup uv run python src/calc_cossim_between_categories.py \
    --model_size "4b" \
    --layer_index 8 \
    --min_num_nouns_per_category 5 \
    --max_num_nouns_per_category 200 \
    --cuda_device "1" \
    --config_filename "target_concepts_mini_13"\
    --dont_get_new_wiki_flag
    > logs/calc_cossim_between_categories_4b_layer8_min5_max200.log 2>&1 &
```


```sh
nohup uv run python src/calc_cossim_between_categories.py \
    --model_size "4b" \
    --layer_index 8 \
    --min_num_nouns_per_category 5 \
    --max_num_nouns_per_category 100 \
    --cuda_device "1" \
    --config_filename "target_concepts_mini_13"\
    --dont_get_new_wiki_flag \
    > logs/calc_cossim_between_categories_4b_layer8_min5_max100.log 2>&1 &
```


```sh
nohup uv run python src/calc_cossim_between_categories.py \
    --model_size "12b" \
    --layer_index 12 \
    --min_num_nouns_per_category 5 \
    --max_num_nouns_per_category 50 \
    --cuda_device "3" \
    > logs/calc_cossim_between_categories_12b_layer12_min5_max50.log 2>&1 &
```
1303288


```sh
nohup uv run python src/calc_cossim_between_categories.py \
    --model_size "12b" \
    --layer_index 12 \
    --min_num_nouns_per_category 5 \
    --max_num_nouns_per_category 50 \
    --cuda_device "4" \
    --config_filename "target_concepts_mini_13"\
    --catnum_plus 40 \
    > logs/calc_cossim_between_categories_12b_layer12_min5_max50_catnum_plus40.log 2>&1 &
```
nvidia-smi

```sh
nohup uv run python src/calc_cossim_between_categories.py \
    --model_size "4b" \
    --layer_index 12 \
    --min_num_nouns_per_category 5 \
    --max_num_nouns_per_category 50 \
    --cuda_device "3" \
    --config_filename "target_concepts_mini_13"\
    --catnum_plus 40 \
    > logs/calc_cossim_between_categories_4b_layer12_min5_max50_catnum_plus40.log 2>&1 &
```
44780

"""