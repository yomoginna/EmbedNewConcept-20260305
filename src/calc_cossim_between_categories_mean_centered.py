
import argparse
import json
import os
import sys
import numpy as np
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


BATCH_SIZE = 2
propnoun_num_for_init_vec=100   #  初期化vecの作成に使う固有名詞の最低数. 例えば100に設定した場合、各カテゴリで最低100個の固有名詞を使用して初期化vecを作成することになる。(実際には、新規概念用にならなかった固有名詞全て使用する)
propnoun_num_for_new_concept = 50 # 新規概念の元にする概念の作成に使う固有名詞の数. 例えば50に設定した場合、各カテゴリで50個の固有名詞を使用して新規概念の元にする概念の作成に使用することになる。
# dont_get_new_wiki_flag = True # False #True # もう新しいwikiページを読み込みたくない場合はTrue. すでに保存済みのwikiページがあるpropernounのみにフィルタリングする.

delete_categories = ["artery"]

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



def fit_whitening_transform(
    X: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: (N, D)
    return:
        mean: (1, D)
        W   : (D, D)
    whitening後に共分散がほぼ単位行列になるような線形変換を返す
    """
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean

    cov = (X_centered.T @ X_centered) / max(X_centered.shape[0] - 1, 1)

    # 対称行列なので eigh を使う
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # 数値安定化
    eigvals = torch.clamp(eigvals, min=eps)

    # W = U diag(1/sqrt(lambda)) U^T
    W = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
    return mean, W


def apply_whitening_transform(
    X: torch.Tensor,
    mean: torch.Tensor,
    W: torch.Tensor
) -> torch.Tensor:
    return (X - mean) @ W


def whiten_embeddings(
    vecs: np.ndarray,
    eps: float = 1e-5,
    l2_normalize_after: bool = True
) -> np.ndarray:
    """
    vecs: (N, D) numpy array
    """
    X = torch.tensor(vecs, dtype=torch.float)

    mean, W = fit_whitening_transform(X, eps=eps)
    X_white = apply_whitening_transform(X, mean, W)

    if l2_normalize_after:
        X_white = torch.nn.functional.normalize(X_white, dim=1)

    return X_white.cpu().numpy()









def main(args):
    print(f"Running with seed: {args.seed}")
    fix_seed(args.seed)
    
    layer_index = args.layer_index
    num_nouns_per_category = args.num_nouns_per_category    
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

   

    # * 全てのカテゴリ・固有名詞リスト の辞書を読み込む (重複等のfiltering済み)
    category_properNouns_dict = loadProperNounData(
        propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept,
        print_flag=False
    )

    # filter: delete_categoriesに含まれるカテゴリをcategory_properNouns_dictから削除する
    for delete_cat in delete_categories:
        if delete_cat in category_properNouns_dict:
            del category_properNouns_dict[delete_cat]
            print(f"Deleted category '{delete_cat}' from category_properNouns_dict.")


    # filter: もう新しいwikiページを読み込みたくない場合は、すでに保存済みのwikiページがあるpropernounのみにフィルタリングする
    if args.dont_get_new_wiki_flag:
        print("Filtering proper nouns to those with already saved wiki pages...")
        filtered_category_properNouns_dict = {}
        for category, propernouns in category_properNouns_dict.items():
            filtered_propnouns = filterProperNounsWithWikiPage(propernouns, wiki_pages_dir)
            if len(filtered_propnouns) >= num_nouns_per_category: # if len(filtered_propnouns) >= min_num_nouns_per_category:
                filtered_category_properNouns_dict[category] = filtered_propnouns
        print(f"concept num: {sum(len(concepts) for concepts in category_properNouns_dict.values())} \
              -> {sum(len(concepts) for concepts in filtered_category_properNouns_dict.values())}")
        all_cats = list(filtered_category_properNouns_dict.keys())  # wiki pageのあるカテゴリのみ対象カテゴリとする
    else:
        # 新しいwikiページを取得してもいい場合は、config内の全てのカテゴリを対象カテゴリとする
        filtered_category_properNouns_dict = category_properNouns_dict
        config_path = os.path.join(project_root, "config", 'target_concepts.json')
        with open(config_path, "r") as f:
            config_all = json.load(f)
        all_cats = list(config_all.keys())
    print(f"カテゴリ選出対象: {len(all_cats)} categories. {all_cats[:5]}...")
         
    # 対象カテゴリの選択
    if catnum_plus is None:
        # # 選出されるカテゴリを固定化する。
        # with open(os.path.join(project_root, "config", "fix_categories.json"), "r") as f:
        #     target_cats = json.load(f).keys()
        target_cats = all_cats

        # ** 確認用
        config_path = os.path.join(project_root, "config", args.config_filename + '.json')
        with open(config_path, "r") as f:
            config = json.load(f)   
        train_target_cats = [cat for cat, propnouns_for_train in config.items() if len(propnouns_for_train) > 0] 
        for cat in train_target_cats:
            if cat not in target_cats:
                print(f"Warning: config内のカテゴリ '{cat}' は、選出されたカテゴリの中にありません。")
        # **
        
    else:
        if args.config_filename == None:
            # ランダムにcatnum_plusカテゴリを選ぶ
            if len(all_cats) < catnum_plus:
                raise ValueError(f"catnum_plus ({catnum_plus}) is larger than the number of available categories ({len(all_cats)}).")
            target_cats = np.random.choice(all_cats, catnum_plus, replace=False)

        else:
            config_path = os.path.join(project_root, "config", args.config_filename + '.json')
            with open(config_path, "r") as f:
                config = json.load(f)    
            target_cats = [cat for cat, propnouns_for_train in config.items() if len(propnouns_for_train) > 0]

            # ランダムにcatnum_plusカテゴリを選び追加する
            if len(all_cats) - len(target_cats) < catnum_plus:
                raise ValueError(f"catnum_plus ({catnum_plus}) is larger than the number of available categories to add ({len(all_cats) - len(target_cats)}).")
            target_cats.extend(np.random.choice(list(set(all_cats) - set(target_cats)), catnum_plus, replace=False))
    
    print(f"Selected {len(target_cats)} target categories: {target_cats[:5]}...")

    cat_to_input_texts = {}
    for cat in target_cats:
        print('\n')
        propnouns = filtered_category_properNouns_dict.get(cat, [])
        print(f"Category '{cat}' has {len(propnouns)} proper nouns before filtering with wiki page availability and summary length.")
        propnouns = list(set(propnouns) - propnouns_outofrange) # min_words ~ max_wordsの範囲内にないsummaryを持つpropnounは次回もwiki apiで呼び出すことがないようフィルタリングする
        print(f"\t -> {len(propnouns)} left after delete proper nouns which has summary out of range.")

        if len(propnouns) < num_nouns_per_category:
            print(f"カテゴリ '{cat}' は、{num_nouns_per_category} 個未満の固有名詞しかないため、分析から除外されます。")
            continue
        

        propnoun_to_repeatwikisummary = {}
        while len(propnoun_to_repeatwikisummary) < num_nouns_per_category  and  len(propnouns) > 0:
            # 1propnounずつwikisummaryを取得し追加する。num_nouns_per_categoryに達するか、propnounsがなくなるまで続ける。
            propnoun = np.random.choice(propnouns, 1)[0]
            propnouns.remove(propnoun)
            # propnoun_to_repeatwikisummaryにget_propnoun_to_repeatwikisummary([propnoun], wiki_pages_dir)の結果を追加
            dic = get_propnoun_to_repeatwikisummary([propnoun], wiki_pages_dir, min_words, max_words)
            if dic is not None and len(dic) > 0:
                propnoun_to_repeatwikisummary.update(dic)
        
        if len(propnoun_to_repeatwikisummary) < num_nouns_per_category:
            print(f"カテゴリ '{cat}' は、{num_nouns_per_category} 個未満の固有名詞のwiki summaryしか取得できなかったため、分析から除外されます。")
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
        dtype=torch.float32,  # torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
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
    # Mean centering
    # whiteningが強過ぎて意味が消えてしまったのでこちら
    # =========================
    # if args.use_mean_centering:
    print("Applying mean centering to valid_vecs...")
    X = torch.tensor(valid_vecs, dtype=torch.float)
    X = X - X.mean(dim=0, keepdim=True)
    valid_vecs = X.cpu().numpy()
    print("Mean centering done.")


    # =========================
    # カテゴリごとに特徴の平均を取る
    # =========================
    category_to_centroid = {}
    for category in set(valid_categories):
        cat_vecs = valid_vecs[np.array(valid_categories) == category]

        # 各固有名詞vecを正規化
        cat_vecs = torch.tensor(cat_vecs, dtype=torch.float)
        cat_vecs = torch.nn.functional.normalize(cat_vecs, dim=1)

        if len(cat_vecs) < num_nouns_per_category: # if len(cat_vecs) < min_num_nouns_per_category:
            print(f"カテゴリ '{category}' は、{num_nouns_per_category} 個未満の有効なサンプルしかないため、分析から除外されます。")
            continue
        centroid = torch.tensor(cat_vecs).mean(dim=0)
        # さらにセントロイドを正規化
        centroid = torch.nn.functional.normalize(centroid, dim=0)
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
    if args.catnum_plus is None:
        filename = f"category_similarity_mean_centered_{args.model_size}_{args.config_filename}_seed{args.seed}.json"
    else:
        filename = f"category_similarity_mean_centered_{args.model_size}_{args.config_filename}_catnum_plus_{args.catnum_plus}_seed{args.seed}.json"
        
    output_path = os.path.join(project_root, "data", "cossim_bw_categories", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "similarity_matrix": similarity_matrix,
            "classification": classification,
        }, f, indent=4)
    print(f"\nResults saved to: {output_path}")

    


from collections import defaultdict, Counter
import os
import json
import math
import re

def aggregate_results_and_analyze(args):
    # ここで、複数のseedの結果を集約して分析する


    all_seed_results = []
    all_categories = set()

    # =========================
    # 各seedの結果を読み込む
    # =========================
    # for seed in range(args.seed_range):
    # for seed in args.seed_list:
        # if args.catnum_plus is None:
        #     filename = f"category_similarity_mean_centered_{args.model_size}_{args.config_filename}_seed{args.seed}.json"
        # else:
        #     filename = f"category_similarity_mean_centered_{args.model_size}_{args.config_filename}_catnum_plus_{args.catnum_plus}_seed{args.seed}.json"
        # path = os.path.join(project_root, "data", "cossim_bw_categories", filename)

    dir_pattern1 = f"category_similarity_mean_centered_{args.model_size}_{args.config_filename}_seed(\\d+).json"
    dir_pattern2 = f"category_similarity_mean_centered_{args.model_size}_{args.config_filename}_catnum_plus_{args.catnum_plus}_seed(\\d+).json"

    for filename in os.listdir(os.path.join(project_root, "data", "cossim_bw_categories")):
        if re.match(dir_pattern1, filename):
            seed = int(re.match(dir_pattern1, filename).group(1))
        elif re.match(dir_pattern2, filename):
            seed = int(re.match(dir_pattern2, filename).group(1))
        else:
            continue

        path = os.path.join(project_root, "data", "cossim_bw_categories", filename)
        with open(path, "r") as f:
            data = json.load(f)

        all_seed_results.append({
            "seed": seed,
            "similarity_matrix": data["similarity_matrix"],
            "classification": data["classification"],
        })

        all_categories.update(data["classification"].keys())

    all_categories = sorted(all_categories)

    # =========================
    # 集計用データ構造
    # =========================
    # own_cat -> near/far に出たカテゴリの出現回数
    near_counter_per_cat = defaultdict(Counter)
    far_counter_per_cat = defaultdict(Counter)

    # own_cat -> other_cat -> 類似度リスト
    sim_values_per_pair = defaultdict(lambda: defaultdict(list))

    # own_cat -> seedごとの near/far
    near_list_per_cat = defaultdict(list)
    far_list_per_cat = defaultdict(list)

    # =========================
    # 集計
    # =========================
    for result in all_seed_results:
        seed = result["seed"]
        similarity_matrix = result["similarity_matrix"]
        classification = result["classification"]

        for own_cat, groups in classification.items():
            # near
            near_items = groups.get("near", [])
            for other_cat, score in near_items:
                near_counter_per_cat[own_cat][other_cat] += 1
                near_list_per_cat[own_cat].append({
                    "seed": seed,
                    "category": other_cat,
                    "score": score,
                })

            # far
            far_items = groups.get("far", [])
            for other_cat, score in far_items:
                far_counter_per_cat[own_cat][other_cat] += 1
                far_list_per_cat[own_cat].append({
                    "seed": seed,
                    "category": other_cat,
                    "score": score,
                })

            # similarity 全体
            if own_cat in similarity_matrix:
                for other_cat, score in similarity_matrix[own_cat].items():
                    if other_cat == own_cat:
                        continue
                    sim_values_per_pair[own_cat][other_cat].append(score)

    # =========================
    # 統計量計算
    # =========================
    def calc_mean(xs):
        return sum(xs) / len(xs) if xs else None

    def calc_std(xs):
        if len(xs) <= 1:
            return 0.0
        m = calc_mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    summary_per_cat = {}

    for own_cat in all_categories:
        near_summary = []
        far_summary = []
        sim_summary = []

        # そのカテゴリに対する全候補カテゴリ
        candidate_cats = set(sim_values_per_pair[own_cat].keys()) \
            | set(near_counter_per_cat[own_cat].keys()) \
            | set(far_counter_per_cat[own_cat].keys())

        for other_cat in sorted(candidate_cats):
            sim_list = sim_values_per_pair[own_cat][other_cat]
            near_count = near_counter_per_cat[own_cat][other_cat]
            far_count = far_counter_per_cat[own_cat][other_cat]

            item = {
                "category": other_cat,
                "mean_similarity": calc_mean(sim_list),
                "std_similarity": calc_std(sim_list),
                "num_observed": len(sim_list),
                "near_count": near_count,
                "far_count": far_count,
            }
            sim_summary.append(item)

            if near_count > 0:
                near_summary.append(item)
            if far_count > 0:
                far_summary.append(item)

        # near は near_count 多い順、同率なら mean_similarity 高い順
        near_summary = sorted(
            near_summary,
            key=lambda x: (-x["near_count"], -(x["mean_similarity"] if x["mean_similarity"] is not None else -999))
        )

        # far は far_count 多い順、同率なら mean_similarity 低い順
        far_summary = sorted(
            far_summary,
            key=lambda x: (-x["far_count"], (x["mean_similarity"] if x["mean_similarity"] is not None else 999))
        )

        # 全ペアの平均類似度ランキング
        sim_high_to_low = sorted(
            sim_summary,
            key=lambda x: -(x["mean_similarity"] if x["mean_similarity"] is not None else -999)
        )
        sim_low_to_high = sorted(
            sim_summary,
            key=lambda x: (x["mean_similarity"] if x["mean_similarity"] is not None else 999)
        )

        summary_per_cat[own_cat] = {
            "near_candidates_ranked": near_summary,
            "far_candidates_ranked": far_summary,
            "most_similar_by_mean": sim_high_to_low[:10],
            "least_similar_by_mean": sim_low_to_high[:10],
            "near_by_seed": near_list_per_cat[own_cat],
            "far_by_seed": far_list_per_cat[own_cat],
        }

    # =========================
    # 表示
    # =========================
    for own_cat in all_categories:
        print("=" * 80)
        print(f"OWN CATEGORY: {own_cat}")

        print("\n[Near に現れたカテゴリランキング]")
        for item in summary_per_cat[own_cat]["near_candidates_ranked"][:10]:
            print(
                f"  {item['category']:<35} "
                f"near_count={item['near_count']}  "
                f"mean_sim={item['mean_similarity']:.4f}  "
                f"std={item['std_similarity']:.4f}  "
                f"far_count={item['far_count']}"
            )

        print("\n[Far に現れたカテゴリランキング]")
        for item in summary_per_cat[own_cat]["far_candidates_ranked"][:10]:
            print(
                f"  {item['category']:<35} "
                f"far_count={item['far_count']}  "
                f"mean_sim={item['mean_similarity']:.4f}  "
                f"std={item['std_similarity']:.4f}  "
                f"near_count={item['near_count']}"
            )

        print("\n[平均類似度が高いカテゴリ TOP10]")
        for item in summary_per_cat[own_cat]["most_similar_by_mean"]:
            print(
                f"  {item['category']:<35} "
                f"mean_sim={item['mean_similarity']:.4f}  "
                f"std={item['std_similarity']:.4f}  "
                f"near_count={item['near_count']}  "
                f"far_count={item['far_count']}"
            )

        print("\n[平均類似度が低いカテゴリ TOP10]")
        for item in summary_per_cat[own_cat]["least_similar_by_mean"]:
            print(
                f"  {item['category']:<35} "
                f"mean_sim={item['mean_similarity']:.4f}  "
                f"std={item['std_similarity']:.4f}  "
                f"near_count={item['near_count']}  "
                f"far_count={item['far_count']}"
            )

    # =========================
    # JSON保存
    # =========================
    output_path = os.path.join(
        project_root, "data", "cossim_bw_categories",
        "aggregated_near_far_analysis_across_seeds.json"
    )

    with open(output_path, "w") as f:
        json.dump(summary_per_cat, f, indent=4)

    print("\nSaved aggregated result to:", output_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="4b", help="モデルサイズ (例: '3b', '12b')")
    parser.add_argument("--layer_index", type=int, default=-1, help="特徴抽出に使用する層のインデックス (例: -1は最終層)")
    parser.add_argument("--min_num_nouns_per_category", type=int, default=5, help="各カテゴリに必要な固有名詞の最小数。これを下回るカテゴリは分析から除外される。")
    parser.add_argument("--num_nouns_per_category", type=int, default=200, help="各カテゴリに使用する固有名詞の最大数。これを超える場合はランダムにサンプリングされる。")
    parser.add_argument("--cuda_device", type=str, default="0", help="使用するCUDAデバイスのID (例: '0', '1', '2', '3')")
    parser.add_argument("--config_filename", type=str, default=None, help="カテゴリと固有名詞の対応を定義したJSONファイルの名前 (例: 'target_concepts_mini_13.json')")
    parser.add_argument("--dont_get_new_wiki_flag", action="store_true", help="新しいwikiページを読み込みたくない場合はこのフラグを立てる。すでに保存済みのwikiページがあるpropernounのみにフィルタリングする。")
    parser.add_argument("--catnum_plus", type=int, default=None, help="ランダムに追加するカテゴリの数。")
    # parser.add_argument("--seed_range", type=int, default=5, help="実験の乱数シードの範囲。")
    # parser.add_argument("--seed", type=int, default=0, help="実験の乱数シード。")
    parser.add_argument("--seed_list", type=int, nargs="*", default=[0], help="実験の乱数シードのリスト。")
    args = parser.parse_args()

    # # for seed in range(args.seed_range):
    # for seed in args.seed_list:
    #     args.seed = seed
    #     main(args)
    
    # main(args)

    # 結果まとめ
    aggregate_results_and_analyze(args)




"""

```sh
nohup uv run python src/calc_cossim_between_categories_with_whitening.py \
    --model_size "12b" \
    --layer_index 12 \
    --num_nouns_per_category 50 \
    --cuda_device "4" \
    --config_filename "target_concepts_mini_13"\
    --catnum_plus 20 \
    --dont_get_new_wiki_flag \
    > logs/calc_cossim_between_categories_12b_layer12_prop50_catnum_plus20.log 2>&1 &
```
2772843

```sh
CUDA_VISIBLE_DEVICES=1
SEED_LIST=(0 4 8)

CUDA_VISIBLE_DEVICES=2
SEED_LIST=(1 5 9)

CUDA_VISIBLE_DEVICES=3
SEED_LIST=(2 6 10)

CUDA_VISIBLE_DEVICES=4
SEED_LIST=(3 7 11)

nohup uv run python src/calc_cossim_between_categories_mean_centered.py \
    --model_size "12b" \
    --layer_index 12 \
    --num_nouns_per_category 100 \
    --cuda_device ${CUDA_VISIBLE_DEVICES} \
    --config_filename "target_concepts_mini_13"\
    --dont_get_new_wiki_flag \
    --seed_list ${SEED_LIST} \
    > logs/calc_cossim_between_categories_12b_layer12_prop100_cuda${CUDA_VISIBLE_DEVICES}.log 2>&1 &
```
3233014
3234041
3235531
3236303


"""