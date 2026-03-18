
# ===== Standard library =====
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import os
import random
import re
import sys
import time

# ===== Third-party =====
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from plotnine import labels
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
# from transformers import get_cosine_schedule_with_warmup, get_cosine_with_min_lr_schedule_with_warmup
import wandb

# ===== Runtime config =====
transformers_logging.set_verbosity_error()

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
# project_root = os.environ["HOME"] # [memo] genkaiを使う場合. "/singularity_home/project/EmbedNewConcept/src/trainMemVec_fromXvec_gemma.py"
sys.path.append(project_root)
print("Project root:", project_root)

from utils.gemma_train_and_test_utils import fix_seed
from utils.handle_data_from_dbpedia_utils import loadProperNounData, loadConceptsForFictConcept


# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # [memo] genkaiを使う時はコメントアウト!! -> 今はargsで指定している。argsを指定しなければ、CUDAについては何も指定しない。
n_feat_in_a_sample = 3  # 学習データの1サンプル = summary(wiki中の本文 or summary, 今回はsummaryを使用) + n_feat_in_a_sample個の特徴文
propnoun_num_for_init_vec=100   #  初期化vecの作成に使う固有名詞の最低数. 例えば100に設定した場合、各カテゴリで最低100個の固有名詞を使用して初期化vecを作成することになる。(実際には、新規概念用にならなかった固有名詞全て使用する)
propnoun_num_for_new_concept = 50 # 新規概念の元にする概念の作成に使う固有名詞の数. 例えば50に設定した場合、各カテゴリで50個の固有名詞を使用して新規概念の元にする概念の作成に使用することになる。

global BATCH_SIZE


# 環境変数読み込み
load_dotenv(os.path.join(project_root, ".env"))
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# from huggingface_hub import login
# access_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
# login(access_token)



# *************************** utils **************************
# wandbの設定
def set_wandb_env(PROJECT_NAME, wandb_dir):
    """wandbのloginと、学習結果記録に必要な設定を行う関数.
    これらはimport wandbの前に設定する必要がある.
    Args:
        - PROJECT_NAME: Wandbのプロジェクト名
        - wandb_dir: 学習結果記録を格納するディレクトリ
    """
    print(f"Wandb Project name: {PROJECT_NAME}")
    os.makedirs(wandb_dir, exist_ok=True)

    tmp_root_path = os.path.join(project_root, "tmp")
    os.makedirs(tmp_root_path, exist_ok=True)

    os.environ["WANDB_MODE"] = "online"  # "disabled", "offline", "online"
    os.environ['TMPDIR'] = tmp_root_path
    os.environ['TEMP'] = tmp_root_path
    os.environ['TMP'] = tmp_root_path
    os.environ['DATA_DIR'] = tmp_root_path
    os.environ['ARTIFACT_DIR'] = tmp_root_path
    os.environ["WANDB_DATA_DIR"] = os.path.join(wandb_dir, ".wandb_data")

    os.environ['WANDB_DIR'] = wandb_dir
    os.environ['WANDB_CACHE_DIR'] = os.path.join(wandb_dir, ".wandb_cache")
    os.environ['WANDB_CONFIG_DIR'] = os.path.join(wandb_dir, ".wandb_config")
    os.environ['DATA_DIR'] = wandb_dir


    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    wandb_instance = None
    wandb_id = wandb.util.generate_id()  # 一意のIDを作成（再利用するため）
    run_name = f"TrainMemVec_{model_name_for_dirname}"
    wandb_instance = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        id=wandb_id,  # 一意のID
        resume="allow",
    )
    wandb_url = wandb_instance.get_url()
    print("WandB URL:", wandb_url)


def save_mem_vec(model, memTokenIds, mem_save_path):
    os.makedirs(os.path.dirname(mem_save_path), exist_ok=True)

    # memTokenIdsの指定した要素の値(=weightのindex)の並び順のまま取り出され, 勝手にソートされることは無い
    try:
        vecs = (
            model.model.embed_tokens.weight[memTokenIds]
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        )
    except AttributeError:
        vecs = (
            model.model.language_model.embed_tokens.weight[memTokenIds]
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        )
    np.save(mem_save_path, vecs)







# *************************** func ***************************

def initializeEmbed(
        model, 
        tokenizer, 
        train_token2tokenid, 
        init_vec_type, 
        category_to_concepts_for_vec, 
        category2initoken_ids, 
        print_flag=False
    ):
    """言語モデルが持つ「語彙 → ベクトル」への変換をids指定のもののみ初期化する
    行数 = 語彙サイズ、列数 = 埋め込み次元。
    trainTokenIds で指定された行（＝特定のトークンに対応するベクトル）だけを操作する．
    * どのモデルでも共通のはず

    Args:
        model: HuggingFaceのモデルオブジェクト
        tokenizer: HuggingFaceのトークナイザオブジェクト
        train_token2tokenid: 学習対象とする特殊トークンとそのtoken_idのmap
        init_vec_type: memory vectorの初期化方法。zeroまたはuniform, または語句. zero->0vec, uniform->一様分布, 語句->指定の語句の埋め込みベクトルで初期化, 数字->指定のコサイン類似度で近い語句のベクトルで初期化
        category_to_concepts_for_vec: カテゴリごとのvec初期化に使用する概念のリスト。init_vec_typeが 'category_COG' の場合に使用
        category2initoken_ids: カテゴリごとの初期化トークンIDのリスト。init_vec_typeが 'category_COG' の場合に使用
        print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
    """    
    # もしinit_vec_typeを数字に変換できる場合は変換する
    try:
        init_vec_type = float(init_vec_type)
    except:
        pass

    # ①
    if init_vec_type == 'uniform':
        # ** 一様分布で初期化 **
        trainTokenIds = list(train_token2tokenid.values())
        try:
            W = model.model.embed_tokens.weight
        except:
            W = model.model.language_model.embed_tokens.weight
        with torch.no_grad():
            idx = torch.as_tensor(trainTokenIds, device=W.device, dtype=torch.long)
            src = torch.empty((len(trainTokenIds), W.shape[1]), device=W.device, dtype=W.dtype)
            src.uniform_(-0.1, 0.1)
            W.index_copy_(0, idx, src)
        
        return model
    

    elif init_vec_type == 'norm_rand':
        # ** ノルム固定の正規化ランダムで初期化 **
        # 各トークン埋め込みベクトルの「方向」はランダム、L2ノルムは一定（例: 0.1）に揃える
        target_norm = 0.1
        eps = 1e-12

        trainTokenIds = list(train_token2tokenid.values())

        # 埋め込み重みへの参照（モデル差異に対応）
        try:
            W = model.model.embed_tokens.weight
        except:
            W = model.model.language_model.embed_tokens.weight

        dim = W.shape[1]
        device = W.device
        dtype = W.dtype

        # ランダム方向（正規分布）
        rand = torch.randn((len(trainTokenIds), dim), device=device, dtype=dtype)

        # L2正規化（各行のノルムを1に）
        rand = rand / (rand.norm(p=2, dim=1, keepdim=True).clamp_min(eps))

        # ノルムを target_norm に拡張して揃える
        rand = rand * target_norm

        # 指定トークン行だけ書き換え
        with torch.no_grad():
            W[trainTokenIds].copy_(rand)

        return model


    elif init_vec_type == 'norm_rand_vocab':
        # ** 正規化ランダムで初期化（N(μ,σ2)のμとσは語彙集合から計算） **
        trainTokenIds = list(train_token2tokenid.values())
        # 埋め込み重み（V x d）への参照（モデル差異に対応）
        try:
            W = model.model.embed_tokens.weight
        except:
            W = model.model.language_model.embed_tokens.weight
        
        # 全要素をまとめて（スカラー1つの平均・標準偏差）
        mu = W.mean().item()
        sigma = W.std(unbiased=False).item()  # ddof=0（母標準偏差）
        print(f"Vocabulary embedding mean: {mu:.4f}, std: {sigma:.4f}")

        rand = torch.normal(mean=mu, std=sigma, size=(len(trainTokenIds), W.shape[1]), device=W.device, dtype=W.dtype)  # (V, d)

        # 指定トークン行だけ書き換え
        with torch.no_grad():
            W[trainTokenIds].copy_(rand)
        return model

    elif init_vec_type == 'zero':
        # ** 0vecで初期化 **
        trainTokenIds = list(train_token2tokenid.values())
        
        try:
            W = model.model.embed_tokens.weight
        except:
            W = model.model.language_model.embed_tokens.weight

        with torch.no_grad():
            trainTokenIds = torch.as_tensor(trainTokenIds, device=W.device, dtype=torch.long)
            W.index_fill_(0, trainTokenIds, 0.0)
        return model
    
    elif init_vec_type == 'category_centroid_plus_random':
        # *** カテゴリ内の90(propnoun_num_for_init_vec-10)固有名詞(固定)を平均化したvec + カテゴリ内の10固有名詞をランダムに選んで平均化したvec を足し合わせたvecで初期化 ***
        # category_COGではカテゴリ内の初期化vec同士に差がなく、性能が上がらなかったため、
        # 中心vecはカテゴリ内で共通させつつ、そこにカテゴリ内の固有名詞の中からランダムに選んだ10個のvecの平均を足し合わせることで、カテゴリ内の初期化vec同士に差をつけてみる方法

        for category, init_token_ids in category2initoken_ids.items():
            # *** このカテゴリに対応する初期化vec作成用の固有名詞リストで初期化vecを作成し、
            # このカテゴリに属す固有名詞(新規概念用)に割り当てたtokenのtoken idsの行を、その初期化vecで初期化する ***
            init_terms_candidate = category_to_concepts_for_vec[category]
            init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), propnoun_num_for_init_vec-10))  # カテゴリ内の固有名詞からランダムに90個選んで中心vecを作成

            # 初期化対象の追加token毎に、10件をランダム選出してmodel
            for init_token_id in init_token_ids:
                init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) # カテゴリ内の固有名詞からランダムに10個選んでランダムvecを作成
                init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用
                    
                # 埋め込み層のinit_token_idsに該当する行を、init_termsのtokenベクトルの平均で初期化する
                model = initVecWithTokenVec(model, tokenizer, init_terms, [init_token_id], print_flag=print_flag)
                print(f"Initialized category '{category}' (new token {tokenizer.decode(init_token_id)}, token_id: {init_token_id}) with {len(init_terms)} concepts: ... ({init_terms[-15:]}).")

        return model
    
    elif init_vec_type == 'category_COG': 
        # *** COG: Center Of Gravity. 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
        
        for category, init_token_ids in category2initoken_ids.items():
            # *** このカテゴリに対応する初期化vec作成用の固有名詞リストで初期化vecを作成し、
            # このカテゴリに属す固有名詞(新規概念用)に割り当てたtokenのtoken idsの行を、その初期化vecで初期化する ***
            init_terms = category_to_concepts_for_vec[category]
            # 埋め込み層のinit_token_idsに該当する行を、init_termsのtokenベクトルの平均で初期化する
            model = initVecWithTokenVec_with_noise(model, tokenizer, init_terms, init_token_ids, print_flag=print_flag)
            print(f"Initialized category '{category}' ({len(init_token_ids)} new tokens) with {len(init_terms)} concepts ({init_terms[:5]}...) for token {[tokenizer.decode(tid) for tid in init_token_ids[:5]]}... .")

        return model

    elif init_vec_type == 'other_category_COG':
        # *** 他のカテゴリのCOGで初期化. 例えば、動物カテゴリの新規概念を、場所カテゴリの固有名詞のベクトルの平均で初期化するなど. category_COGに対する比較用 ***
        # どのカテゴリのCOGで初期化するかは、初期化対象token毎に毎回ランダムに選ぶ 
        # (category_COGでは同一カテゴリを同じCOGで初期化していたのに対し、こちらは毎回ランダムに選ぶため、同一カテゴリ内でもtoken毎に異なるCOGで初期化されることになる)
        # (なるべく色々なカテゴリのCOGで初期化するため、ランダムに選ぶ方式にしている)
        for category, init_token_ids in category2initoken_ids.items():
            other_categories = [c for c in category_to_concepts_for_vec.keys() if c != category]
            print(f"Category '{category}' will be initialized with COG of other categories: {other_categories[:5]}.")
            for init_token_id in init_token_ids:
                # それぞれのtoken_idを、毎回ランダムに選んだ他のカテゴリのCOGで初期化する
                other_category = random.choice(other_categories) # 他のカテゴリをランダムに選ぶ
                init_terms = category_to_concepts_for_vec[other_category] # 他のカテゴリの概念リストを初期化vec作成に使用
                model = initVecWithTokenVec_with_noise(model, tokenizer, init_terms, [init_token_id], print_flag=print_flag)
                # if print_flag:
                print(f"Initialized category '{category}' (new token {init_token_id}) with {len(init_terms)} concepts ({init_terms[:5]}...) from other category '{other_category}' for token {[tokenizer.decode(tid) for tid in init_token_ids[:5]]}... .")
        return model

    else: 
        # ** 指定の語句で初期化 (句の場合は単純にmean poolingする) **
        init_terms = [init_vec_type]  # 'a chair' など
        model = initVecWithTokenVec(model, tokenizer, init_terms, trainTokenIds, print_flag=print_flag)
        return model




def initVecWithTokenVec(model, tokenizer, init_terms, init_target_ids, print_flag=False):
    """
    埋め込み層の特定の行を、指定の語句のベクトルで初期化する関数.
    Args:
    - model: HuggingFaceのモデルオブジェクト
    - tokenizer: HuggingFaceのトークナイザオブジェクト
    - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
    - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
    - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか

    Returns:
    - model: 対象の行が初期化されたモデルオブジェクト

    memo:
    - 全てのtermのtokenを集めて、全tokenの平均を一括で計算すると、token 数が多い句ほど重みが大きくなり、カテゴリの重心が計算できなくなる（test結果も悪かった）
        → term毎にtokenの平均を計算してから、termの平均を取る方法に変更する
    """
    # ** 指定の語句で初期化 (句の場合は単純にその句をtokenizeした結果をmean poolingしてterm平均vecとする) **
    try:
        E = model.model.embed_tokens.weight  # (vocab, d)
    except:
        E = model.model.language_model.embed_tokens.weight  # (vocab, d)

    STOPWORDS = {
        "a", "an", "the", "of", "in", "on", "at", "for", "to", "and", "or", "with"
    }
    def remove_stopwords_from_text(text):
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return " ".join(filtered)

    # 1. term毎に平均vecを計算してから加算
    sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
    for term in init_terms:
        term = remove_stopwords_from_text(term)                         # aやtheなどの重要度の低い単語をtermから除去
        if term.strip() == "":
            # もしstopword除去後にtermが空になってしまった場合は、このtermは初期化vecの計算に使用せずスキップする
            # [memo] これがなかった時に、あるtermでtoken_ids=[]になり、E[token_ids] が shape (0, d) となった。これにより、その .mean(dim=0) は NaN になり、sum_vec += term_avg_vec で sum_vec 全体が NaN に汚染され、lossがずっとNaNになってしまった。
            print(f"Term '{term}' is empty after stopword removal. Skipping this term for initialization.")
            continue
        token_ids = tokenizer.encode(term, add_special_tokens=False)    # term内のidを取得 (自分でspace区切りする必要はない)
        if token_ids is None:
            raise ValueError(f"Token '{term}' not found in tokenizer vocabulary.")
        term_avg_vec = E[token_ids].mean(dim=0)  # term内のtokenのベクトルの平均を取る
        sum_vec += term_avg_vec
    if sum_vec.norm().item() == 0.0:
        # もし全てのtermがstopwordのみで構成されていて、stopword除去後に全てのtermが空になってしまい、sum_vecが0ベクトルのままになってしまった場合は、初期化vecの計算に使用するtermがないことになるため、エラーを出す
        raise ValueError(f"All terms resulted in zero vectors after stopword removal. Cannot initialize with zero vector.")

    # 2. term間の平均vecを計算
    init_src = sum_vec / len(init_terms)  # term間の平均を取る
    # [確認用] 平均pool後の init のノルムを計算
    if print_flag:
        norm = init_src.norm(p=2).item()
        print(f"Initial vector norm after mean pool: {norm:.4f}, value(~10): {init_src[:10]}")

    # # 3. ノルムを語彙平均に合わせる [memo] ノルムを合わせる必要は無さそうなのでコメントアウトした
    # target_norm = E.norm(dim=1).mean() # 目標のノルム: 語彙全体のベクトルのノルムの平均
    # src_norm = init_src.norm(p=2)   # 現在の init_src のノルム
    # eps = 1e-12 # 0除算防止用

    # # ノルムを target_norm に合わせて vec内の各値をrescale
    # init_src = init_src * (target_norm / (src_norm + eps))

    # # [確認用] 平均pool+語彙平均normに揃えた init のノルムを計算
    # if print_flag:
    #     norm = init_src.norm(p=2).item()
    #     print(f"Initial vector norm after norm adjustment: {norm:.4f}, target norm: {target_norm:.4f}, value(~10): {init_src[:10]}")


    # 4. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化. 
    # with torch.no_grad():
    #     init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)#.view(-1)     # torch long型に変換
    #     src = init_src.expand(len(init_target_ids), -1)           # init_target_idsの形に合わせてinit_src行を増幅した(ように見えるviewを)作成
    #     E.index_copy_(dim=0, index=init_target_ids, source=src)   # Eのinit_target_ids行(複数)をsrcで上書き
    with torch.no_grad():
        init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
        n = len(init_target_ids)
        src = init_src.unsqueeze(0).repeat(n, 1)   # (n, d) # 全トークンをカテゴリ重心で埋める
        E.index_copy_(dim=0, index=init_target_ids, source=src)
    
    # [確認用] model内のembedが書き換わっているかを確認:
    if print_flag:
        try:
            E = model.model.embed_tokens.weight  # (vocab, d)
        except:
            E = model.model.language_model.embed_tokens.weight  # (vocab, d)
        for v in E[init_target_ids]:
            print(f"\t『{init_target_ids}』 id vecs are updated with {init_src[:5]}... -> after: {v[:5]}...\n")
    return model


def initVecWithTokenVec_with_noise(model, tokenizer, init_terms, init_target_ids, print_flag=False):
    """
    埋め込み層の特定の行を、指定の語句のベクトルで初期化する関数.
    Args:
    - model: HuggingFaceのモデルオブジェクト
    - tokenizer: HuggingFaceのトークナイザオブジェクト
    - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
    - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
    - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか

    Returns:
    - model: 対象の行が初期化されたモデルオブジェクト

    memo:
    - 全てのtermのtokenを集めて、全tokenの平均を一括で計算すると、token 数が多い句ほど重みが大きくなり、カテゴリの重心が計算できなくなる（test結果も悪かった）
        → term毎にtokenの平均を計算してから、termの平均を取る方法に変更する
    """
    # ** 指定の語句で初期化 (句の場合は単純にmean poolingする) **

    try:
        E = model.model.embed_tokens.weight  # (vocab, d)
    except:
        E = model.model.language_model.embed_tokens.weight  # (vocab, d)

    STOPWORDS = {
        "a", "an", "the", "of", "in", "on", "at", "for", "to", "and", "or", "with"
    }
    def remove_stopwords_from_text(text):
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return " ".join(filtered)
    # term = "a chair in the room"
    # print(remove_stopwords_from_text(term))  # chair room


    # 1. term毎に平均vecを計算してから加算
    sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
    for term in init_terms:
        term = remove_stopwords_from_text(term)                         # aやtheなどの重要度の低い単語をtermから除去
        if term.strip() == "":
            # もしstopword除去後にtermが空になってしまった場合は、このtermは初期化vecの計算に使用せずスキップする
            # [memo] これがなかった時に、あるtermでtoken_ids=[]になり、E[token_ids] が shape (0, d) となった。これにより、その .mean(dim=0) は NaN になり、sum_vec += term_avg_vec で sum_vec 全体が NaN に汚染され、lossがずっとNaNになってしまった。
            print(f"Term '{term}' is empty after stopword removal. Skipping this term for initialization.")
            continue
        token_ids = tokenizer.encode(term, add_special_tokens=False)    # term内のidを取得 (自分でspace区切りする必要はない)
        if token_ids is None:
            raise ValueError(f"Token '{term}' not found in tokenizer vocabulary.")
        term_avg_vec = E[token_ids].mean(dim=0)  # term内のtokenのベクトルの平均を取る
        sum_vec += term_avg_vec
    if sum_vec.norm().item() == 0.0:
        # もし全てのtermがstopwordのみで構成されていて、stopword除去後に全てのtermが空になってしまい、sum_vecが0ベクトルのままになってしまった場合は、初期化vecの計算に使用するtermがないことになるため、エラーを出す
        raise ValueError(f"All terms resulted in zero vectors after stopword removal. Cannot initialize with zero vector.")

    # 2. term間の平均vecを計算
    init_src = sum_vec / len(init_terms)  # term間の平均を取る
    # [確認用] 平均pool後の init のノルムを計算
    # if print_flag:
    norm = init_src.norm(p=2).item()
    print(f"Initial vector norm after mean pool: {norm:.4f}, value(~10): {init_src[:10]}")

    # # 3. ノルムを語彙平均に合わせる [memo] ノルムを合わせる必要は無さそうなのでコメントアウトした
    # target_norm = E.norm(dim=1).mean() # 目標のノルム: 語彙全体のベクトルのノルムの平均
    # src_norm = init_src.norm(p=2)   # 現在の init_src のノルム
    # eps = 1e-12 # 0除算防止用

    # # ノルムを target_norm に合わせて vec内の各値をrescale
    # init_src = init_src * (target_norm / (src_norm + eps))

    # # [確認用] 平均pool+語彙平均normに揃えた init のノルムを計算
    # if print_flag:
    #     norm = init_src.norm(p=2).item()
    #     print(f"Initial vector norm after norm adjustment: {norm:.4f}, target norm: {target_norm:.4f}, value(~10): {init_src[:10]}")


    # 4. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化. 
    # with torch.no_grad():
    #     init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)#.view(-1)     # torch long型に変換
    #     src = init_src.expand(len(init_target_ids), -1)           # init_target_idsの形に合わせてinit_src行を増幅した(ように見えるviewを)作成
    #     E.index_copy_(dim=0, index=init_target_ids, source=src)   # Eのinit_target_ids行(複数)をsrcで上書き

    # 4'. (1tokenだけの学習時はcategory_COGだけ正解率80%まで行ったのに、複数tokensを一気に訓練するとほとんどaccが上がらなかった。カテゴリ内のtoken全てを同じ重心ベクトルで初期化していることが原因かもしれないのでノイズを加えて少しCOGvecをずらしてみる)
    with torch.no_grad():
        init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)

        n = len(init_target_ids)
        d = init_src.shape[0]

        # まず全トークンをカテゴリ重心で埋める
        src = init_src.unsqueeze(0).repeat(n, 1)   # (n, d)

        # 微小ノイズを作る
        noise = torch.randn((n, d), device=E.device, dtype=E.dtype)

        # 各行をL2正規化して「方向だけランダム」にする
        eps = 1e-12
        noise = noise / noise.norm(p=2, dim=1, keepdim=True).clamp_min(eps)

        # ノイズの大きさを、重心ノルムのごく一部にする
        noise_scale = 3e-3   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
        init_norm = init_src.norm(p=2).clamp_min(eps)
        noise = noise * (init_norm * noise_scale)

        # 重心 + 微小ノイズ
        src = src + noise

        E.index_copy_(dim=0, index=init_target_ids, source=src)
    
    # [確認用] model内のembedが書き換わっているかを確認:
    if print_flag:
        try:
            E = model.model.embed_tokens.weight  # (vocab, d)
        except:
            E = model.model.language_model.embed_tokens.weight  # (vocab, d)
        for v in E[init_target_ids]:
            print(f"\t『{init_target_ids}』 id vecs are updated with {init_src[:5]}... -> after: {v[:5]}...\n")
    return model


class GradZeroHook:
    """ 特定の行(=tokenのidx)の勾配を0にするフック
    学習対象外のembedding行の勾配は全て0にするために用いる
    token_id ごとに True / False を直接切り替えることはできないため, この勾配フックを使って、学習したい token_id 以外の勾配を 0 にする
    * どのモデルでも共通のはず
    """
    def __init__(self, embeddingsToKeep):
        self.embeddingsToKeep = embeddingsToKeep # どの行(=tokenのidx)の勾配を0にするかはGradZeroHook class自身が持っておく必要があるため，インスタンス生成時にselfに保存する

    def setGradsToZeroHook(self, grad):
        grad = grad.clone() # 新しいメモリ上に値をコピー. これは，元の勾配テンソルを直接変更しないようにするための安全策。PyTorchの勾配計算では，元のテンソルが他の場所で使われている可能性があるため，直接変更すると予期せぬ副作用が発生することがある。clone()を使うことで，元のテンソルを保護しつつ，新しいテンソル上で安全に操作ができるようになる。
        grad[self.embeddingsToKeep] = 0.0 # 指定された行の勾配を0にする
        return grad


def prepareGemmaModel(
        model, 
        tokenizer, 
        train_token2tokenid, 
        init_vec_type, 
        category_to_concepts_for_vec, 
        category2initoken_ids, 
        print_flag=False
    ):
    """ Gemmaモデルを埋め込み層のみ学習できるように準備する
    1. model準備
    2. embedding以外のパラメータを凍結
    3. 指定のトークン以外は勾配が0.0になるようにフックを設定
    4. 損失関数準備

    Args:
        model: HuggingFaceのモデルオブジェクト
        num_trainTargetTokens: 学習対象とする特殊トークンの数 (架空のobjの数) [memo] llama, qwenのコードにはこれを追加していなかった。勾配をfreezeするtoken数をなるべく多くしてメモリ使用量を抑えるために導入
        init_vec_type: memory vectorの初期化方法。zeroまたはuniform, または語句. zero->0vec, uniform->一様分布, 語句->指定の語句の埋め込みベクトルで初期化, 数字->指定のコサイン類似度で近い語句のベクトルで初期化
        category_to_concepts_for_vec: カテゴリごとのvec初期化に使用する概念のリスト。init_vec_typeが 'category_COG' の場合に使用
        category2initoken_ids: カテゴリごとの初期化トークンIDのリスト。init_vec_typeが 'category_COG' の場合に使用
    memo:
    * Qwenとは違い，special_tokenが最初から用意されているため，tokenizerの拡張は不要．
        * よって，tokenizerもここで準備する
    * special_tokenは全て初期化し, 学習可能な状態にする
    """
    print('Prepare model')
    
    # *** embedding層以外の勾配を凍結 ***
    for param in model.parameters():
        param.requires_grad = False

    # *** embedding層内でも，対象外のtoken(reserved_special_token以外)の勾配を凍結 ***
    # <unused0>から順にnum_trainTargetTokens個のtokenIDを学習対象とし、この予約済み特殊token以外は勾配が0.0になるようにhookをかける. 
    # 学習可能tokenをなるべく少なく制限する理由は、gemma3のtokenizerに6242個もunused tokenが存在しており、これら全てを学習対象にしてしまうとモデルサイズが大きくなりすぎてしまうため。
    

    # [memo] ここでembed_tokens.weightが見つからないというerror。1bでは大丈夫だったのになぜ -> gemma-3-1bは言語モデルのみだが、4b以降はvision encoderが含まれているため、(vision_tower)と(language_model)の2つの大きなモジュールがmodelの直下に存在している。そのため、embedding層にアクセスするにはmodel.model.language_model.embed_tokens.weightとなる。model.model.embed_tokens.weightが見つからなかった時にprint(model.model)を表示したことで判明。
    # token_id ごとに True / False を直接切り替えることはできない. 勾配フック(GradZeroHook)を使って、学習したい token_id 以外の勾配を 0 にする
    try:
        # * 1b以下はvision_towerがないため、従来通りでOK
        model.model.embed_tokens.weight.requires_grad = True
    except:
        # print(model.model)
        # raise ValueError("モデルに embed_tokens.weight が見つかりません。Gemma3ベースのモデルを指定していることを確認してください。")
        # * 4b以上はvision_towerがあるため、language_modelを経由してアクセスする
        model.model.language_model.embed_tokens.weight.requires_grad = True
    

    # # *** embedding層内でも，対象外のtoken(reserved_special_token以外)の勾配を凍結 ***
    # # <unused0>から順にnum_trainTargetTokens個のtokenIDを学習対象とし、この予約済み特殊token以外は勾配が0.0になるようにhookをかける. 
    # # 学習可能tokenをなるべく少なく制限する理由は、gemma3のtokenizerに6242個もunused tokenが存在しており、これら全てを学習対象にしてしまうとモデルサイズが大きくなりすぎてしまうため。
    trainTokenIds = list(train_token2tokenid.values())
    
    try:
        # 上のtry-except同様の理由(vision_towerの有無)で処理を分ける.
        embeddingsToKeep = [i for i in range(model.model.embed_tokens.weight.shape[0]) if i not in trainTokenIds]
        gzh = GradZeroHook(embeddingsToKeep)
        model.model.embed_tokens.weight.register_hook(gzh.setGradsToZeroHook) # 登録したフックは勾配計算直後に呼び出される．関数の返り値がNoneなら元の勾配が，返り値がテンソルならそのテンソルが新しい勾配として使われる．
    except:
        embeddingsToKeep = [i for i in range(model.model.language_model.embed_tokens.weight.shape[0]) if i not in trainTokenIds]
        gzh = GradZeroHook(embeddingsToKeep)
        model.model.language_model.embed_tokens.weight.register_hook(gzh.setGradsToZeroHook)



    # *** 訓練対象tokenの埋め込みを初期化 ***
    model = initializeEmbed(
        model, 
        tokenizer, 
        train_token2tokenid, 
        init_vec_type, 
        category_to_concepts_for_vec,
        category2initoken_ids,
        print_flag=print_flag
    )

    criteria = torch.nn.CrossEntropyLoss()
    model.train()
    return model, criteria


def evaluateModel(model, tokenizer, evalInputs, evalOutputTexts, verbose=False):
    """
    Evaluate the model on the given inputs and outputs.
    """
    model.eval()
    # total_val_loss = 0.0
    # total_val_tokens = 0

    maxNewTokens = 10
    numCorrect = 0
    print('Evaluating on %d samples...'%len(evalInputs))
    with torch.no_grad():
        for i in range(len(evalInputs)):
            generation = model.generate(
                                    torch.LongTensor([evalInputs[i]]).to(model.device),
                                    max_new_tokens=maxNewTokens, 
                                    do_sample=False,
                                    repetition_penalty=1.05,
                                )[0]
            decodedGeneration = tokenizer.decode(generation)


            if verbose and i//2==0 or i==len(evalInputs)-1: # if verbose and i==0 or i==len(evalInputs)//2 or i==len(evalInputs)-1:
                print('--- Sample %d ---'%i)
                # print('\tP:', decodedGeneration)
                # print('\tT:', evalOutputTexts[i])
                print(decodedGeneration.startswith(evalOutputTexts[i]))
            
            if decodedGeneration.startswith(evalOutputTexts[i]):
                # もし生成テキストが正解テキストと一致したら
                numCorrect += 1

    acc = numCorrect / len(evalInputs)
    return acc



# ********************* train **********************

def train(model_size, 
          model, 
          tokenizer, 
          criteria, 
          train_samples, 
          memTokenIds, 
          padTokenId, 
          save_mem_dir, 
          lr=0.01, 
          maxEpochs=50, 
          earlyStoppingCount=5
    ):
    """
    [WIP] earlyStoppingCount は未使用になっている
    """
    if int(model_size) == 1:
        # global BATCH_SIZE
        BATCH_SIZE = 256 # 512
        factor=0.5
        min_lr=1e-04
        if maxEpochs >= 300:
            patience=100 # 300
            cooldown=200 # 100
        else:
            # maxEpochsが小さい場合はpatience, cooldownも小さくする
            patience=30
            cooldown=20
    
    elif int(model_size) == 2:
        BATCH_SIZE = 128 #256
        factor=0.9
        min_lr=1e-07
        if maxEpochs >= 300:
            patience=50
            cooldown=150
        else:
            # maxEpochsが小さい場合はpatience, cooldownも小さくする
            patience=30
            cooldown=10

    elif int(model_size) == 4:
        BATCH_SIZE = 4 # 16 (zao01なら16で行けそう) 4 (zao00は8でout of memoryになった)
        factor=0.9
        min_lr=1e-07
        if maxEpochs >= 300:
            patience=50
            cooldown=150
        else:
            # maxEpochsが小さい場合はpatience, cooldownも小さくする
            patience=30
            cooldown=10
    
    elif int(model_size) == 9:
        BATCH_SIZE = 32 # 64
        factor=0.95 #0.5
        min_lr=5e-08  #1e-06
        patience=30 # 100
        cooldown=100 # 500

    elif int(model_size) == 12:
        BATCH_SIZE = 4 # 16 # 64
        factor=0.95 #0.5
        min_lr=5e-08  #1e-06
        patience=30 # 100
        cooldown=100 # 500
    else:
        pass


    params = {
        'lr':lr, 
        'betas':(0.9, 0.95), 
        'weight_decay':0.0
    }
    try:
        opt = torch.optim.AdamW(
            model.model.embed_tokens.parameters(),
            **params
        )
    except:
        opt = torch.optim.AdamW(
            model.model.language_model.embed_tokens.parameters(),
            **params
        )

    scheduler = ReduceLROnPlateau(
        opt,
        mode='min',        # 'min': 減る=改善 (例: loss), 'max': 増える=改善 (例: accuracy)
        factor=factor,        # 悪化が続いたら lrを*倍に小さくする. 0.6B:0.5, 4B:0.8
        patience=patience,        # 何エポック分の非改善を許すか. bestの値(最も低いloss)から数えてこのエポック数分でthreshold以上の大きさの改善が見られなければlrを下げるか.
        threshold=1e-3,    # 「改善」とみなす最小変化量. threshold_mode='rel' の場合は相対的な変化量つまり%, 'abs' の場合は絶対的な変化量で判定
        threshold_mode='rel',  # 'rel' は相対, 'abs' は絶対差で判定
        cooldown=cooldown,        # lr 低下後、何エポックは様子見するか
        min_lr=min_lr, # 0.0,        # 下限
        eps=0, # 1e-05,         # lrの変化が微小すぎるときの更新抑制. old_lr - new_lr <= eps なら更新しない
    )


    maxLength = 0
    trainingData = []
    evalInputs = []
    evalOutputTexts = []
    temp_i = 0
    for sample in train_samples:
        # tokenized = tokenizer(sample)
        inputIds = tokenizer.encode(sample, add_special_tokens=True)

        if maxLength < len(inputIds):
            maxLength = len(inputIds)

        trainingData.append(inputIds) # 次token予測タスクのデータ
        evalInputs.append(inputIds[:-2]) # rel毎にルールベースで文にしているため，relの後の単語も同一になる文が多い. そのため, 最後の2tokenのみ(<word> + '.')の予測で可否を評価することにする
        evalOutputTexts.append(sample)

        if temp_i < 5:
            print(f"inputIds: {inputIds}")
            print(f"decoded inputIds: {tokenizer.decode(inputIds)}")
            print(f"evalInputIds: {evalInputs[-1]}") # 最後に追加したevalInputIdsを表示
            print(f"decoded evalInputIds: {tokenizer.decode(evalInputs[-1])}")
            print()
            temp_i += 1

    # ** padding ** 
    # : [[132, 45, 67], [23, 78]] -> [[132, 45, 67, [PAD], [PAD]], [23, 78, [PAD], [PAD], [PAD]]]
    trainingData = [t_ids + [padTokenId] * (maxLength - len(t_ids)) for t_ids in trainingData]
    trainingData = torch.LongTensor(trainingData).to(model.device)
    
    indices = list(range(len(trainingData)))


    accLog = {}
    print("Start training...")

    for epoch in tqdm(range(maxEpochs)):
        print('Epoch %d/%d'%(epoch+1, maxEpochs))

        # このepochの学習
        model.train()
        random.shuffle(indices)
        totalLoss = 0.0

        # epoch毎にshuffleしたindicesの順番を記録. 後からどのtextが最後に学習されたかを確認するため.
        id_to_text = {id: train_samples[id] for id in indices} # train_samplesをshuffle
        save_shuffled_samples_path = os.path.join(save_mem_dir, f"shuffled_samples_epoch{epoch+1}.json")
        with open(save_shuffled_samples_path, 'w') as f:
            json.dump(id_to_text, f, indent=4, ensure_ascii=False)

        for step, i in enumerate(range(0, len(indices), BATCH_SIZE)):
            batchIds = indices[i:i+BATCH_SIZE]
            # xs = trainingData[batchIds, :-1] # input_ids
            # ys = trainingData[batchIds, 1:]  # labels
            # [:-1], [1:] のように次token予測のためにずらす必要はないらしい。Hugging Face の causal LM では大抵、モデル内部で自動的に処理されるから。
            input_ids = trainingData[batchIds]
            attention_mask = (input_ids != padTokenId).long() # e.g. [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]] のような形で、padTokenIdの位置が0になるマスクを作成
            token_type_ids = torch.zeros_like(input_ids)    # 2つのシーケンスを識別するバイナリマスク. e.g. [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] のような形 (全て0でOK)
            
            labels = input_ids.clone()
            # labels[labels == padTokenId] = -100 # PAD の位置は loss 計算から無視させたい。CrossEntropyLossのignore_indexに合わせて、padTokenIdの位置を-100に変換しておく
            labels[attention_mask == 0] = -100    # pad_token_id == eos_token_id だと eosも-100になってしまうため、attention_maskが0の位置、つまりpadTokenIdの位置を-100に変換しておく

            opt.zero_grad()
            output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                labels=labels
            )
            # output = output.logits.view(-1, output.logits.shape[2])
            # loss = criteria(output, labels.flatten())
            loss = output.loss
            loss.backward()
            opt.step()

            # 勾配の確認
            if epoch == 0:
                try:
                    if sum(model.model.embed_tokens.weight.grad[1000]) == 0:
                        print(f"OK: 学習対象でないtokenID {1000} の勾配が0です")
                    if sum(model.model.embed_tokens.weight.grad[memTokenIds[0]]) != 0:
                        print(f"OK: 学習対象tokenID {memTokenIds[0]} の勾配が0ではありません")
                except:
                    if sum(model.model.language_model.embed_tokens.weight.grad[1000]) == 0:
                        print(f"OK: 学習対象でないtokenID {1000} の勾配が0です")
                    if sum(model.model.language_model.embed_tokens.weight.grad[memTokenIds[0]]) != 0:
                        print(f"OK: 学習対象tokenID {memTokenIds[0]} の勾配が0ではありません")
            # model.model.embed_tokens.weight[trainTokenIds] *= (1 - 0.01 * 0.1)  # lr * wd  # weight-decayを使う場合はここで手動
            

            # WandB logging
            log_interval=1 # batch_sizeを大きくしているので小さめ

            # log_intervalが1epoch内のstep数よりも大きい値に設定されている場合は1epoch毎にログを出力
            if log_interval > len(batchIds):
                log_interval = len(batchIds)
            
            if step % log_interval == 0:
                # print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2  # MB単位
                gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                wandb.log({
                    "loss": loss.item(), 
                    "epoch": epoch, 
                    "step": step, 
                    # "param_sum": next(model.parameters()).data.sum().item(),
                    "lr": opt.param_groups[0]["lr"], 
                    "gpu_memory_allocated_MB": gpu_memory,
                    "gpu_memory_reserved_MB": gpu_reserved
                })
            totalLoss += loss.item()
            # totalLoss += loss.detach().cpu().tolist()


        # 検証データはないため，train lossでschedulerを更新する. 余裕があれば検証データ（別ルールによるtripletの言い換え)を作り，それをevaluateModelに入れてlossを計算するようにする
        num_steps = math.ceil(len(indices) / BATCH_SIZE)
        avgLoss = totalLoss / num_steps
        if scheduler is not None:
            scheduler.step(avgLoss)
        print('epoch %d loss:' % (epoch+1), avgLoss)



        # **** 検証とschedulerの更新 ****
        if epoch%10 == 0:
            acc = evaluateModel(model, tokenizer, evalInputs, evalOutputTexts, verbose=True)
            accLog[epoch] = {'eval acc': acc}
            print(acc)
            if acc==1.0:
                break

        if epoch%100 == 0 or epoch in list(range(10)) + [10, 15, 20, 25, 40, 50, 60, 70, 80, 90]:
            save_mem_path = os.path.join(save_mem_dir, f"{epoch}.pth")
            save_mem_vec(model, memTokenIds, save_mem_path)
            print(f"trained vecs at {epoch} are saved in {save_mem_path}.")
                        
    return model, accLog







# *************************************************************** main ***************************************************************
def main(args):
    print_flag = False

    seed = args.seed
    model_size = args.model_size
    lr = args.lr
    maxEpochs = args.max_epochs
    # target_concept_list = [args.target_concept] # 🟠 修正前
    target_concepts_filename = args.target_concepts_filename # 🟠 修正後
    init_vec_type = args.init_vec_type

    trained_date = datetime.now().strftime("%Y%m%d")


    global model_name_for_dirname
    # [WIP] 'it'と'pt'のどちらが良いかは未検証.とりあえず'it'で統一.
    if model_size in ['2', '9']:
        model_version = 2
    elif model_size in ['1', '4', '12']:
        model_version = 3
    else:
        pass
    model_name = f"google/gemma-{model_version}-{model_size}b-it" # [memo] 'gemma-'部分は変えないこと!! -を消すとモデルがloadできない．さらにそのエラーメッセージは，"huggingface-cli login"をして，という関係ないmessageになるので注意!
    model_name_for_dirname = f"gemma-{model_version}-{model_size}B-lr{lr}-{trained_date}_seed{seed}"


    # *** tokenizer/modelをload. ただしmodelの設定はここではまだ行わない ***
    tokenizer = AutoTokenizer.from_pretrained(model_name) #, token=access_token)

    if 'gemma-' not in model_name.lower():
        print('The specified model does not seem Gemma3-based model.')
        print('Calculation for non-Gemma3 models is not implemented yet [TODO]')
        raise ValueError("The specified model does not seem Gemma3-based model.")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")#, token=access_token)

    if tokenizer.pad_token_id is None:
        # llama系の場合はpad_tokenが設定されていないことがあるため，以下のようにeos_tokenをpad_tokenに設定する. gemma3は設定済みだった
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id


    # *** dir/file path 設定 ***
    train_data_dir = os.path.join(project_root, 'data', 'train_data') # 🟠train_data_dir = os.path.join(project_root, 'data', 'triplets')

    # *** 🟠 修正後(2026/01/18~) 引数でconfig/{target_concepts_filename}で指定されたconcept群を学習対象とするように変更 ***
    class_to_target_concepts_path = os.path.join(project_root, 'config', target_concepts_filename)
    print(f"Loading target concepts from: {class_to_target_concepts_path}")
    if not os.path.exists(class_to_target_concepts_path) or target_concepts_filename.split('.')[-1] != 'json':
        raise ValueError(f"指定されたtarget_concepts_filename '{target_concepts_filename}' が存在しないか，jsonファイルではありません。configディレクトリ内の正しいjsonファイル名を指定してください。")
    with open(class_to_target_concepts_path, 'r') as f:
        class_to_target_concepts_config = json.load(f)
    config_concept_list = sum(class_to_target_concepts_config.values(), [])
    print(f"Target concepts specified in config: {config_concept_list}")
    save_mem_dir = os.path.join(project_root, "memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")

    # もしすでに同名のディレクトリが存在していたら、_2のように末尾に連番をつける
    original_save_mem_dir = save_mem_dir
    counter = 2
    while os.path.exists(save_mem_dir):
        save_mem_dir = f"{original_save_mem_dir}_{counter}"
        counter += 1
    print('save_mem_dir:', save_mem_dir)
    os.makedirs(save_mem_dir, exist_ok=True)


    # ********* train data取得準備 *********
    # 全てのカテゴリ・固有名詞リスト の辞書を読み込む (重複等のfiltering済み)
    filtered_category_properNouns_dict = loadProperNounData(
        propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept,
        print_flag=print_flag
    )

    # ***** target_concept_listに学習対象のconcept名を追加 *****
    # ** 学習データが存在するconcept名のみを抽出.
    trainable_concept_list = []
    for filename in os.listdir(train_data_dir):
        concept_name = filename.split('.json')[0].replace('_', ' ')
        trainable_concept_list.append(concept_name)
    print(f"\ntrainable concepts: {trainable_concept_list}")

    # ** configファイルの指定に基づき，学習対象conceptを絞り込む
    if config_concept_list[0] not in [None, 'None']:
        # * 学習対象conceptが個別に指定されている場合. (config/target_concepts.jsonで指定):
        category_to_conceptsForFict = defaultdict(list)
        # config_concept_listに含まれ、学習データが存在し、所属するカテゴリが特定できるconceptのみを抽出
        for tcat, tconc_lst in class_to_target_concepts_config.items(): # for tc in config_concept_list:
            for tconc in tconc_lst:
                if tconc in trainable_concept_list:
                    category_to_conceptsForFict[tcat].append(tconc)
    else:
        # * 学習対象conceptが個別に指定されていない場合: そのままtarget_concept_listの(学習可能な概念)全てを学習対象とする
        category_to_conceptsForFict = defaultdict(list)
        for category, concepts in filtered_category_properNouns_dict.items():
            for concept in concepts:
                if concept in trainable_concept_list:
                    category_to_conceptsForFict[category].append(concept)

    target_concept_list = sum(category_to_conceptsForFict.values(), [])
    target_concept_list = sorted(target_concept_list) # target_concept_list をアルファベット順にsort
    print(f"Class - target concepts mapping (after filtering with trainable concepts):")
    if print_flag:
        for category, concepts in category_to_conceptsForFict.items():
            print(f"  {category}: {concepts}")
    print("Target concept list:", target_concept_list, '\n')

    # 架空の概念用の固有名詞から、その所属カテゴリを引けるようにするためのmap
    conceptForFict2category_map = {conceptForFict: category for category, concepts in category_to_conceptsForFict.items() for conceptForFict in concepts} 
    print(f"categories in conceptForFict2category_map: {list(conceptForFict2category_map.values())[:5]} ...") # 先頭5カテゴリを表示

    

    # ***** 各カテゴリ内で、vec初期化に使用する固有名詞を取得 *****
    # # 各概念毎に、架空の概念用の特徴の生成に成功した(=架空の概念用の)固有名詞を取得する -> train data 作成時点で特徴獲得には成功している
    # category_to_concepts_for_fictconcept = loadConceptsForFictConcept()
    # 次に、各カテゴリの全固有名詞から、架空の概念用の特徴の生成に成功した固有名詞を除外し、残った固有名詞全てをvec初期化用の概念とする
    category_to_concepts_for_vec = {}
    for category in filtered_category_properNouns_dict.keys():
        if len(filtered_category_properNouns_dict[category]) < 1:
            # もし、そもそも固有名詞が1つもないカテゴリがあれば、そのカテゴリはcategory_to_concepts_for_vecから除外する。
            continue
        propernouns_for_init_vec = list(set(filtered_category_properNouns_dict[category]) - set(category_to_conceptsForFict[category]))
        category_to_concepts_for_vec[category] = propernouns_for_init_vec
        if print_flag:
            print(f"category: {category}, proper nouns for vec initialization: {len(propernouns_for_init_vec)}, {propernouns_for_init_vec[:5]} ...") # 先頭5個を表示


    # ********* train data取得準備 *********
    # *** concept数の分だけ空きtokenを確保する ***
    trainTokenIds = [tokenizer.convert_tokens_to_ids(f'<unused{i}>') for i in range(len(target_concept_list))]
    trainTokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in trainTokenIds]

    train_token2tokenid = {}
    for id, token in zip(trainTokenIds, trainTokens):
        train_token2tokenid[token] = id
    if len(trainTokens) > 6: # 先頭3つと末尾3つを表示するための条件分岐
        print(f"train target tokens: {len(trainTokens)}, {trainTokens[:3]} ... {trainTokens[-3:]}") # 先頭3つと末尾3つを表示
    else:
        print(f"train target tokens: {len(trainTokens)}, {trainTokens}") # 全て表示


    # *** (架空の)concept名: 空きトークン の割り当て辞書作成 ***
    conceptForFict2token_map = {}
    memTokenIds = []
    for target_concept, trainable_token, token_id in zip(target_concept_list, trainTokens, trainTokenIds):
        conceptForFict2token_map[target_concept] = trainable_token
        memTokenIds.append(token_id)
        # print(f"{token_id}: {target_concept} -> {trainable_token}")

    # concept-token割り当てを保存
    save_path = os.path.join(save_mem_dir, "token_assignment.json")
    with open(save_path, "w") as f:
        json.dump(conceptForFict2token_map, f, ensure_ascii=False, indent=4)
    print(f"Saved conceptForFict2token_map to {save_path}")


    # *** categoryごとに、割り当てた空きtoken idをリストにまとめる. ***
    # これは、'category_COG' 系の初期化の場合、categoryごとのvec初期化の際に、同じカテゴリの概念に割り当てたtokenのidをまとめて初期化するために使用する. 
    # (カテゴリ毎に重心vecを作成するため、同じカテゴリ内の概念に該当する空きtokenは同じ重心vecで初期化するから。)
    category2initoken_ids = defaultdict(list)
    for conceptForFict in conceptForFict2token_map.keys():
        category = conceptForFict2category_map.get(conceptForFict)
        if category is None:
            raise ValueError(f"Concept '{conceptForFict}' not found in any category.")
        tk = conceptForFict2token_map[conceptForFict]
        tk_id = train_token2tokenid[tk]
        category2initoken_ids[category].append(tk_id)




    # ****** tripletを読み込み学習データを構築 ******
    train_samples = []
    for target_concept in target_concept_list:
        # load data
        filename = target_concept.replace(' ', '_') + '.json'
        with open(os.path.join(train_data_dir, filename), 'r') as f:
            data = json.load(f)
        wiki_text = data['summary'] # data['text']も選べるが、fact sentencesに比べて大き過ぎるのでsummaryを使用している
        facts = data['facts']

        # 対応する空token名を取得. <unused0>など
        unused_token = conceptForFict2token_map[target_concept]

        # concept名/"It" を割り当てられた空tokenに置換 (大小区別なし)
        wiki_text_with_token = re.sub(re.escape(target_concept), unused_token, wiki_text, flags=re.IGNORECASE)
        facts_with_token = [fact.replace('It', unused_token) for fact in facts]
        facts_with_token = random.sample(facts_with_token, len(facts_with_token)) # factsの順番をランダムに入れ替える。これも、学習データの多様性を高めるための工夫。

        # featuresをn個ずつに分割して、1sampleあたり、summary1つ+特徴文n個の形式にする。最後の余りはそのまま1sampleにする。
        with open(os.path.join(project_root, 'data', 'templates', 'train_sample_format.json'), "r") as f:
            train_sample_format = json.load(f)
        train_sample_template = train_sample_format['train_sample']
        train_fact_sentence_template = train_sample_format['train_fact_sentence']

        for i in range(0, len(facts_with_token), n_feat_in_a_sample):
            fact_sentences = facts_with_token[i:i+n_feat_in_a_sample]
            fact_sentences_str = "\n".join([train_fact_sentence_template.format(fact_sentence=fs) for fs in fact_sentences])
            train_sample = train_sample_template.format(
                target_concept=target_concept,
                summary=wiki_text_with_token, 
                fact_sentences=fact_sentences_str
            )
            train_samples.append(train_sample)
            if print_flag:
                print(train_sample)
                print("**********")
        
        # 各(架空)概念毎に1sample表示する
        print(f"Example train 1 sample for concept '{target_concept}':")
        print(train_samples[-1])
        print("**********")

    print(f"Total {len(train_samples)} train samples created.")


    # # Wandb設定
    PROJECT_NAME = os.path.basename(save_mem_dir) # save_mem_dirの一番最後だけ取ってくる
    wandb_dir = os.path.join(project_root, "memvec_wandb_logs", model_name_for_dirname)
    set_wandb_env(PROJECT_NAME, wandb_dir)


    # ********* model等の準備: 予約済み特殊トークンの埋め込みの初期化など *********
    model, criteria = prepareGemmaModel(
        model, 
        tokenizer, 
        train_token2tokenid, 
        init_vec_type, 
        category_to_concepts_for_vec,
        category2initoken_ids,
        print_flag=print_flag
    )
    padTokenId = tokenizer.vocab[tokenizer.pad_token]
    



    # ********* train *********
    start_time = time.time()
    model, accLog = train(
        model_size,
        model,
        tokenizer,
        criteria, 
        train_samples,
        memTokenIds,
        padTokenId, 
        save_mem_dir,
        lr, 
        maxEpochs, 
        earlyStoppingCount=5
    )
    print('accLog:', accLog)

    # detailedHistory = {epoch:{c:accLog[epoch]['detailed'][c] for c in accLog[epoch]['detailed']} for epoch in accLog}
    detailedHistory = accLog
    df = pd.DataFrame(detailedHistory)
    # df.to_csv(args.output)

    # *** save ***
    save_hist_path = f'{project_root}/memvec_training_history/{model_name_for_dirname}.csv'
    os.makedirs(os.path.dirname(save_hist_path), exist_ok=True)
    df.to_csv(save_hist_path)

    # 最後のmemory vectorをセーブしたい場合
    save_mem_path = os.path.join(save_mem_dir, f"{maxEpochs}.pth")
    save_mem_vec(model, memTokenIds, save_mem_path)
    print(f"trained vecs at {maxEpochs} are saved in {save_mem_path}.")


    # 訓練ループ全体の時間を計測
    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    wandb.finish()










# ********************* 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--target_concept', type=str, default=None, help='学習対象とするconcept. 例: "British_Library". Noneの場合はtripletsディレクトリ内にある全conceptsが対象となる。Noneの場合は、同時に複数の架空concept(に対応するtoken<unk0>, <unk1>, <unk2>, <unk3>, ...)を学習させると言うこと。') # listで複数指定できるようにしたかったが、文字列のリストをコマンドラインで渡せなかったので1つだけ指定する。
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"')
    parser.add_argument('--model_size', type=str, default='12', help='モデルサイズ (例: 4, 9, 12)')
    parser.add_argument('--lr', type=float, default=0.01, help='学習率')
    parser.add_argument('--max_epochs', type=int, default=600, help='最大エポック数')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    # parser.add_argument('--init_vec_type', type=str, default='zero', help='memory vectorの初期化方法。zero or uniform or norm_rand, または語句. zero->0vec, uniform->一様分布, 語句->指定の語句の埋め込みベクトルで初期化, 数字->各conceptに対するその数字の類似度の語句ベクトルで初期化')
    parser.add_argument('--thread_id', type=int, nargs='?', default=0, help='複数process同時に実行する場合のthread id (0 or 1). これにより,実行する設定(seed, init_vec_typeの組)が被らないように調整する')
    parser.add_argument('--process_num', type=int, nargs='?', default=2, help='同時に実行するprocess数')
    parser.add_argument('--seed_num', type=int, nargs='?', default=10, help='シードの数. 例えば10に設定した場合、seed0からseed9までの10個のシードで学習を実行することになる。')
    args = parser.parse_args()

    processNum = args.process_num # 同時に実行するprocess数

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        
    task_id = -1
    for seed in range(args.seed_num):
        args.seed = seed # mainにargsとして渡すためにargs.seedに代入している。main内でargs.seedを参照することで、現在のシード値を取得できるようになる。
        
        # if seed == 1 and args.model_size=='12':
        #     # もう途中まで実行済みなので，残りを実行
        #     init_vec_type_lst = ['zero', 'uniform', 'norm_rand', 0.0]
        # else:
        #     # 通常
        #     init_vec_type_lst = ['zero', 'uniform', 'norm_rand', 'norm_rand_vocab', 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        init_vec_type_lst = ['category_centroid_plus_random', 'other_category_COG', 'norm_rand_vocab', 'zero', 'uniform', 'norm_rand', 'category_COG', ]
        # init_vec_type_lst = ['category_centroid_plus_random', 'other_category_COG', 'norm_rand_vocab', 'uniform',]
        # init_vec_type_lst = ['category_COG', 'other_category_COG', 'zero', ]
        # init_vec_type_lst = ['uniform', 'norm_rand', 'norm_rand_vocab']
        for init_vec_type in init_vec_type_lst:

            print(f"\n\n=== Training with init_vec_type: {init_vec_type}, seed: {seed} ===")
            args.init_vec_type = str(init_vec_type)

            task_id += 1

            # if seed == 0 and init_vec_type == 'category_COG':
            #     print("Skipping task_id %d (seed %d, init_vec_type %s) because it has already been executed." % (task_id, seed, init_vec_type))
            #     continue # すでに途中まで実行済みなので、残りを実行
            
            if task_id % processNum != args.thread_id:
                # 複数process同時に実行する場合, thread_idに応じてtask_idが偶数or奇数の設定のみを実行する
                print(f"Skipping task_id {task_id} for thread_id {args.thread_id}")
                continue

            fix_seed(seed)
            main(args)

            # GPUメモリ解放
            torch.cuda.empty_cache()
            # 3秒待機
            time.sleep(3)