
# ===== Standard library =====
from datetime import datetime
import json
import os
import random
import re
import sys
from tqdm import tqdm
from collections import defaultdict

# ===== Third-party =====
import pandas as pd
import torch
from transformers import logging as transformers_logging

# ===== Runtime config =====
transformers_logging.set_verbosity_error()

project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
# project_root = os.environ["HOME"] # [memo] genkaiを使う場合. "/singularity_home/project/EmbedNewConcept/src/trainMemVec_fromXvec_gemma.py"
sys.path.append(project_root)
print("Project root:", project_root)

from utils.wikipedia_api_utils import extract_wiki_main_text, fetch_wikipedia_page, load_wikisummary
from utils.handle_text_utils import get_first_few_sentences, repeat_text
from utils.handle_data_from_dbpedia_utils import load_prop_nouns
from utils.embedding_utils import extract_hidden_states, get_mix_layers

N_COMPONENTS = 2
NOISE_SCALE = 5e-3   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
BATCH_SIZE = 4 #16 #8


wiki_pages_dir = os.path.join(project_root, "data", "wiki_pages")
category_similarity_path = os.path.join(project_root, 'data', 'cossim_bw_categories', 'aggregated_near_far_analysis_across_seeds.json')

# for debug
debug_print_initterms_flag = True






class EmbedInitializer:
    def __init__(self, 
                 init_vec_type,
                 pool_hs_type,
                 train_target_category_lst=None, 
                 propnoun_num_for_init_vec=None, 
                 model_name=None,
                 save_mem_dir=None, 
                 min_words=None,
                 max_words=None
                 ):

        self.model_name = model_name.split("/")[-1]  # "gemma-3-12b-it"のようなモデル名だけを取り出す
        self.save_mem_dir = save_mem_dir             # パラメータを保存したい。2026/04/06に追加
        self.init_vec_type = init_vec_type
        self.train_target_category_lst = train_target_category_lst
        self.propnoun_num_for_init_vec = propnoun_num_for_init_vec
        self.pool_hs_type = pool_hs_type    # single_last/eos/mean,  inputsに対する隠れ状態を、term中の全subtokenに対して平均するか、term中の最後のtokenに対応する隠れ状態を使うか
        self.layer_to_globalHSMeanVec = {}
        self.category_to_layer_to_otherHSMeanVec = defaultdict(dict)  # category_to_layer_to_otherHSMeanVec[category][layer_idx] = other_hidden_mean_vec for that category and layer
        self.global_primary_vec_by_layer = {}       # layer代表vecはその単層の主成分
        self.num_propNouns_in_cat_for_globalHSMean = 100

        self.propnoun_to_wikisummary = {}   # これが必要な関数を実行する際に、中身が空なら読み込む。少し時間とメモリを食うので不要なら読み込まない。
        self.repeat_prompt = False   # promptを2回繰り返すプロンプトを使うかどうかのフラグ。
        self.min_words = min_words
        self.max_words = max_words

        self.other_init_use_target_candidates_only = True    # Trueなら、other系の初期化に、学習対象カテゴリの候補カテゴリを使う。Falseなら、学習対象カテゴリの候補カテゴリは使わず、使用可能な全カテゴリからランダムに選ぶ。つまり target_concepts.json中のカテゴリのみ or loadProperNounDataで集めた全てのカテゴリ
        

        if "Repeat" in init_vec_type:
            self.repeat_prompt = True
        else:
            # print(f"init_vec_type: {init_vec_type} does not require global hidden mean calculation. Skipping that step.")
            pass

        if self.save_mem_dir is not None:
            # ***** self.save_mem_dir にこの訓練のパラメータを辞書保存 *****
            path = os.path.join(self.save_mem_dir, "embed_initializer_params.json")
            params_to_save = {
                "model_name": self.model_name,
                "init_vec_type": self.init_vec_type,
                "train_target_category_lst": self.train_target_category_lst,
                "propnoun_num_for_init_vec": self.propnoun_num_for_init_vec,
                "pool_hs_type": self.pool_hs_type,

                "train_date_time":  datetime.now().strftime("%Y%m%d%H%M%S"),
                "num_of_global_vec_primary_components": N_COMPONENTS,
                "noise_scale": NOISE_SCALE,
                "other_init_use_target_candidates_only": self.other_init_use_target_candidates_only,
            }

            with open(path, "w") as f:
                json.dump(params_to_save, f)


    # ================================ 埋め込みベクトルの初期化関数(handler) ================================
    def initializeEmbed(
            self,
            model, 
            tokenizer,
            train_token2tokenid, 
            init_vec_type, 
            category_to_concepts_for_vec, 
            category2initoken_ids,
            layer_idx=None,
            print_flag=False,
            return_init_vecs_only=False,    # Trueなら、初期化されたモデルを返すのではなく、train_token2tokenidで指定されたtoken idsに対応する初期化vecのみを返す
        ):
        """言語モデルが持つ「語彙 → ベクトル」への変換をids指定のもののみ初期化する
        行数 = 語彙サイズ、列数 = 埋め込み次元。
        trainTokenIds で指定された行（＝特定のトークンに対応するベクトル）だけを操作する．
        * どのモデルでも共通のはず

        Args:
            model: HuggingFaceのモデルオブジェクト
            tokenizer: HuggingFaceのトークナイザオブジェクトtrain_token2tokenid: 学習対象とする特殊トークンとそのtoken_idのmap
            init_vec_type: memory vectorの初期化方法。zeroまたはuniform, または語句. zero->0vec, uniform->一様分布, 語句->指定の語句の埋め込みベクトルで初期化, 数字->指定のコサイン類似度で近い語句のベクトルで初期化
            category_to_concepts_for_vec: カテゴリごとのvec初期化に使用する概念のリスト。init_vec_typeが 'category_COG' の場合に使用
            category2initoken_ids: カテゴリごとの初期化トークンIDのリスト。init_vec_typeが 'category_COG' の場合に使用
            layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。init_vec_typeが 'category_centroid_by_hidden_state_mean' の場合に使用
            print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """    
        # もしinit_vec_typeを数字に変換できる場合は変換する
        try:
            init_vec_type = float(init_vec_type)
        except:
            pass

        handlers = {
            "uniform": lambda: self.initvec_by_uniform(model, train_token2tokenid),                 # ** 一様分布で初期化 **
            "norm_rand": lambda: self.initvec_by_norm_rand(model, train_token2tokenid),             # ** ノルム固定の正規化ランダムで初期化 **
            "norm_rand_vocab": lambda: self.initvec_by_norm_rand_vocab(model, train_token2tokenid), # ** 正規化ランダムで初期化（N(μ,σ2)のμとσは語彙集合から計算） **
            "zero": lambda: self.initvec_by_zero(model, train_token2tokenid),                       # ** 0vecで初期化 **


            # ============================================================================================
            # 2026/04/09
            # wiki summaryを2回繰り返してプロンプトとし、2文目の隠れ状態から初期化vecを作成する方法: https://openreview.net/forum?id=Ahlrf2HGJR の手法. src_visualize/plot_gemma_hidden_states_3d.py でカテゴリ同士が他の手法よりも分離できていたため.
            "CatCent_by_WikiSummaryRepeatHSMixed": lambda: self.initialize_embeds_by_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                mix_layers=True,
                print_flag=True,
                return_init_vecs_only=return_init_vecs_only
            ),
            "farCatCent_by_WikiSummaryRepeatHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True,
                print_flag=True,
                return_init_vecs_only=return_init_vecs_only
            ),

            "nearCatCent_by_WikiSummaryRepeatHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                other_type="near", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True,
                print_flag=True,
                return_init_vecs_only=return_init_vecs_only
            ),
        }
        print(handlers.keys())

        if init_vec_type in handlers:
            print(f"Initializing embeddings with method: {init_vec_type}")
            return handlers[init_vec_type]()

        raise ValueError(f"Unknown init_vec_type: {init_vec_type}. Available methods: {list(handlers.keys())}")

    







    def initVecWithTokenVec(self, model, tokenizer, init_terms, init_target_ids, print_flag=False):
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
        E = self._get_model_info(model)[0]  # (vocab, d)


        # 1. term毎に平均vecを計算してから加算
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
        for term in init_terms:
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

        # 3. ノルムを語彙平均に合わせる [memo] ノルムを合わせる必要は無さそうなので削除済み
        # 4. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化.
        with torch.no_grad():
            init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
            n = len(init_target_ids)
            src = init_src.unsqueeze(0).repeat(n, 1)   # (n, d) # 全トークンをカテゴリ重心で埋める
            E.index_copy_(dim=0, index=init_target_ids, source=src)
        
        # [確認用] model内のembedが書き換わっているかを確認:
        if print_flag:
            E = self._get_model_info(model)[0]  # (vocab, d)
            for v in E[init_target_ids]:
                print(f"\t『{init_target_ids}』 id vecs are updated with {init_src[:5]}... -> after: {v[:5]}...\n")
        return model


    def initVecWithTokenVec_with_noise(self, model, tokenizer, init_terms, init_target_ids, print_flag=False):
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
        E = self._get_model_info(model)[0]  # (vocab, d)

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
            E = self._get_model_info(model)[0]  # (vocab, d)
            for v in E[init_target_ids]:
                print(f"\t『{init_target_ids}』 id vecs are updated with {init_src[:5]}... -> after: {v[:5]}...\n")
        return model








    # ============================================= 初期化vec作成用関数の実装 ===============================================

    # wikiのsummary入力時の隠れ状態で初期化vecを作成する方法は、global_hidden_meanを計算する必要がないため、ここでは何もしない
    # 'CatCent_by_wikiSummary_HSMixed', 'otherCatCent_by_wikiSummary_HSMixed',

    # ========================== 基本的な初期化方法 ==========================

    def initvec_by_uniform(self, model, train_token2tokenid):
        # ** 一様分布で初期化 **
        trainTokenIds = list(train_token2tokenid.values())
        W = self._get_model_info(model)[0]
        with torch.no_grad():
            idx = torch.as_tensor(trainTokenIds, device=W.device, dtype=torch.long)
            src = torch.empty((len(trainTokenIds), W.shape[1]), device=W.device, dtype=W.dtype)
            src.uniform_(-0.1, 0.1)
            W.index_copy_(0, idx, src)
        return model
    


    def initvec_by_norm_rand(self, model, train_token2tokenid):
        # ** ノルム固定の正規化ランダムで初期化 **
        # 各トークン埋め込みベクトルの「方向」はランダム、L2ノルムは一定（例: 0.1）に揃える
        target_norm = 0.1
        eps = 1e-12

        trainTokenIds = list(train_token2tokenid.values())

        W = self._get_model_info(model)[0]

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



    def initvec_by_norm_rand_vocab(self, model, train_token2tokenid):
        # ** 正規化ランダムで初期化（N(μ,σ2)のμとσは語彙集合から計算） **
        trainTokenIds = list(train_token2tokenid.values())
        # 埋め込み重み（V x d）への参照（モデル差異に対応）
        W = self._get_model_info(model)[0]
        
        # 全要素をまとめて（スカラー1つの平均・標準偏差）
        mu = W.mean().item()
        sigma = W.std(unbiased=False).item()  # ddof=0（母標準偏差）
        print(f"Vocabulary embedding mean: {mu:.4f}, std: {sigma:.4f}")
        rand = torch.normal(mean=mu, std=sigma, size=(len(trainTokenIds), W.shape[1]), device=W.device, dtype=W.dtype)  # (V, d)
        
        with torch.no_grad():
            W[trainTokenIds].copy_(rand)
        return model


    def initvec_by_zero(self, model, train_token2tokenid):
        # ** 0vecで初期化 **
        trainTokenIds = list(train_token2tokenid.values())
        W = self._get_model_info(model)[0]
        with torch.no_grad():
            trainTokenIds = torch.as_tensor(trainTokenIds, device=W.device, dtype=torch.long)
            W.index_fill_(0, trainTokenIds, 0.0)
        return model






    # ========================== 任意の関数で初期化vecを作成する方法 ==========================

    def initialize_embeds_by_category_centroid_by_function(
        self, 
        model, 
        tokenizer, 
        category_to_concepts_for_vec, 
        category2initoken_ids, 
        initvec_func, 
        layer_idx=None,
        mix_layers=False, 
        print_flag=False,
        return_init_vecs_only=False  # Trueなら、初期化されたモデルを返すのではなく、train_token2tokenidで指定されたtoken idsに対応する初期化vecのみを返す
        ):
        """
        # *** 初期vec作成用の固有名詞リストをカテゴリ毎に用意し、任意の関数で初期vecを作成する方法. ***

        新規概念と同じカテゴリ内の既存概念100個を利用し，1つの新規概念の初期化用ベクトルを作成する．
        * 90 / 100 個は，カテゴリ内の新規概念全てに共通させる既存概念
        * 10 / 100 個は，初期化対象の新規概念毎にランダムに選ぶ既存概念 (token毎に異なるものを選ぶ) とする．
            10個を新規概念ごとに変えるのは，カテゴリ内の全ての新規概念の初期ベクトルが全く同じになると勾配の更新で全ての新規概念のベクトルが同じように動いてしまい、結果的に新規概念同士の区別がつかなくなってしまう可能性があるため。10個程度ランダムに変えることで、初期化ベクトルに多様性を持たせる。
        
        return_init_vecs_only=Trueの場合は，token_idは不要のため，category2initoken_idsは{category: [token名, ...]}の形式も受け付ける
        """

        if return_init_vecs_only:
            # 初期化されたモデルを返すのではなく、train_token2tokenidで指定されたtoken idsに対応する初期化vecのみを返す場合は、初期化vecを辞書に保存していく
            init_vecs_for_category = defaultdict(list)  # init_vecs_for_category[own_category] = [init_vec for that category]

        for own_category, init_token_ids in category2initoken_ids.items():
            # *** このカテゴリに対応する初期化vec作成用の固有名詞リストで初期化vecを作成し、
            # このカテゴリに属す固有名詞(新規概念用)に割り当てたtokenのtoken idsの行を、その初期化vecで初期化する ***
            init_terms_candidate = category_to_concepts_for_vec[own_category]

            if len(init_terms_candidate) < (self.propnoun_num_for_init_vec - 10  +  10 * len(init_token_ids)):
                # カテゴリ内の固有名詞が、初期化vec作成における、「カテゴリ内固定成分 + token毎のランダムな成分」のための固有名詞数に足りない場合は、エラーを出して終了
                raise ValueError(f"Not enough concepts in category '{own_category}' to sample for centroid vec. Required: at least {self.propnoun_num_for_init_vec}, Available: {len(init_terms_candidate)}. Please reduce the number of concepts needed for centroid vec or add more concepts to the category.")
            

            init_terms_for_centroid = random.sample(init_terms_candidate, self.propnoun_num_for_init_vec-10)  # カテゴリ内の固有名詞からランダムに(propnoun_num_for_init_vec-10)個選んで中心vecを作成

            # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
            init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

            # ** 新規token毎に初期vecに変化をつけるためのnoiseとして、初期化対象の追加token毎に、10件をランダム選出する **
            # init_terms_for_random では、token毎に異なるtermを使いたい（token間で被るとvecの多様性が下がるため)
            # token毎に重複のない、初期vecのrandom成分用のtermsリスト: termsを重複無しで10個ずつランダムに分けたリストを作成する
            init_terms_for_random_list = random.sample(init_terms_candidate, 10 * len(init_token_ids))  # まず、元のリストをシャッフルする =カテゴリ内の固有名詞からランダムに(10 * token数)個選ぶ
            init_terms_for_random_chunks = [init_terms_for_random_list[i:i + 10] for i in range(0, len(init_terms_for_random_list), 10)]  # ランダムに選んだ固有名詞を、10個ずつのチャンクに分ける
            
            for i, init_token_id in enumerate(init_token_ids):
                init_terms_for_random = init_terms_for_random_chunks[i]
                init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞 + ランダムvec用の固有名詞 を初期化vec作成に使用
                    
                # init_termsから、初期化vecを作成する
                init_src = initvec_func(
                    model, 
                    tokenizer, 
                    init_terms,
                    layer_idx=layer_idx,
                    mix_layers=mix_layers,
                    print_flag=print_flag
                )

                if return_init_vecs_only:
                    # 初期化されたモデルを返すのではなく、train_token2tokenidで指定されたtoken idsに対応する初期化vecのみを返す場合は、初期化vecを辞書に保存していく
                    init_vecs_for_category[own_category].append(init_src.cpu().numpy())  # tensorをnumpyに変換して保存
                    continue

                # 埋め込み層のinit_token_ids (<unusedx>) に該当する行を、まとめてinit_srcで初期化.
                E = self._get_model_info(model)[0]
                with torch.no_grad():
                    init_target_ids = torch.as_tensor([init_token_id], device=E.device, dtype=torch.long)
                    src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
                    E.index_copy_(dim=0, index=init_target_ids, source=src)
                
                print(f"Initialized category '{own_category}' (new token {tokenizer.decode(init_token_id)}, token_id: {init_token_id}) with {len(init_terms)} concepts: ... ({init_terms[-15:]}).")
        if return_init_vecs_only:
            return init_vecs_for_category
        return model



    def initialize_embeds_by_other_category_centroid_by_function(
        self, 
        model, 
        tokenizer, 
        category_to_concepts_for_vec, 
        category2initoken_ids, 
        initvec_func, 
        other_type,
        layer_idx=None,
        mix_layers=False,
        print_flag=False,
        return_init_vecs_only=False  # Trueなら、初期化されたモデルを返すのではなく、train_token2tokenidで指定されたtoken idsに対応する初期化vecのみを返す
        ):

        with open(category_similarity_path, 'r') as f:
            category_similarity = json.load(f)

        if return_init_vecs_only:
            # 初期化されたモデルを返すのではなく、train_token2tokenidで指定されたtoken idsに対応する初期化vecのみを返す場合は、初期化vecを辞書に保存していく
            init_vecs_for_category = defaultdict(dict)  # init_vecs_for_category[own_category][init_token_id] = init_vec for that category and token_id

        

        # main: 初期化対象token毎に毎回ランダムに選んだ他のカテゴリのCOGで初期化する
        for own_category, init_token_ids in category2initoken_ids.items():

            # 他のカテゴリのCOGで初期化する場合、category2initoken_ids外（今回新規概念として埋め込むtokenのあるカテゴリ以外）からも候補のカテゴリを選んで良い。そのためcategory_to_concepts_for_vecから直接取得する’
            if self.other_init_use_target_candidates_only:
                # category2initoken_idsのカテゴリのみを、他カテゴリ選出候補とする場合
                category_candidates = list(category2initoken_ids.keys())
            else:
                # loadProperNounData で取得できた全てのカテゴリを選出候補とする場合
                category_candidates = list(category_to_concepts_for_vec.keys())
            
            other_category_candidates = []
            for category in category_candidates:
                # カテゴリ内の固有名詞が、初期化vec作成における、「カテゴリ内固定成分 + token毎のランダムな成分」のための固有名詞数に足りるカテゴリのみ、この他カテゴリ選出候補に入れる。
                if len(category_to_concepts_for_vec[category]) >= self.propnoun_num_for_init_vec - 10  +  10 * len(init_token_ids):
                    other_category_candidates.append(category)

            # ** 他のカテゴリをランダムに選ぶ
            # other_categories = [c for c in other_category_candidates if c != own_category]  # terms数の不足するカテゴリを削除済みであるcategory_to_centroid_termsから他カテゴリを選ぶ
            # other_category = random.choice(other_categories)
            # print(f"Category '{own_category}' is initialized with centroid of other category: {other_category}. This is chosen from {len(other_categories)} categories: {other_categories[:20]}...")

            # ** 他のカテゴリを、同カテゴリから(cossimが)最も遠いカテゴリにする。カテゴリの意味の近さが初期化に影響するという対照実験のため。 **
            # other_category = category_similarity["classification"][own_category]["far"][0][0]
            if other_type == "far":
                other_category = category_similarity[own_category]['least_similar_by_mean'][0]['category']    # seed間のcossim平均が最も低いカテゴリを選ぶ
            elif other_type == "near":
                other_category = category_similarity[own_category]['most_similar_by_mean'][0]['category']    # seed間のcossim平均が最も高いカテゴリを選ぶ
            else:
                raise ValueError(f"Invalid other_type '{other_type}'. Must be 'far' or 'near'.")
            print(f"Category '{own_category}' is initialized with centroid of other category: {other_category}.")
            if other_category not in category_to_concepts_for_vec:
                raise ValueError(f"Other category '{other_category}' selected for initializing category '{own_category}' is not in category_to_concepts_for_vec. Change mode into 'dont_get_new_wiki_flag = False' to get new wiki pages.")

            init_terms_candidate = category_to_concepts_for_vec[other_category]
            init_terms_for_centroid = random.sample(init_terms_candidate, self.propnoun_num_for_init_vec-10)

            # centroid に使用済みのtermsをterms候補から削除
            init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))
            print(f"after sampling for centroid: {len(init_terms_candidate)} terms left. ({10 * len(init_token_ids)} terms are needed)")


            # 新規token毎に初期vecに変化をつけるためのnoiseとして、初期化対象の追加token毎に、10件をランダム選出
            # init_terms_for_random では、token毎に異なるtermを使いたい（token間で被るとvecの多様性が下がるため)
            # token毎に重複のない、初期vecのrandom成分用のtermsリスト: termsを重複無しで10個ずつランダムに分けたリストを作成する
            init_terms_for_random_list = random.sample(init_terms_candidate, 10 * len(init_token_ids))  # まず、元のリストをシャッフルする =カテゴリ内の固有名詞からランダムに(10 * token数)個選ぶ
            init_terms_for_random_chunks = [init_terms_for_random_list[i:i + 10] for i in range(0, len(init_terms_for_random_list), 10)]  # ランダムに選んだ固有名詞を、10個ずつのチャンクに分ける
            

            for i, init_token_id in enumerate(init_token_ids):
                init_terms_for_random = init_terms_for_random_chunks[i] # token毎のランダム性を保つための成分
                init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞 + ランダムvec用の固有名詞 を初期化vec作成に使用
                

                if model is not None:
                    # dataの状態を確認するために modelをNoneで呼び出すこともあるため、modelがNoneでない場合にのみ初期化処理を行う
                    init_target_ids = [init_token_id]
                    init_src = initvec_func(
                        model, 
                        tokenizer,
                        init_terms,
                        layer_idx=layer_idx, 
                        mix_layers=mix_layers, 
                        print_flag=print_flag
                    )

                    if debug_print_initterms_flag:
                        print(f"[INIT] own_category={own_category}, other_category={other_category}, token_id={init_token_id}")
                        print(f"[INIT] init_terms size={len(init_terms)}")
                        print(f"[INIT] init_src shape={tuple(init_src.shape)}")
                        print(f"[INIT] init_src finite={torch.isfinite(init_src).all().item()}")
                        print(f"[INIT] init_src has_nan={torch.isnan(init_src).any().item()}, has_inf={torch.isinf(init_src).any().item()}")
                        print(f"[INIT] init_src min={init_src.min().item()}, max={init_src.max().item()}, mean={init_src.mean().item()}, norm={init_src.norm().item()}")

                        if torch.isnan(init_src).any().item():
                            print(f"[ERROR] init_src contains NaN values for token_id {init_token_id} in category '{own_category}' initialized with other category '{other_category}'. Check the initvec_func for potential issues with NaN generation.")
                            # for initterm in init_terms:
                            #     print(f"[ERROR] init term: '{initterm}'")

                            # promptを全て表示するために、debug_print_initterms_prompt_flagをTrueにして再度initvec_funcを呼び出す
                            # debug_print_initterms_prompt_flag = True
                            init_src = initvec_func(
                                model, 
                                tokenizer, 
                                init_terms,
                                layer_idx=layer_idx, 
                                mix_layers=mix_layers, 
                                print_flag=print_flag,
                                debug_print_initterms_prompt_flag=True
                            )
                            raise ValueError(f"init_src contains NaN values for token_id {init_token_id} in category '{own_category}' initialized with other category '{other_category}'. Check the initvec_func for potential issues with NaN generation.")
                        

                    if return_init_vecs_only:
                        # 初期化されたモデルを返すのではなく、train_token2tokenidで指定されたtoken idsに対応する初期化vecのみを返す場合は、初期化vecを辞書に保存していく
                        init_vecs_for_category[own_category][init_token_id] = init_src.cpu().numpy()  # tensorをnumpyに変換して保存
                        continue

                    # 埋め込み層のinit_token_ids (<unusedx>) に該当する行を、まとめてinit_srcで初期化.
                    E = self._get_model_info(model)[0]
                    with torch.no_grad():
                        init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
                        src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
                        E.index_copy_(dim=0, index=init_target_ids, source=src)
                    
                print(f"===\n\tNew token {tokenizer.decode(init_token_id)} in category '{own_category}' is initialized with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
        
        if return_init_vecs_only:
            return init_vecs_for_category
        return model



    # ---------- random成分を除去したversionも実装 ----------

    def initialize_embeds_by_category_centroid_by_function_without_random(
        self, 
        model, 
        tokenizer, 
        category_to_concepts_for_vec, 
        category2initoken_ids, 
        initvec_func, 
        layer_idx=None,
        mix_layers=False, 
        print_flag=False
        ):
        # *** 初期vec作成用の固有名詞リストをカテゴリ毎に用意し、任意の関数で初期vecを作成する方法. ***

        for own_category, init_token_ids in category2initoken_ids.items():
            # *** このカテゴリに対応する初期化vec作成用の固有名詞リストで初期化vecを作成し、
            # このカテゴリに属す固有名詞(新規概念用)に割り当てたtokenのtoken idsの行を、その初期化vecで初期化する ***
            init_terms_candidate = category_to_concepts_for_vec[own_category]

            if len(init_terms_candidate) < self.propnoun_num_for_init_vec:
                # カテゴリ内の固有名詞が、初期化vec作成における、「カテゴリ内固定成分」のための固有名詞数に足りない場合は、エラーを出して終了
                raise ValueError(f"Not enough concepts in category '{own_category}' to sample for centroid vec. Required: at least {self.propnoun_num_for_init_vec}, Available: {len(init_terms_candidate)}. Please reduce the number of concepts needed for centroid vec or add more concepts to the category.")

            init_terms_for_centroid = random.sample(init_terms_candidate, self.propnoun_num_for_init_vec)  # カテゴリ内の固有名詞からランダムに(propnoun_num_for_init_vec)個選んで中心vecを作成

            # init_termsから、初期化vecを作成する
            init_src = initvec_func(
                model, 
                tokenizer, 
                init_terms_for_centroid,
                layer_idx=layer_idx, 
                mix_layers=mix_layers,
                print_flag=print_flag
            )

            # 埋め込み層のinit_token_ids (<unusedx>) に該当する行を、まとめてinit_srcで初期化.
            E = self._get_model_info(model)[0]
            with torch.no_grad():
                init_target_ids = torch.as_tensor(init_token_ids, device=E.device, dtype=torch.long)
                src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
                E.index_copy_(dim=0, index=init_target_ids, source=src)
            
            print(f"Initialized category '{own_category}' with {len(init_terms_for_centroid)} concepts: ... ({init_terms_for_centroid[-15:]}).")
        return model



    def initialize_embeds_by_other_category_centroid_by_function_without_random(
        self, 
        model, 
        tokenizer, 
        category_to_concepts_for_vec, 
        category2initoken_ids, 
        initvec_func, 
        layer_idx=None,
        other_type=None,
        mix_layers=False,
        print_flag=False
        ):

        with open(category_similarity_path, 'r') as f:
            category_similarity = json.load(f)

        
        # main: 初期化対象token毎に毎回ランダムに選んだ他のカテゴリのCOGで初期化する
        for own_category, init_token_ids in category2initoken_ids.items():

            # 他のカテゴリのCOGで初期化する場合、category2initoken_ids外（今回新規概念として埋め込むtokenのあるカテゴリ以外）からも候補のカテゴリを選んで良い。そのためcategory_to_concepts_for_vecから直接取得する’
            if self.other_init_use_target_candidates_only:
                # category2initoken_idsのカテゴリのみを、他カテゴリ選出候補とする場合
                category_candidates = list(category2initoken_ids.keys())
            else:
                # loadProperNounData で取得できた全てのカテゴリを選出候補とする場合
                category_candidates = list(category_to_concepts_for_vec.keys())
            
            other_category_candidates = []
            for category in category_candidates:
                # カテゴリ内の固有名詞が、初期化vec作成における、「カテゴリ内固定成分」のための固有名詞数に足りるカテゴリのみ、この他カテゴリ選出候補に入れる。
                if len(category_to_concepts_for_vec[category]) >= self.propnoun_num_for_init_vec:
                    other_category_candidates.append(category)

            # ** 他のカテゴリをランダムに選ぶ
            # other_categories = [c for c in other_category_candidates if c != own_category]  # terms数の不足するカテゴリを削除済みであるcategory_to_centroid_termsから他カテゴリを選ぶ
            # other_category = random.choice(other_categories)
            # print(f"Category '{own_category}' is initialized with centroid of other category: {other_category}. This is chosen from {len(other_categories)} categories: {other_categories[:20]}...")

            # ** 他のカテゴリを、同カテゴリから(cossimが)最も遠いカテゴリにする。カテゴリの意味の近さが初期化に影響するという対照実験のため。 **
            # other_category = category_similarity["classification"][own_category]["far"][0][0]
            if other_type == "far":
                other_category = category_similarity[own_category]['least_similar_by_mean'][0]['category']    # seed間のcossim平均が最も低いカテゴリを選ぶ
            elif other_type == "near":
                other_category = category_similarity[own_category]['most_similar_by_mean'][0]['category']    # seed間のcossim平均が最も高いカテゴリを選ぶ
            else:
                raise ValueError(f"Invalid other_type '{other_type}'. Must be 'far' or 'near'.")
            print(f"Category '{own_category}' is initialized with centroid of other category: {other_category}.")
            if other_category not in category_to_concepts_for_vec:
                raise ValueError(f"Other category '{other_category}' selected for initializing category '{own_category}' is not in category_to_concepts_for_vec. Change mode into 'dont_get_new_wiki_flag = False' to get new wiki pages.")

            init_terms_for_centroid = random.sample(category_to_concepts_for_vec[other_category], self.propnoun_num_for_init_vec)

            if model is not None:
                # dataの状態を確認するために modelをNoneで呼び出すこともあるため、modelがNoneでない場合にのみ初期化処理を行う
   
                init_src = initvec_func(
                    model, 
                    tokenizer, 
                    init_terms_for_centroid,
                    layer_idx=layer_idx, 
                    mix_layers=mix_layers, 
                    print_flag=print_flag
                )

                # 埋め込み層のinit_token_ids (<unusedx>) に該当する行を、まとめてinit_srcで初期化.
                E = self._get_model_info(model)[0]
                with torch.no_grad():
                    init_target_ids = torch.as_tensor(init_token_ids, device=E.device, dtype=torch.long)
                    src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
                    E.index_copy_(dim=0, index=init_target_ids, source=src)
                
            print(f"===\n\tCategory '{own_category}' is initialized with layer {layer_idx}'s hidden state of {len(init_terms_for_centroid)} concepts: ... ({init_terms_for_centroid[-15:]}), from other category '{other_category}'.")
        return model







    # ========================== initvec_func として入力する任意の関数一覧の実装 ==========================
    def make_initvec_by_wiki_summary_and_hidden_state(
        self,
        model, 
        tokenizer, 
        init_terms,
        layer_idx=None, 
        mix_layers=False,
        print_flag=False,
        debug_print_initterms_prompt_flag=False, # [memo] この引数は他のinitvec_funcには付けていないため注意。他のinitvec_funcでこの引数をTrueにしても、エラーが出るだけ。
        return_init_vecs_list_only=False  # Trueなら、init vecsを平均poolする前の、term_vecsのリストを返す
        ):
        # *** 初期vecを、固有名詞毎のwikiのsummary文入力時の隠れ状態から作る。 ***

        E, num_hidden_layers = self._get_model_info(model)
        if print_flag:
            print(f"⭐️num_hidden_layers: {num_hidden_layers}")

        # 1. term毎にpromptを用意
        prompt_lst = []
        for term in init_terms:
            if term.strip() == "":
                continue
            # ** このterm(prop noun)を説明する wiki page の summary を取得し、前処理を行う **
            # 辞書にまだ保存されていなければ、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに格納する
            if term not in self.propnoun_to_wikisummary:
                # self._load_wikisummary(term)
                self.propnoun_to_wikisummary[term] = load_wikisummary(term, wiki_pages_dir)
            summary = self.propnoun_to_wikisummary.get(term)

            # 短すぎor長すぎるsummaryがあるため、最初の数文だけをsummaryとして使用する. (30~300単語に収まるように調整) 30単語未満のsummaryは、十分な情報が得られない可能性があるため、初期化vecの計算に使用しない. 
            # min_words, max_words = 30, 300 # 30->50に変更すると、そこまで長いsummaryが少ないようで、init vecが0vecとなりlossがNanになってしまった。minは30でキープする
            summary = get_first_few_sentences(summary, self.min_words, self.max_words)
            if summary is None:
                print(f"'{term}' のWikipedia summaryは、{self.min_words} ~ {self.max_words}単語の範囲内に収まらないため、スキップします。") # 最初の100文字だけ表示
                # min_words ~ max_wordsの範囲内にないsummaryを持つpropnounは次回もwiki apiで呼び出すことがないよう記録しておく
                with open(os.path.join(project_root, "data", f"propnouns_summary_outofrange_{self.min_words}_{self.max_words}.txt"), "a") as f:
                    f.write(term + "\n")
                continue

            if debug_print_initterms_prompt_flag:
                print(f"[INIT] term: '{term}'\n summary: '{summary}' \n(length in words: {len(summary.split())})\n")

            if self.repeat_prompt:
                # *** 初期vecを、固有名詞毎のwikiのsummary文を2回入力して、2回目の文内token位置の隠れ状態から作る場合: https://openreview.net/forum?id=Ahlrf2HGJR の手法 ***
                summary = repeat_text(summary, 2)
            prompt_lst.append(summary)
        
        if len(prompt_lst) == 0:
            raise ValueError(f"No valid summaries found for the given terms. Cannot create initialization vector.")
        print(f"Created prompt list for terms: {prompt_lst[:5]}... (total {len(prompt_lst)})")


        # 2. 各語句ベクトルを作成する. wiki summary をモデルに入力し、pool_hs_type に応じてsummary中の全token/最終token/eos位置の隠れ状態をその語句のベクトルとする
        # term_vecs = self._extract_hidden_states(
        #     model, 
        #     tokenizer, 
        #     prompt_lst, 
        #     batch_size=BATCH_SIZE, 
        #     layer_index=layer_idx, 
        #     mix_layers=True, 
        #     print_flag=False
        # )
        term_vecs = extract_hidden_states(
            model, 
            tokenizer, 
            prompt_lst, 
            pool_hs_type=self.pool_hs_type,
            batch_size=BATCH_SIZE, 
            layer_index=layer_idx, 
            mix_layers=True, 
            print_flag=False
        )
        term_vecs = torch.from_numpy(term_vecs) # extract_hidden_statesでは numpy で返ってくるので tensor に変換する
        
        if return_init_vecs_list_only:
            return term_vecs
        

        # 3. term vec間の平均vecを計算
        sum_vec = term_vecs.sum(dim=0)  # バッチ内のterm_vecを合計して、sum_vecとする
        if sum_vec.norm().item() == 0.0 or len(prompt_lst) == 0:
            raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
        init_src = sum_vec / len(prompt_lst)

        # 4. 微小ノイズを加える
        d = init_src.shape[0]
        noise = torch.randn(d, device=init_src.device, dtype=init_src.dtype)    # torch.randn(d, device=E.device, dtype=E.dtype)

        # 各行をL2正規化して「方向だけランダム」にする
        eps = 1e-12
        noise = noise / noise.norm(p=2, dim=0, keepdim=True).clamp_min(eps)

        # ノイズの大きさを、重心ノルムのごく一部にする
        noise_scale = NOISE_SCALE   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
        init_norm = init_src.norm(p=2).clamp_min(eps)
        noise = noise * (init_norm * noise_scale)

        # 重心 + 微小ノイズ
        init_src = init_src + noise

        # 5. ノルムを語彙中央値に合わせる [memo] hidde stateのノルムは埋め込み層のノルムと大きく異なる可能性があるため、ノルムを合わせる
        target_norm = E.norm(dim=1).median().item()  # 埋め込み行のノルムの中央値をターゲットノルムとする
        init_src_norm = init_src.norm().item()
        if init_src_norm > 0:
            init_src = init_src / init_src_norm * target_norm  # ターゲットノルムに合わせてスケーリング
        
        return init_src.to(E.device)










    # ========================== その他の細かい処理 ==========================

    def _get_model_info(self, model):
        try:
            E = model.model.embed_tokens.weight
            num_hidden_layers = model.config.num_hidden_layers
        except AttributeError:
            E = model.model.language_model.embed_tokens.weight
            num_hidden_layers = model.config.text_config.num_hidden_layers
        return E, num_hidden_layers
    

    # def _get_mix_layers(self, layer_idx, num_hidden_layers):
    #     if layer_idx == -1 or layer_idx == num_hidden_layers:
    #         mixed_layers = [-1, -2, -3]                           # 最終層とその前の2層を平均する
    #     elif layer_idx == 0:
    #         mixed_layers = [0, 1, 2]                              # 最初の層とその後の2層を平均する
    #     else:
    #         mixed_layers = [layer_idx-1, layer_idx, layer_idx+1]  # 指定層の前後3層を平均する
    #     # else:
    #     #     raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be -1, 0, or a positive integer less than num_hidden_layers.")
    #     return mixed_layers


    def _extract_term_vec(self, inputs, layer_idx, num_hidden_layers, all_hs=None, layer_hs=None, mix_layers=False):
        """
        各場合に応じてterm_vecを抽出する関数. pool_hs_type と mix_layers の組み合わせに応じて、term_vecの抽出方法が変わる.
        Args:
        * all_hs: 全層の隠れ状態のリスト. 各要素は [batch, seq_len, d].  前後３層mix用。
        * layer_hs: 指定層の隠れ状態. [batch, seq_len, d].  単一層用の引数。
        * inputs: モデルへの入力. attention_maskを使って語句中のtoken数を計算するために必要.
        * layer_idx: 現在処理している層のindex
        * num_hidden_layers: モデルの隠れ層の総数. mix_layers=Trueの場合に、前後の層を計算するために必要.
        * mix_layers: Trueなら前後3層の隠れ状態を平均してterm_vecとする。Falseなら単一層の隠れ状態をterm_vecとする。

        Return:
        * term_vec: 語句のベクトル. [d]
        """
        seq_len = inputs["attention_mask"].sum(dim=1).item() 
        last_token_idx = seq_len - 1                    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
    
        # *** 前後3層mixでterm_vecを作る場合 ***
        if mix_layers:
            mix_layers = get_mix_layers(layer_idx, num_hidden_layers)
            # ** 前後3層の隠れ状態を平均する:
            layer_hs_mix = torch.stack(
                [all_hs[lid] for lid in mix_layers],
                dim=0
            )  # 指定層の出力 [3, 1, seq_len, d]

            if self.pool_hs_type in ["last_token", "eos"]:
                # ** term中の最後のtokenのみでterm_vecを作る場合:
                term_vec = layer_hs_mix[:, 0, last_token_idx, :].mean(dim=0)   # [1, seq_len, d] -> [d] 前後3層の最後のtokenの隠れ状態を平均する.
            if self.pool_hs_type == "mean_pool":
                # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                term_vec = layer_hs_mix[:, 0, :seq_len, :].mean(dim=1).mean(dim=0)    # [3, 1, seq_len, d] -> [seq_len, d] -> [d]
            # else:
                # raise ValueError(f"Unknown pool_hs_type: {pool_hs_type}")
        
        # ***** 単一層でterm_vecを作る場合 *****
        else:            
            if self.pool_hs_type in ["last_token", "eos"]:
                # ** term中の最後のtokenのみでterm_vecを作る場合:
                term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
            elif self.pool_hs_type == "mean_pool":
                # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [d]
            # else:
                # raise ValueError(f"Unknown pool_hs_type: {pool_hs_type}")
        return term_vec



    # @torch.no_grad()
    # def _extract_hidden_states(
    #     self, 
    #     model, 
    #     tokenizer, 
    #     text_list, 
    #     batch_size=8, 
    #     layer_index=-1, 
    #     mix_layers=True, 
    #     print_flag=False):
    #     """
    #     各テキストの末尾にEOSを明示的に追加し、
    #     EOSトークン位置の hidden state を返す。

    #     Returns:
    #         np.ndarray of shape (N, hidden_dim)
    #     """

    #     E, num_hidden_layers = self._get_model_info(model)


    #     all_vecs = []
    #     for i in range(0, len(text_list), batch_size):
    #         batch_texts = text_list[i:i + batch_size]
    #         if print_flag:
    #             print(f"Processing batch {i // batch_size + 1}/{(len(text_list) + batch_size - 1) // batch_size} for hidden state extraction...")

    #         if self.pool_hs_type == "eos":
    #             # EOS を明示的に末尾へ追加
    #             batch_texts = [text + tokenizer.eos_token for text in batch_texts]
            
    #         inputs = tokenizer(
    #             batch_texts,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=False,   # truncation=Trueとすると、どこかでtokenが切り捨てられてしまい、self.repeat_prompt=True・mean_poolの時に隠れ状態を平均する対象のトークン位置がずれてしまう。
    #             add_special_tokens=False #last_token_is_eos#LAST_TOKEN_IS_EOS, -> pool_hs_type == "eos"の場合は明示的にeosを追加済みなので、ここをTrueにするとeosが重複して2つ付く可能性がある。そのためここはFalseで良い。
    #         ).to(model.device) 

    #         input_ids = inputs["input_ids"]
    #         attention_mask = inputs["attention_mask"]

    #         with torch.no_grad():
    #             outputs = model(**inputs, output_hidden_states=True)
    #             # hidden_states は tuple:
    #             # 0: embedding出力, 1..L: 各層出力
    #             all_hs = outputs.hidden_states

    #         if mix_layers:
    #             # ** 前後3層mixでterm_vecを作る場合 **
    #             target_layer_hs = torch.stack(
    #                 [all_hs[lid] for lid in self._get_mix_layers(layer_index, num_hidden_layers)],
    #                 dim=0
    #             )  # 指定層の出力 [3, batch_size, seq_len, d]
    #         else:
    #             # ** 単一層でterm_vecを作る場合 **
    #             target_layer_hs = all_hs[layer_index].unsqueeze(0)      # (1, batch_size, seq_len, d)

            
            
    #         # *** pool_hs_type に応じて、vectorを抽出する位置を決定 ***
    #         if self.pool_hs_type == "eos":
    #             # 各系列について EOS token の最後の出現位置を取る
    #             eos_mask = (input_ids == tokenizer.eos_token_id)

    #         for t_idx in range(input_ids.size(0)):

    #             # ** 1 が立っている位置を取得 **
    #             # e.g.  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] -> valid_pos = [6, 7, 8, 9, 10, 11] 
    #             valid_pos = torch.nonzero(attention_mask[t_idx], as_tuple=False).squeeze(-1)
    #             # print(f"valid_pos for batch {t_idx}: {valid_pos}")
    #             if valid_pos.numel() == 0:
    #                 # 全部 padding の場合
    #                 pos_begin = 0
    #                 pos_end = 0
    #             else:
    #                 pos_begin = valid_pos[0].item()
    #                 pos_end = valid_pos[-1].item() + 1   # slice用に end は exclusive

                    
    #             if self.pool_hs_type == "eos":
    #                 eos_positions = torch.where(eos_mask[t_idx])[0]
    #                 if len(eos_positions) == 0:
    #                     raise ValueError(f"EOS token が見つかりません: {batch_texts[t_idx]}")
    #                 eos_pos = eos_positions[-1].item()
    #                 pos_begin = eos_pos
    #                 pos_end = eos_pos + 1

    #             elif self.pool_hs_type == "last_token":
    #                 pos_begin = pos_end - 1

    #             elif self.pool_hs_type == "mean_pool":
    #                 if self.repeat_prompt:
    #                     # *** 初期vecを、固有名詞毎のwikiのsummary文を2回入力して、2回目の文内token位置の隠れ状態から作る場合: https://openreview.net/forum?id=Ahlrf2HGJR の手法 ***
    #                     # wiki summaryを繰り返してプロンプトとする場合は、2回目の文のみの隠れ状態を平均する
    #                     pos_begin_second_sent = (pos_begin + pos_end) // 2 # == pos_begin + (pos_end - pos_begin) / 2
    #                     pos_begin = pos_begin_second_sent
    #                 else:
    #                     pass # デフォルトは、入力文全体の隠れ状態の平均を取る
    
    #             else:
    #                 raise ValueError(f"Unknown pool_hs_type: {self.pool_hs_type}")

    #             # ** vectorを抽出 **
    #             # vec = layer_hs[t_idx, pos_begin:pos_end, :].mean(dim=0)  # (H,)
    #             term_vec = target_layer_hs[:, t_idx, pos_begin:pos_end, :].mean(dim=0).mean(dim=0)  # (mix層数, batch_size, seq_len, d) -> batch内のt_idxに該当する層&平均対象のtoken位置を指定: (mix層数, meanpool対象token数, d) -> 前後3層を平均した隠れ状態のうち、valid_tokenの部分を平均する: (meanpool対象token数, d) -> meanpool対象tokenを平均する: (d)
    #             all_vecs.append(term_vec.detach().cpu())

    #             if print_flag:
    #                 # どの位置のtokenの隠れ状態が使われるのかを確認するためのprint文
    #                 print(f"target_layer_hs shape: {target_layer_hs.shape} -> term_vec: {term_vec.shape}")  # (mix層数, batch_size, seq_len, d)
    #                 print(f"pos_begin: {pos_begin}, pos_end: {pos_end}")
    #                 print(f"\tattention_mask: {attention_mask[t_idx]},\n\t valid_pos: {valid_pos}, \n\t valid part in batch_text: {input_ids[t_idx][pos_begin:pos_end]}")

    #     return torch.stack(all_vecs, dim=0) # np.stack(all_vecs, axis=0)

        

