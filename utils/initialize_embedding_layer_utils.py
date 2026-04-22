
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

N_COMPONENTS = 2
NOISE_SCALE = 5e-3   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
LAMBDA_ = 0.0   # global_vecを引くときの重み. 0.1あたりから試す. 0.1だと少し改善するが、0.2だとさらに改善する。 0.3はまだ試していないが後で試す
# LAST_TOKEN_IS_EOS = True  # termの最後のtokenが<EOS>であるかどうか。Trueなら、<eos>トークン位置を最終トークン位置とする。Falseなら、term内の最後のtokenを最終トークン位置とする。
BATCH_SIZE = 4 #16 #8


wiki_pages_dir = os.path.join(project_root, "data", "wiki_pages")
category_similarity_path = os.path.join(project_root, 'data', 'cossim_bw_categories', 'aggregated_near_far_analysis_across_seeds.json')
debug_initvec_flag = True

# *************************** func ***************************

def compute_pca_components(X, n_components=10):
    """
    X: [N, d]
    return:
        mean_vec: [d]
        pcs: [n_components, d]   # 上位主成分
        explained_ratio: [n_components]
    """
    X = X.float()

    # 1. 平均中心化
    mean_vec = X.mean(dim=0, keepdim=True)      # [1, d]
    X_centered = X - mean_vec                   # [N, d]

    # 2. SVD
    # X_centered = U S Vh
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # 主成分方向
    pcs = Vh[:n_components]   # [k, d]

    # 寄与率
    eigvals = (S ** 2) / (X_centered.shape[0] - 1)
    explained_ratio = eigvals[:n_components] / eigvals.sum()

    return mean_vec.squeeze(0), pcs, explained_ratio



def save_pca_components(save_path, mean_vec, pcs, explained_ratio, meta=None):
    """
    PCA結果を保存する

    Args:
        save_path: 保存先ファイルパス (.pt 推奨)
        mean_vec: [d]
        pcs: [k, d]
        explained_ratio: [k]
        meta: 追加情報を入れたいときのdict
              例: {"layer_idx": 10, "pool_hs_type": "mean", "mix_layers": True}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    payload = {
        "mean_vec": mean_vec.detach().cpu(),
        "pcs": pcs.detach().cpu(),
        "explained_ratio": explained_ratio.detach().cpu(),
        "meta": meta if meta is not None else {},
    }

    torch.save(payload, save_path)
    print(f"Saved PCA components to: {save_path}")


def load_pca_components(load_path, map_location="cpu"):
    """
    保存済みPCA結果を読み込む

    Returns:
        mean_vec: [d]
        pcs: [k, d]
        explained_ratio: [k]
        meta: dict
    """
    payload = torch.load(load_path, map_location=map_location)

    mean_vec = payload["mean_vec"]
    pcs = payload["pcs"]
    explained_ratio = payload["explained_ratio"]
    meta = payload.get("meta", {})

    print(f"Loaded PCA components from: {load_path}")
    return mean_vec, pcs, explained_ratio, meta





class EmbedInitializer:
    def __init__(self, 
                 model_name, 
                 save_mem_dir, 
                 init_vec_type, 
                 train_target_category_lst, 
                 propnoun_num_for_init_vec, 
                 model, 
                 tokenizer, 
                 pool_hs_type,
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
        # self.global_primary_vec_by_mixed_layer = {} # layer代表vecはその前後との平均の主成分
        self.global_primary_vec_by_layer = {}       # layer代表vecはその単層の主成分
        self.num_propNouns_in_cat_for_globalHSMean = 100

        self.propnoun_to_wikisummary = {}   # これが必要な関数を実行する際に、中身が空なら読み込む。少し時間とメモリを食うので不要なら読み込まない。
        self.repeat_prompt = False   # promptを2回繰り返すプロンプトを使うかどうかのフラグ。
        self.min_words = min_words
        self.max_words = max_words

        # self.category_to_other_category = {} # other系の初期化の場合、学習対象のカテゴリ毎に、どの他カテゴリを初期化に使うかを固定するための辞書。新規概念毎に異なるカテゴリを使って初期化すると、初期vec間の多様性が同カテゴリ初期化時に比べて大きくなり、不公平になるため。
        # self.other_init_use_the_same_other_category = True # Trueなら、全カテゴリのother系初期化に同じカテゴリを使う。Falseなら、カテゴリ毎にother系初期化に使うカテゴリを変える。
        self.other_init_use_target_candidates_only = True    # Trueなら、other系の初期化に、学習対象カテゴリの候補カテゴリを使う。Falseなら、学習対象カテゴリの候補カテゴリは使わず、使用可能な全カテゴリからランダムに選ぶ。つまり target_concepts.json中のカテゴリのみ or loadProperNounDataで集めた全てのカテゴリ
        
        # self.category_to_concepts_for_other = {} # 他カテゴリで初期化する場合は、学習データがないカテゴリも参照したい。
        # random.seed(seed)# 呼び出し側でseed固定するので不要そう


        # ***** global vector の作成が必要なinit_vec_typeの場合はここで作成 *****
        # ** debiaseあり
        # * global_hidden_meanを、自カテゴリ含む全部のカテゴリの固有名詞のベクトルの平均で計算する方法
        if init_vec_type in [
            'categoryCentroid_by_DebiasedHiddenState', 'otherCategoryCentroid_by_DebiasedHiddenState',
            'categoryCentroid_by_DebiasedHSMixed', 'otherCategoryCentroid_by_DebiasedHSMixed']:
            self.calculateGlobalHiddenStateMean(model, tokenizer)            
        # * global_hidden_meanを、全カテゴリ中の、自カテゴリ以外全部の固有名詞から計算する方法
        elif init_vec_type in ['CatCentroid_by_OthCatDebiasedHSMixed', 'otherCatCentroid_by_OthCatDebiasedHSMixed']:
            self.calculateGlobalHSMean_by_OtherCatPropNouns(model, tokenizer, mix_layers=True)   
        
        # ** debiaseあり・主成分ベクトルを使う方法. 全カテゴリに共通する主成分成分を、初期化用vecから引く方法. カテゴリ間のvecの方向に差をつけようとした。
        # * 前後3層の隠れ状態を平均する
        elif init_vec_type in ['CatCent_by_GlbPrimDebiasedHSMixed', 'otherCatCent_by_GlbPrimDebiasedHSMixed']:
            self.calculateGlobalHSMean_by_GlbPrimDebiasedHSMixed(model, tokenizer, n_components=N_COMPONENTS, mix_layers=True)   # global_hidden_meanを、全カテゴリの主成分で計算する方法. mix_layers=Trueは、指定層の前後3層の隠れ状態を平均してterm_vecを作る方法. mix_layers=Falseは、指定層の隠れ状態のみでterm_vecを作る方法. どちらも試す
        # * 単一層の隠れ状態を使う
        elif init_vec_type in ['CatCent_by_GlbPrimDebiasedHS', 'otherCatCent_by_GlbPrimDebiasedHS']:
            self.calculateGlobalHSMean_by_GlbPrimDebiasedHSMixed(model, tokenizer, n_components=N_COMPONENTS, mix_layers=False)   # global_hidden_meanを、全カテゴリの主成分で計算する方法. mix_layers=Trueは、指定層の前後3層の隠れ状態を平均してterm_vecを作る方法. mix_layers=Falseは、指定層の隠れ状態のみでterm_vecを作る方法. どちらも試す
    
        # ** debias なし
        if init_vec_type in ["CatCent_by_WikiSummaryRepeatHSMixed", "otherCatCent_by_WikiSummaryRepeatHSMixed"]:
            self.repeat_prompt = True
        
        else:
            print(f"init_vec_type: {init_vec_type} does not require global hidden mean calculation. Skipping that step.")


        if model is not None:
            # dataの状態だけprintするために、model=Noneとすることがある。その場合はmodel関連の処理はskipする。
            # ***** self.save_mem_dir にこの訓練のパラメータを辞書保存 *****
            path = os.path.join(self.save_mem_dir, "embed_initializer_params.json")
            params_to_save = {
                "model_name": self.model_name,
                "init_vec_type": self.init_vec_type,
                "train_target_category_lst": self.train_target_category_lst,
                "propnoun_num_for_init_vec": self.propnoun_num_for_init_vec,
                # "seed": seed,
                "pool_hs_type": self.pool_hs_type,

                "train_date_time":  datetime.now().strftime("%Y%m%d%H%M%S"),
                "num_of_global_vec_primary_components": N_COMPONENTS,
                "noise_scale": NOISE_SCALE,
                "lamda_": LAMBDA_,
                # "last_token_is_eos": self.last_token_is_eos # LAST_TOKEN_IS_EOS,
                # "other_init_use_the_same_other_category": self.other_init_use_the_same_other_category,
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
            print_flag=False
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


            # *** (Type1) カテゴリ内の90(propnoun_num_for_init_vec-10)固有名詞(固定)を平均化したvec + カテゴリ内の10固有名詞をランダムに選んで平均化したvec を足し合わせたvecで初期化 ***
            "category_centroid_plus_random": lambda: self.initvec_by_other_category_centroid_plus_random(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, print_flag=print_flag
            ),

            # [WIP] 未実装(不要) *** (Type1) category_centroid_plus_random の対。他のカテゴリの中心vec + 他のカテゴリのランダムvecで初期化. 新概念 apple の初期化にvehicleカテゴリの代表ベクトルを利用するなど ***

            # ============================================================================================    
            # COG: Center Of Gravity. 
            # *** 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
            # vec_propnoun = mean(token_vecs_in_propnoun) -> 各カテゴリの初期化vec = mean(vec_propnoun_in_category)
            "category_COG_by_simple_mean": lambda: self.initvec_by_category_COG_by_simple_mean(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, print_flag=print_flag
            ),

            # *** 他のカテゴリのCOGで初期化. 例えば、動物カテゴリの新規概念を、場所カテゴリの固有名詞のベクトルの平均で初期化するなど. category_COGに対する比較用 ***
            # どのカテゴリのCOGで初期化するかは、初期化対象token毎に毎回ランダムに選ぶ 
            # (category_COGでは同一カテゴリを同じCOGで初期化していたのに対し、こちらは毎回ランダムに選ぶため、同一カテゴリ内でもtoken毎に異なるCOGで初期化されることになる)
            # (なるべく色々なカテゴリのCOGで初期化するため、ランダムに選ぶ方式にしている)
            "other_category_COG_by_simple_mean": lambda: self.initvec_by_other_category_COG_by_simple_mean(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, print_flag=print_flag
            ),


            # ============================================================================================ 
            # 2026/03/19 
            # *** (Type2) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
            # Type2: 各prop nounをモデルに入力し、語句内の最終token位置における指定層の隠れ状態をその固有名詞のベクトルとする方法。
            # vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)
            "category_centroid_by_hidden_state_mean": lambda: self.initialize_embeds_by_category_centroid_by_function(  # self.initvec_by_category_centroid_by_hidden_state_mean(, initVecWithMeanVecOfTermHiddenStates
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_hidden_state, 
                layer_idx=layer_idx,
                mix_layers=False,
                print_flag=True
            ),

            # *** (Type2) 他のカテゴリのCOGで初期化. category_centroid_by_hidden_state_mean の対。例えば、動物カテゴリの新規概念を、場所カテゴリの固有名詞のベクトルの平均で初期化するなど. ***
            # * カテゴリ毎のcentroid vec作成用の固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
            "other_category_centroid_by_hidden_state_mean": lambda: self.initialize_embeds_by_other_category_centroid_by_function( # self.initvec_by_other_category_centroid_by_hidden_state_mean(, initVecWithMeanVecOfTermHiddenStates
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_hidden_state, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=False, 
                print_flag=True
            ),

            # ============================================================================================ 
            # 2026/03/20
            # ***  (Type3) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
            # 更に、等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
            # 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
            # また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。(type2では、最後のsub-tokenの隠れ状態のみ)
            # vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)
            "categoryCentroid_by_DebiasedHiddenState": lambda: self.initialize_embeds_by_category_centroid_by_function( # self.initvec_by_category_centroid_by_debiased_hidden_state(, initVecWithMeanVecOfDebiasedTermHiddenStates
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                mix_layers=True, 
                print_flag=True
            ),
            # *** (Type3) 他のカテゴリのCOGで初期化. category_centroid_by_debiased_hidden_state の対。***
            "otherCategoryCentroid_by_DebiasedHiddenState": lambda: self.initialize_embeds_by_other_category_centroid_by_function( # self.initvec_by_other_category_centroid_by_debiased_hidden_state(, initVecWithMeanVecOfDebiasedTermHiddenStates
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True, 
                print_flag=True
            ),


            
            # ============================================================================================
            # [memo] 
            # 'CatCentroid_by_OthCatDebiasedHSMixed'と'CatCent_by_GlbPrimDebiasedHSMixed', 'CatCent_by_GlbPrimDebiasedHS', 'categoryCentroid_by_DebiasedHSMixed',
            # 'otherCatCentroid_by_OthCatDebiasedHSMixed'と'otherCatCent_by_GlbPrimDebiasedHSMixed', 'otherCatCent_by_GlbPrimDebiasedHS', 'otherCategoryCentroid_by_DebiasedHSMixed',
            # が全く同じ処理なのは、make_initvec_by_terms_with_debiased_hidden_state_by_global_vec()内部でinit_vec_typeに応じて参照するglobal_vecを切り替えているから。
            
            # 2026/03/21
            # *** (Type4) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
            # type3に加え、指定した層だけでなく、その前後の層と平均したものをterm_vecとする。例えば、layer_idx=5を指定した場合、layer4, layer5, layer6の隠れ状態の平均をterm_vecとする。
            "categoryCentroid_by_DebiasedHSMixed": lambda: self.initialize_embeds_by_category_centroid_by_function( # self.initvec_by_category_centroid_by_debiased_and_mixed_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                mix_layers=True, 
                print_flag=True
            ),
            # *** (Type4) 他のカテゴリのCOGで初期化. category_centroid_by_debiased_hidden_state の対。***
            "otherCategoryCentroid_by_DebiasedHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function( # self.initvec_by_other_category_centroid_by_debiased_and_mixed_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True, 
                print_flag=True
            ),


            # ======================================================
            # 2026/03/21 -2
            # *** (Type5) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
            # type4のglobal vecの作り方を変更した。
            # * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
            #   * この時、global_hidden_mean は、他カテゴリ全部の平均で計算する。（Type4は自カテゴリも含めた平均）
            "CatCentroid_by_OthCatDebiasedHSMixed": lambda: self.initialize_embeds_by_category_centroid_by_function( # self.initvec_by_category_centroid_by_global_vec_debiased_and_mixed_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                mix_layers=True, 
                print_flag=True
            ),
            # *** Type5の対 ***
            "otherCatCentroid_by_OthCatDebiasedHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function( # self.initvec_by_other_category_centroid_by_global_vec_debiased_and_mixed_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True, 
                print_flag=True
            ),


            # ========================================================
            # 2026/03/21 -3
            # *** debiaseあり・主成分ベクトルを使う方法. 全カテゴリに共通する主成分成分を、初期化用vecから引く方法. カテゴリ間のvecの方向に差をつけようとした。***
            
            # * 前後3層の隠れ状態を平均する mix_layers = True
            "CatCent_by_GlbPrimDebiasedHSMixed": lambda: self.initialize_embeds_by_category_centroid_by_function( # self.initvec_by_category_centroid_by_global_vec_debiased_and_mixed_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                mix_layers=True,
                print_flag=True
            ),
            "otherCatCent_by_GlbPrimDebiasedHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function( # self.initvec_by_other_category_centroid_by_global_vec_debiased_and_mixed_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True,
                print_flag=True
            ),

            # * 単一層の隠れ状態を使う mix_layers = False
            "CatCent_by_GlbPrimDebiasedHS": lambda: self.initialize_embeds_by_category_centroid_by_function( # self.initvec_by_category_centroid_by_global_vec_debiased_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                mix_layers=False,
                print_flag=True
            ),
            "otherCatCent_by_GlbPrimDebiasedHS": lambda: self.initialize_embeds_by_other_category_centroid_by_function( # self.initvec_by_other_category_centroid_by_global_vec_debiased_hidden_state(, initVecWithGlobalVecDebiasedTermHS
                model, 
                tokenizer, 
                category_to_concepts_for_vec, 
                category2initoken_ids, 
                initvec_func=self.make_initvec_by_terms_with_debiased_hidden_state_by_global_vec, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=False,
                print_flag=True
            ),

            # ============================================================================================
            # 2026/04/04
            # 初期vecを、カテゴリの固有名詞ではなく、wikiのsummary文を入力した時の最終tokenの隠れ状態から作る。
            # 固有名詞を構成する単語は、別の意味を持っていることが多いのではないかという懸念から。例えば、board_gameカテゴリの'Unlock!'というゲーム名は、単語だけ見れば鍵を開けるという意味だが、ここではゲームを代表するベクトルを作りたいため、単語の意味と欲しい意味が異なる。そのためwiki説明文の利用を試す。
            
            # * 単層の隠れ状態を使う
            "CatCent_by_WikiSummaryHS": lambda: self.initialize_embeds_by_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                mix_layers=False,
                print_flag=True
            ),
            "otherCatCent_by_WikiSummaryHS": lambda: self.initialize_embeds_by_other_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=False,
                print_flag=True
            ),
            #  * 前後3層の隠れ状態を平均する
            "CatCent_by_WikiSummaryHSMixed": lambda: self.initialize_embeds_by_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                mix_layers=True,
                print_flag=True
            ),
            "otherCatCent_by_WikiSummaryHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True,
                print_flag=True
            ),

            # ============================================================================================
            # 2026/04/09
            # wiki summaryを2回繰り返してプロンプトとし、2文目の隠れ状態から初期化vecを作成する方法: https://openreview.net/forum?id=Ahlrf2HGJR の手法. src_visualize/plot_gemma_hidden_states_3d.py でカテゴリ同士が他の手法よりも分離できていたため.
            "CatCent_by_WikiSummaryRepeatHSMixed": lambda: self.initialize_embeds_by_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                mix_layers=True,
                print_flag=True
            ),
            "otherCatCent_by_WikiSummaryRepeatHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True,
                print_flag=True
            ),

            "nearCatCent_by_WikiSummaryRepeatHSMixed": lambda: self.initialize_embeds_by_other_category_centroid_by_function(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                other_type="near", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True,
                print_flag=True
            ),

            # -------------------------------------------------------------------------------------------
            # 2026/04/19
            # wiki summary を2回繰り返してプロンプトとし、2文目の隠れ状態から初期化vecを作成する方法。ただし、token毎のランダム成分は入れず、カテゴリの中心ベクトルのみで初期化する。
            "CatCent_by_WikiSummRepeatHSMix_noRand": lambda: self.initialize_embeds_by_category_centroid_by_function_without_random(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                mix_layers=True,
                print_flag=True
            ),
            "otherCatCent_by_WikiSummRepeatHSMix_noRand": lambda: self.initialize_embeds_by_other_category_centroid_by_function_without_random(
                model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, initvec_func=self.make_initvec_by_wiki_summary_and_hidden_state, 
                layer_idx=layer_idx,
                other_type="far", # 学習対象カテゴリの候補カテゴリの中からランダムに選ぶ方式
                mix_layers=True,
                print_flag=True
            )

        }
        print(handlers.keys())

        if init_vec_type in handlers:
            print(f"Initializing embeddings with method: {init_vec_type}")
            return handlers[init_vec_type]()

        if init_vec_type == "other_category_centroid_plus_random":
            raise NotImplementedError("other_category_centroid_plus_random is still WIP")

        # # ============================================================================================
        # if init_vec_type not in handlers:
        #     # ** 指定の語句で初期化 (句の場合は単純にmean poolingする) **
        #     print(f"Initializing embeddings with method: {init_vec_type}")
        #     init_terms = [init_vec_type]     # 'a chair' など
        #     return self.initVecWithTokenVec(
        #         model, tokenizer, init_terms, train_token2tokenid, print_flag=print_flag
        #     )
        raise ValueError(f"Unknown init_vec_type: {init_vec_type}. Available methods: {list(handlers.keys())}")

    
    

        


    # ====================================== debiase用のカテゴリ共通成分を計算する方法の種類 ============================
    def calculateGlobalHSMean_by_GlbPrimDebiasedHSMixed(self, model, tokenizer, n_components, mix_layers):
        # ****** (Type3) global_hidden_meanを、全カテゴリの主成分で計算する (各隠れ層の代表vecは、その前後の層との平均) ******
        print("Calculating global hidden state mean for debiasing by global primary components...")
        if mix_layers:
            save_globalPrimComp_dir = os.path.join(project_root, "data", "dbpedia", f"global_primary_components_mixedlayers_{self.model_name}_{self.pool_hs_type}", "n_components_10") # os.path.join(project_root, "data", "dbpedia", "global_primary_components", f"n_components_{n_components}")
        else:
            save_globalPrimComp_dir = os.path.join(project_root, "data", "dbpedia", f"global_primary_components_singledlayer_{self.model_name}_{self.pool_hs_type}", "n_components_10") # os.path.join(project_root, "data", "dbpedia", "global_primary_components_singledlayer", f"n_components_{n_components}")

        # dbpediaから収集した全ての固有名詞を収集
        # [memo] この処理は_load_prop_nouns()に置き換えた
        # prop_nouns = self._load_prop_nouns(exclude_category=None, per_cat_limit=self.num_propNouns_in_cat_for_globalHSMean)
        prop_nouns = load_prop_nouns(exclude_category=None, per_cat_limit=self.num_propNouns_in_cat_for_globalHSMean)

        E, num_hidden_layers = self._get_model_info(model)

        # *** 既に保存されている主成分があれば読み込む ***
        if n_components <= 10:
            # 最初にn_components=10で計算済みのため、n_componentsが10より少ない場合は10で保存されているファイルから必要なn_components分だけ切り取って使う
            for layer_idx in range(num_hidden_layers+1):
                save_globalPrimComp_path = os.path.join(save_globalPrimComp_dir, f"global_primary_components_layer_{layer_idx}.pt")
                if os.path.exists(save_globalPrimComp_path):
                    global_mean_vec, pcs, explained_ratio, meta = load_pca_components(save_globalPrimComp_path)
                    self.global_primary_vec_by_layer[layer_idx] = pcs[:n_components]

            # if n_components != 10:
            #     # もし指定されたn_componentsが10でない場合は、保存されている10成分のファイルから新たにn_components成分のファイルを作る
            #     for layer_idx in range(num_hidden_layers+1):

            # 全ての層の主成分が既に保存されていれば、計算せずに終了する
            if all(layer_idx in self.global_primary_vec_by_layer for layer_idx in range(num_hidden_layers + 1)):
                print("Global primary components for all layers are already calculated and loaded. Skipping calculation.")
                return
        

        # *** term毎に、語句をモデルに入力した後の、全隠れ層における隠れ状態を平均して保持する ***
        layer_to_hsVecs = {}
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        for term in tqdm(prop_nouns, miniters=1000, desc="Initializing embeddings"):   #all_prop_nouns:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            inputs = tokenizer(
                term, 
                return_tensors="pt", 
                # add_special_tokens=self.last_token_is_eos
            ).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得する

            # ** モデルに入力して、各層の隠れ状態を語句のベクトルとして取得する **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = out.hidden_states  # [layer_num, batch, seq_len, d]
                    


            # =========  *** 各層の隠れ状態を加算する *** =========
            # ***** 単一層でterm_vecを作る場合 *****
            for layer_idx, layer_hs in enumerate(hs): 
                if layer_idx in self.global_primary_vec_by_layer.keys():
                    # 既に生成されたlayerはskipする
                    continue
                if layer_idx not in layer_to_hsVecs.keys():
                    layer_to_hsVecs[layer_idx] = []
                
                term_vec = self._extract_term_vec(
                    inputs=inputs,
                    layer_idx=layer_idx,
                    num_hidden_layers=num_hidden_layers,
                    all_hs=hs,
                    layer_hs=layer_hs,
                    mix_layers=mix_layers
                )
                        
                layer_to_hsVecs[layer_idx].append(term_vec.cpu())  # GPUからCPUに移してリストに追加


        # *** layer毎に主成分を計算する ***
        for layer_idx, hsVecs in layer_to_hsVecs.items():
            if layer_idx in self.global_primary_vec_by_layer.keys():
                # 既に生成されたlayerはskipする
                continue
            save_globalPrimComp_path = os.path.join(save_globalPrimComp_dir, f"global_primary_components_layer_{layer_idx}.pt")
            X = torch.stack(hsVecs)
            # PCAで主成分を計算する
            global_mean_vec, pcs, explained_ratio = compute_pca_components(X, n_components)
            self.global_primary_vec_by_layer[layer_idx] = pcs[:n_components]
            
            # 保存する
            save_pca_components(
                save_path=save_globalPrimComp_path,
                mean_vec=global_mean_vec,
                pcs=pcs,
                explained_ratio=explained_ratio,
                meta={
                    "layer_idx": layer_idx,
                    "pool_hs_type": self.pool_hs_type,
                    "mix_layers": mix_layers,
                    "num_samples": X.shape[0],
                    "hidden_dim": X.shape[1],
                }
            )
            print(global_mean_vec.shape)
            print(pcs.shape)
            print(explained_ratio)

    


    def calculateGlobalHSMean_by_OtherCatPropNouns(self, model, tokenizer, mix_layers=True):
        """ ****** (Type2) global_hidden_meanを、他カテゴリ全部の固有名詞から計算する ******
        Args:
            * model: HuggingFaceのモデルオブジェクト
            * tokenizer: HuggingFaceのトークナイザオブジェクト
            * mix_layers: Trueなら前後3層の隠れ状態を平均してterm_vecとする。Falseなら単一層の隠れ状態をterm_vecとする。
        """
        print("Calculating global hidden state mean for debiasing...")

        # ** 準備
        E, num_hidden_layers = self._get_model_info(model)
        target_norm = E.norm(dim=1).median().item()     # 語彙中央値を目標ノルムとする
        

        # dbpediaから収集した固有名詞をモデルに入力し、全隠れ層における隠れ状態を平均して保持する
        # propNoun_dir = os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")
        for own_category in self.train_target_category_lst:
            print(f"Calculating global hidden state mean using prop nouns in other categories than '{own_category}'...")


            # [memo] この処理は_load_prop_nouns()に置き換えた
            # prop_nouns = self._load_prop_nouns(
            prop_nouns = load_prop_nouns(
                exclude_category=own_category,                              # own_categoryのprop nounはglobal_hidden_meanの計算に使用しない
                per_cat_limit=self.num_propNouns_in_cat_for_globalHSMean    # 全部追加すると多すぎたので、各カテゴリからランダムに20個だけ追加することにする
            )


            # *** term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
            layer_to_hsSumVec = {}
            valid_init_term_count = 0   # ""でない有効なtermの数をカウント
            for term in tqdm(prop_nouns, miniters=100, desc="Initializing embeddings"):   #all_prop_nouns:
                if term.strip() == "":
                    continue
                valid_init_term_count += 1
                inputs = tokenizer(
                    term, 
                    return_tensors="pt", 
                    # add_special_tokens=self.last_token_is_eos     # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう
                ).to(model.device)

                # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                    hs = out.hidden_states  # [layer_num, batch, seq_len, d]
                
                # *** 各層の隠れ状態を加算する ***
                for layer_idx, layer_hs in enumerate(hs): 
                    if layer_idx not in layer_to_hsSumVec.keys():
                        layer_to_hsSumVec[layer_idx] = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成

                    term_vec = self._extract_term_vec(
                        inputs=inputs,
                        layer_idx=layer_idx,
                        num_hidden_layers=num_hidden_layers,
                        all_hs=hs,
                        layer_hs=layer_hs,
                        mix_layers=mix_layers
                    )
                            
                    # ノルムを語彙中央値に合わせる
                    term_vec_norm = term_vec.norm().item()
                    if target_norm > 0:
                        term_vec *= target_norm / term_vec_norm
                    layer_to_hsSumVec[layer_idx] += term_vec

            # 平均する
            for layer_idx, sumvec in layer_to_hsSumVec.items():
                self.category_to_layer_to_otherHSMeanVec[own_category][layer_idx] = sumvec / valid_init_term_count
            
        print("Calculation of global hidden state mean is Finished.\n")






    def calculateGlobalHiddenStateMean(self, model, tokenizer):
        # ****** initVecWithMeanVecOfDebiasedTermHiddenStates 用に、hidden_stateの平均vecとしてglobal_hidden_meanを計算する ******
        print("Calculating global hidden state mean for debiasing...")

        # dbpediaから収集した全ての固有名詞をモデルに入力し、全隠れ層における隠れ状態を平均して保持する
        # [memo] この処理は_load_prop_nouns()に置き換えた
        # prop_nouns = self._load_prop_nouns(
        prop_nouns = load_prop_nouns(
            exclude_category=None, 
            per_cat_limit=self.num_propNouns_in_cat_for_globalHSMean    # 全部追加すると多すぎたので、各カテゴリからランダムに20個だけ追加することにする
        )

        E = self._get_model_info(model)[0]

        # ノルムを語彙中央値
        target_norm = E.norm(dim=1).median().item() 

        # *** term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        layer_to_hsSumVec = {}
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        for term in tqdm(prop_nouns, miniters=100, desc="Initializing embeddings"):   #all_prop_nouns:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            # inputs = tokenizer(term, return_tensors="pt").to(model.device)
            inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = out.hidden_states  # [layer_num, batch, seq_len, d]
            
            # *** 各層の隠れ状態を加算する ***
            for layer_idx, layer_hs in enumerate(hs): 
                if layer_idx not in layer_to_hsSumVec.keys():
                    layer_to_hsSumVec[layer_idx] = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
                
                if self.pool_hs_type == "single_last":
                    # ** term中の最後のtokenのみでterm_vecを作る場合:
                    last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                    term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
                elif self.pool_hs_type == "mean":
                    # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                    seq_len = inputs["attention_mask"].sum().item()
                    term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [d]


                # ノルムを語彙中央値に合わせる
                term_vec_norm = term_vec.norm().item()
                if target_norm > 0:
                    term_vec *= target_norm / term_vec_norm
                layer_to_hsSumVec[layer_idx] += term_vec


        # 平均する
        for layer_idx, sumvec in layer_to_hsSumVec.items():
            self.layer_to_globalHSMeanVec[layer_idx] = sumvec / valid_init_term_count
        
        print("Calculation of global hidden state mean is Finished.\n")










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

    
    def initvec_by_other_category_centroid_plus_random(self, model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, print_flag=False):
        # *** (Type1) カテゴリ内の90(propnoun_num_for_init_vec-10)固有名詞(固定)を平均化したvec + カテゴリ内の10固有名詞をランダムに選んで平均化したvec を足し合わせたvecで初期化 ***
        # Type1: 各prop noun内のtokenベクトルを平均したvecをその固有名詞ベクトルとする方法。
        # category_COGではカテゴリ内の初期化vec同士に差がなく、性能が上がらなかったため、
        # 中心vecはカテゴリ内で共通させつつ、そこにカテゴリ内の固有名詞の中からランダムに選んだ10個のvecの平均を足し合わせることで、カテゴリ内の初期化vec同士に差をつけてみる方法

        for category, init_token_ids in category2initoken_ids.items():
            # *** このカテゴリに対応する初期化vec作成用の固有名詞リストで初期化vecを作成し、
            # このカテゴリに属す固有名詞(新規概念用)に割り当てたtokenのtoken idsの行を、その初期化vecで初期化する ***
            init_terms_candidate = category_to_concepts_for_vec[category]
            init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10))  # カテゴリ内の固有名詞からランダムに90個選んで中心vecを作成

            # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
            init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

            # 初期化対象の追加token毎に、10件をランダム選出してmodel
            for init_token_id in init_token_ids:
                init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) # カテゴリ内の固有名詞からランダムに10個選んでランダムvecを作成
                init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用
                    
                # 埋め込み層のinit_token_idsに該当する行を、init_termsのtokenベクトルの平均で初期化する
                model = self.initVecWithTokenVec(model, tokenizer, init_terms, [init_token_id], print_flag=print_flag)
                print(f"Initialized category '{category}' (new token {tokenizer.decode(init_token_id)}, token_id: {init_token_id}) with {len(init_terms)} concepts: ... ({init_terms[-15:]}).")
        return model
    

    # ***
    def initvec_by_category_COG_by_simple_mean(self, model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, print_flag=False):
        # COG: Center Of Gravity. 
        # *** 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
        # vec_propnoun = mean(token_vecs_in_propnoun) -> 各カテゴリの初期化vec = mean(vec_propnoun_in_category)
        for category, init_token_ids in category2initoken_ids.items():
            # *** このカテゴリに対応する初期化vec作成用の固有名詞リストで初期化vecを作成し、
            # このカテゴリに属す固有名詞(新規概念用)に割り当てたtokenのtoken idsの行を、その初期化vecで初期化する ***
            init_terms = category_to_concepts_for_vec[category]
            # 埋め込み層のinit_token_idsに該当する行を、init_termsのtokenベクトルの平均で初期化する
            model = self.initVecWithTokenVec_with_noise(model, tokenizer, init_terms, init_token_ids, print_flag=print_flag)
            print(f"Initialized category '{category}' ({len(init_token_ids)} new tokens) with {len(init_terms)} concepts ({init_terms[:5]}...) for token {[tokenizer.decode(tid) for tid in init_token_ids[:5]]}... .")

        return model

    
    def initvec_by_other_category_COG_by_simple_mean(self, model, tokenizer, category_to_concepts_for_vec, category2initoken_ids, print_flag=False):
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
                model = self.initVecWithTokenVec_with_noise(model, tokenizer, init_terms, [init_token_id], print_flag=print_flag)
                # if print_flag:
                print(f"Initialized category '{category}' (new token {init_token_id}) with {len(init_terms)} concepts ({init_terms[:5]}...) from other category '{other_category}' for token {[tokenizer.decode(tid) for tid in init_token_ids[:5]]}... .")
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
        print_flag=False
        ):
        # *** 初期vec作成用の固有名詞リストをカテゴリ毎に用意し、任意の関数で初期vecを作成する方法. ***

        for own_category, init_token_ids in category2initoken_ids.items():
            # *** このカテゴリに対応する初期化vec作成用の固有名詞リストで初期化vecを作成し、
            # このカテゴリに属す固有名詞(新規概念用)に割り当てたtokenのtoken idsの行を、その初期化vecで初期化する ***
            init_terms_candidate = category_to_concepts_for_vec[own_category]

            if len(init_terms_candidate) < self.propnoun_num_for_init_vec - 10  +  10 * len(init_token_ids):
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
                    own_category, 
                    init_terms,
                    layer_idx=layer_idx, 
                    lambda_=LAMBDA_, 
                    mix_layers=mix_layers,
                    print_flag=print_flag
                )

                # 埋め込み層のinit_token_ids (<unusedx>) に該当する行を、まとめてinit_srcで初期化.
                E = self._get_model_info(model)[0]
                with torch.no_grad():
                    init_target_ids = torch.as_tensor([init_token_id], device=E.device, dtype=torch.long)
                    src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
                    E.index_copy_(dim=0, index=init_target_ids, source=src)
                
                print(f"Initialized category '{own_category}' (new token {tokenizer.decode(init_token_id)}, token_id: {init_token_id}) with {len(init_terms)} concepts: ... ({init_terms[-15:]}).")
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
                        own_category, 
                        init_terms,
                        layer_idx=layer_idx, 
                        lambda_=LAMBDA_, 
                        mix_layers=mix_layers, 
                        print_flag=print_flag
                    )

                    if debug_initvec_flag:
                        print(f"[INIT] own_category={own_category}, other_category={other_category}, token_id={init_token_id}")
                        print(f"[INIT] init_terms size={len(init_terms)}")
                        print(f"[INIT] init_src shape={tuple(init_src.shape)}")
                        print(f"[INIT] init_src finite={torch.isfinite(init_src).all().item()}")
                        print(f"[INIT] init_src has_nan={torch.isnan(init_src).any().item()}, has_inf={torch.isinf(init_src).any().item()}")
                        print(f"[INIT] init_src min={init_src.min().item()}, max={init_src.max().item()}, mean={init_src.mean().item()}, norm={init_src.norm().item()}")

                        

                    # 埋め込み層のinit_token_ids (<unusedx>) に該当する行を、まとめてinit_srcで初期化.
                    E = self._get_model_info(model)[0]
                    with torch.no_grad():
                        init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
                        src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
                        E.index_copy_(dim=0, index=init_target_ids, source=src)
                    
                print(f"===\n\tNew token {tokenizer.decode(init_token_id)} in category '{own_category}' is initialized with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
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
                own_category, 
                init_terms_for_centroid,
                layer_idx=layer_idx, 
                lambda_=LAMBDA_, 
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
                    own_category, 
                    init_terms_for_centroid,
                    layer_idx=layer_idx, 
                    lambda_=LAMBDA_, 
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
        own_category,   # global vecを使わないので不要 (どのカテゴリのglobal vecを使うかの選択がないため)
        init_terms,
        layer_idx=None, 
        lambda_=None,   # global vecを使わないので不要
        mix_layers=False,
        print_flag=False
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

            if self.repeat_prompt:
                # *** 初期vecを、固有名詞毎のwikiのsummary文を2回入力して、2回目の文内token位置の隠れ状態から作る場合: https://openreview.net/forum?id=Ahlrf2HGJR の手法 ***
                summary = repeat_text(summary, 2)
            prompt_lst.append(summary)
        
        if len(prompt_lst) == 0:
            raise ValueError(f"No valid summaries found for the given terms. Cannot create initialization vector.")
        print(f"Created prompt list for terms: {prompt_lst[:5]}... (total {len(prompt_lst)})")


        # 2. 各語句ベクトルを作成する. wiki summary をモデルに入力し、pool_hs_type に応じてsummary中の全token/最終token/eos位置の隠れ状態をその語句のベクトルとする
        term_vecs = self._extract_hidden_states(
            model, 
            tokenizer, 
            prompt_lst, 
            batch_size=BATCH_SIZE, 
            layer_index=layer_idx, 
            mix_layers=True, 
            print_flag=False
        )
        

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



    # initVecWithGlobalVecDebiasedTermHS, initVecWithMeanVecOfDebiasedTermHiddenStates
    def make_initvec_by_terms_with_debiased_hidden_state_by_global_vec(
        self,
        model, 
        tokenizer, 
        own_category, 
        init_terms, 
        # init_target_tokenids, 
        layer_idx=None, 
        lambda_=None, 
        mix_layers=False, 
        print_flag=False
        ):
        """語句をモデルに入力し、語句中の最終tokenを入れた後の、モデル内の指定層における、中心化した隠れ状態をその語句のベクトルとし、
            埋め込み層の特定の行を、指定した語句の集合のベクトルの平均で初期化する関数.
            2026/03/21-3

            * 等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
            * 近傍層に特徴が分散している可能性があるので、前後3層の隠れ状態を平均する
        方法: 
            * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
                * global_hidden_mean は 全カテゴリで、PCAによる主成分で構成したベクトルを使用する
            * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。(type2では、最後のsub-tokenの隠れ状態のみ)
            Args:
            - model: HuggingFaceのモデルオブジェクト
            - tokenizer: HuggingFaceのトークナイザオブジェクト
            - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
            # - init_target_tokenids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
            - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
        - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """
        E, num_hidden_layers = self._get_model_info(model)
        if print_flag:
            print(f"⭐️num_hidden_layers: {num_hidden_layers}")

        # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成

        for term in init_terms:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            inputs = tokenizer(
                term, 
                return_tensors="pt", 
                # add_special_tokens=self.last_token_is_eos     # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう
            ).to(model.device)

            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = out.hidden_states  # tuple of (batch_size, seq_len, hidden_size)のリスト. 長さはnum_hidden_layers+1 (embedding層の出力も含むため)
                layer_hs = hs[layer_idx]

            term_vec = self._extract_term_vec(
                inputs=inputs,
                layer_idx=layer_idx,
                num_hidden_layers=num_hidden_layers,
                all_hs=hs,
                layer_hs=layer_hs,
                mix_layers=mix_layers
            )
        
            sum_vec += term_vec

        # 2. term間の平均vecを計算
        if sum_vec.norm().item() == 0.0 or valid_init_term_count == 0:
            raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
        own_centroid = sum_vec / valid_init_term_count


        # 3. 中心化: 隠れ層の平均vecを引き、カテゴリ代表vec間の方向の差を目立たせる
        if self.init_vec_type in ['CatCent_by_GlbPrimDebiasedHSMixed', 'otherCatCent_by_GlbPrimDebiasedHSMixed',
                            'CatCent_by_GlbPrimDebiasedHS', 'otherCatCent_by_GlbPrimDebiasedHS']:
            pcs = self.global_primary_vec_by_layer.get(layer_idx).to(
                device=own_centroid.device,
                dtype=own_centroid.dtype
            )
            proj = (own_centroid @ pcs.T) @ pcs                  # [d]
            global_vec = proj
        
        elif self.init_vec_type in ['CatCentroid_by_OthCatDebiasedHSMixed', 'otherCatCentroid_by_OthCatDebiasedHSMixed']:
            global_vec = self.category_to_layer_to_otherHSMeanVec[own_category][layer_idx]
        elif self.init_vec_type in ['categoryCentroid_by_DebiasedHSMixed', 'otherCategoryCentroid_by_DebiasedHSMixed'
                                    'categoryCentroid_by_DebiasedHiddenState', 'otherCategoryCentroid_by_DebiasedHiddenState']:
            global_vec = self.layer_to_globalHSMeanVec[layer_idx]



        global_vec = global_vec.to(device=own_centroid.device, dtype=own_centroid.dtype)

        init_src = own_centroid - lambda_ * global_vec
        # [memp これはだめな手法] own_centroid のノルムに対してlambda_倍した大きさのノルムの global_vec を引く. global_vec は、全カテゴリで、PCAによる主成分で構成したベクトル
        # init_src = own_centroid - (global_vec / global_vec.norm().clamp_min(1e-12) * own_centroid.norm().clamp_min(1e-12)) * lambda_   # global_vecをown_centroidのノルムに合わせてスケーリングしてから引く

        # 4. 微小ノイズを加える
        d = init_src.shape[0]

        # 微小ノイズを作る
        noise = torch.randn(d, device=E.device, dtype=E.dtype)

        # 各行をL2正規化して「方向だけランダム」にする
        eps = 1e-12
        noise = noise / noise.norm(p=2, dim=0, keepdim=True).clamp_min(eps)

        # ノイズの大きさを、重心ノルムのごく一部にする
        noise_scale = NOISE_SCALE   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
        init_norm = init_src.norm(p=2).clamp_min(eps)
        noise = noise * (init_norm * noise_scale)

        # 重心 + 微小ノイズ
        init_src = init_src + noise

        # 4. ノルムを語彙中央値に合わせる [memo] hidde stateのノルムは埋め込み層のノルムと大きく異なる可能性があるため、ノルムを合わせる
        target_norm = E.norm(dim=1).median().item()  # 埋め込み行のノルムの中央値をターゲットノルムとする
        init_src_norm = init_src.norm().item()
        if init_src_norm > 0:
            init_src = init_src / init_src_norm * target_norm  # ターゲットノルムに合わせてスケーリング
        
        return init_src



    # initVecWithMeanVecOfTermHiddenStates
    def make_initvec_by_terms_with_hidden_state(
        self,
        model, 
        tokenizer, 
        own_category, # 使わないが、initvec_funcで統一して関数を呼ぶために引数として受け取る
        init_terms,
        layer_idx=None, 
        lambda_=None, 
        mix_layers=False,
        print_flag=False
        ):
        """語句をモデルに入力し、語句中の最終tokenを入れた後の、モデル内の指定層における隠れ状態をその語句のベクトルとし、
        埋め込み層の特定の行を、指定した語句の集合のベクトルの平均で初期化する関数.
        Args:
        - model: HuggingFaceのモデルオブジェクト
        - tokenizer: HuggingFaceのトークナイザオブジェクト
        - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
        - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
        - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """     
        E, num_hidden_layers = self._get_model_info(model)
        if print_flag:
            print(f"⭐️num_hidden_layers: {num_hidden_layers}")

        # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
        for term in init_terms:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            inputs = tokenizer(
                term, 
                return_tensors="pt", 
                # add_special_tokens=self.last_token_is_eos     # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう
            ).to(model.device)
            
            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = out.hidden_states  # tuple of (batch_size, seq_len, hidden_size)のリスト. 長さはnum_hidden_layers+1 (embedding層の出力も含むため)
                layer_hs = hs[layer_idx]


            term_vec = self._extract_term_vec(
                inputs=inputs,
                layer_idx=layer_idx,
                num_hidden_layers=num_hidden_layers,
                all_hs=hs,
                layer_hs=layer_hs,
                mix_layers=mix_layers
            )
            sum_vec += term_vec


        # 2. term間の平均vecを計算
        if sum_vec.norm().item() == 0.0 or valid_init_term_count == 0:
            raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
        init_src = sum_vec / valid_init_term_count


        # 3. 微小ノイズを加える
        # n = len(init_target_ids)
        d = init_src.shape[0]

        # 微小ノイズを作る
        noise = torch.randn(d, device=E.device, dtype=E.dtype)

        # 各行をL2正規化して「方向だけランダム」にする
        eps = 1e-12
        noise = noise / noise.norm(p=2, dim=0, keepdim=True).clamp_min(eps)

        # ノイズの大きさを、重心ノルムのごく一部にする
        noise_scale = 2e-3   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
        init_norm = init_src.norm(p=2).clamp_min(eps)
        noise = noise * (init_norm * noise_scale)

        # 重心 + 微小ノイズ
        init_src = init_src + noise


        # 4. ノルムを語彙平均に合わせる [memo] hidde stateのノルムは埋め込み層のノルムと大きく異なる可能性があるため、ノルムを合わせる
        target_norm = E.norm(dim=1).median().item()  # 埋め込み行のノルムの中央値をターゲットノルムとする
        init_src_norm = init_src.norm().item()
        if init_src_norm > 0:
            init_src = init_src / init_src_norm * target_norm  # ターゲットノルムに合わせてスケーリング

        return init_src








    # ========================== その他の細かい処理 ==========================

    def _get_model_info(self, model):
        try:
            E = model.model.embed_tokens.weight
            num_hidden_layers = model.config.num_hidden_layers
        except AttributeError:
            E = model.model.language_model.embed_tokens.weight
            num_hidden_layers = model.config.text_config.num_hidden_layers
        return E, num_hidden_layers

    # [memo] handle_data_from_dbpedia_utils.pyに移動
    # def _load_prop_nouns(self, exclude_category=None, per_cat_limit=None):
    #     """dbpediaから収集した全ての固有名詞を収集
    #     """
    #     propNoun_dir = os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")

    #     prop_nouns = []
    #     for category_file in os.listdir(propNoun_dir):
    #         if not category_file.endswith(".csv"):
    #             continue
    #         # もしexclude_categoryが指定されていれば、そのカテゴリの固有名詞は読み込まない
    #         category = category_file.removesuffix(".csv").replace("_", " ")
    #         if exclude_category == category:
    #             continue

    #         df = pd.read_csv(os.path.join(propNoun_dir, category_file))
    #         labels = df["label"].dropna().tolist()
    #         # 全部追加すると多すぎたので、各カテゴリからランダムに指定数(100や20など)個だけ追加することにする
    #         k = min(per_cat_limit or len(labels), len(labels))
    #         prop_nouns.extend(random.sample(labels, k))
    #     return prop_nouns


    def _get_mix_layers(self, layer_idx, num_hidden_layers):
        if layer_idx == -1 or layer_idx == num_hidden_layers:
            mixed_layers = [-1, -2, -3]                           # 最終層とその前の2層を平均する
        elif layer_idx == 0:
            mixed_layers = [0, 1, 2]                              # 最初の層とその後の2層を平均する
        else:
            mixed_layers = [layer_idx-1, layer_idx, layer_idx+1]  # 指定層の前後3層を平均する
        # else:
        #     raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be -1, 0, or a positive integer less than num_hidden_layers.")
        return mixed_layers


    # ** [memo] utilsのwiki utilsに移動
    # def _load_wikisummary(self, propnoun):
    #     """dbpediaから収集した固有名詞のwikipedia summaryを読み込んで、propnoun_to_wikisummaryに保存する。
    #     data/wiki_pages に未保存であれば、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに保存する
    #     """
    #     # print("Loading Wikipedia summaries for prop nouns...")
    #     wiki_pages_dir = os.path.join(project_root, "data", "wiki_pages")

    #     filename = self._change_propnoun_to_filename(propnoun) + ".json"  # ファイル名に使用できない文字を置換
    #     wikipage_path = os.path.join(wiki_pages_dir, filename)
        
    #     # * 未取得の場合、wikipedia apiから取得して保存する
    #     if not os.path.exists(wikipage_path):
    #         wiki_info = fetch_wikipedia_page(propnoun, lang="en")
    #         if wiki_info["exists"] == False:
    #             print(f"Wikipedia page for concept '{propnoun}' DOES NOT exist. Skipping generation.")
    #             return None
    #         # 本文を切り出す
    #         main_text = extract_wiki_main_text(wiki_info['text'])
    #         wiki_info['text'] = main_text

    #         # 保存
    #         with open(wikipage_path, "w") as f:
    #             json.dump(wiki_info, f, ensure_ascii=False, indent=4)
    
    #     # * 今ここで保存した or すでに保存されているwikipedia summaryを読み込む
    #     with open(wikipage_path, "r") as f:
    #         wiki_page = json.load(f)
    #         summary = wiki_page.get("summary")
    #         if summary:
    #             self.propnoun_to_wikisummary[propnoun] = summary
    #             # print(f"Loaded Wikipedia summary for '{propnoun}' from wiki_pages.")
    #             return summary
    #         else:
    #             print(f"No summary found in wiki page for '{propnoun}' in wiki_pages.")
    #             return None
            
    # ** [memo] utilsの handle text utilsに移動
    # def _change_propnoun_to_filename(self, propnoun):
    #     """固有名詞を、ファイル名に使用できない文字を置換して、ファイル名に変換する関数。
    #     例: "New York" -> "New_York"
    #     例: "A/B" -> "A_B"
    #     """
    #     filename = re.sub(r'[/\\ ]', '_', propnoun)  # ファイル名に使用できない文字を置換
    #     return filename


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
            mix_layers = self._get_mix_layers(layer_idx, num_hidden_layers)
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



    @torch.no_grad()
    def _extract_hidden_states(self, model, tokenizer, text_list, batch_size=8, layer_index=-1, mix_layers=True, print_flag=False):
        """
        各テキストの末尾にEOSを明示的に追加し、
        EOSトークン位置の hidden state を返す。

        Returns:
            np.ndarray of shape (N, hidden_dim)
        """

        E, num_hidden_layers = self._get_model_info(model)


        all_vecs = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]
            if print_flag:
                print(f"Processing batch {i // batch_size + 1}/{(len(text_list) + batch_size - 1) // batch_size} for hidden state extraction...")

            if self.pool_hs_type == "eos":
                # EOS を明示的に末尾へ追加
                batch_texts = [text + tokenizer.eos_token for text in batch_texts]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False #last_token_is_eos#LAST_TOKEN_IS_EOS, -> pool_hs_type == "eos"の場合は明示的にeosを追加済みなので、ここをTrueにするとeosが重複して2つ付く可能性がある。そのためここはFalseで良い。
            ).to(model.device) 

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # hidden_states は tuple:
                # 0: embedding出力, 1..L: 各層出力
                all_hs = outputs.hidden_states

            if mix_layers:
                # ** 前後3層mixでterm_vecを作る場合 **
                target_layer_hs = torch.stack(
                    [all_hs[lid] for lid in self._get_mix_layers(layer_index, num_hidden_layers)],
                    dim=0
                )  # 指定層の出力 [3, batch_size, seq_len, d]
            else:
                # ** 単一層でterm_vecを作る場合 **
                target_layer_hs = all_hs[layer_index].unsqueeze(0)      # (1, batch_size, seq_len, d)

            
            
            # *** pool_hs_type に応じて、vectorを抽出する位置を決定 ***
            if self.pool_hs_type == "eos":
                # 各系列について EOS token の最後の出現位置を取る
                eos_mask = (input_ids == tokenizer.eos_token_id)

            for t_idx in range(input_ids.size(0)):

                # ** 1 が立っている位置を取得 **
                # e.g.  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] -> valid_pos = [6, 7, 8, 9, 10, 11] 
                valid_pos = torch.nonzero(attention_mask[t_idx], as_tuple=False).squeeze(-1)
                # print(f"valid_pos for batch {t_idx}: {valid_pos}")
                if valid_pos.numel() == 0:
                    # 全部 padding の場合
                    pos_begin = 0
                    pos_end = 0
                else:
                    pos_begin = valid_pos[0].item()
                    pos_end = valid_pos[-1].item() + 1   # slice用に end は exclusive

                    
                if self.pool_hs_type == "eos":
                    eos_positions = torch.where(eos_mask[t_idx])[0]
                    if len(eos_positions) == 0:
                        raise ValueError(f"EOS token が見つかりません: {batch_texts[t_idx]}")
                    eos_pos = eos_positions[-1].item()
                    pos_begin = eos_pos
                    pos_end = eos_pos + 1

                elif self.pool_hs_type == "last_token":
                    pos_begin = pos_end - 1

                elif self.pool_hs_type == "mean_pool":
                    if self.repeat_prompt:
                        # *** 初期vecを、固有名詞毎のwikiのsummary文を2回入力して、2回目の文内token位置の隠れ状態から作る場合: https://openreview.net/forum?id=Ahlrf2HGJR の手法 ***
                        # wiki summaryを繰り返してプロンプトとする場合は、2回目の文のみの隠れ状態を平均する
                        pos_begin_second_sent = (pos_begin + pos_end) // 2 # == pos_begin + (pos_end - pos_begin) / 2
                        pos_begin = pos_begin_second_sent
                    else:
                        pass # デフォルトは、入力文全体の隠れ状態の平均を取る
    
                else:
                    raise ValueError(f"Unknown pool_hs_type: {self.pool_hs_type}")

                # ** vectorを抽出 **
                # vec = layer_hs[t_idx, pos_begin:pos_end, :].mean(dim=0)  # (H,)
                term_vec = target_layer_hs[:, t_idx, pos_begin:pos_end, :].mean(dim=0).mean(dim=0)  # (mix層数, batch_size, seq_len, d) -> batch内のt_idxに該当する層&平均対象のtoken位置を指定: (mix層数, meanpool対象token数, d) -> 前後3層を平均した隠れ状態のうち、valid_tokenの部分を平均する: (meanpool対象token数, d) -> meanpool対象tokenを平均する: (d)
                all_vecs.append(term_vec.detach().cpu())

                if print_flag:
                    # どの位置のtokenの隠れ状態が使われるのかを確認するためのprint文
                    print(f"target_layer_hs shape: {target_layer_hs.shape} -> term_vec: {term_vec.shape}")  # (mix層数, batch_size, seq_len, d)
                    print(f"pos_begin: {pos_begin}, pos_end: {pos_end}")
                    print(f"\tattention_mask: {attention_mask[t_idx]},\n\t valid_pos: {valid_pos}, \n\t valid part in batch_text: {input_ids[t_idx][pos_begin:pos_end]}")

        return torch.stack(all_vecs, dim=0) # np.stack(all_vecs, axis=0)

        

