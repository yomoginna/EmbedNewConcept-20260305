
# ===== Standard library =====
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

N_COMPONENTS = 2
NOISE_SCALE = 5e-3   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
LAMBDA_ = 0.0   # global_vecを引くときの重み. 0.1あたりから試す. 0.1だと少し改善するが、0.2だとさらに改善する。 0.3はまだ試していないが後で試す

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
              例: {"layer_idx": 10, "term_vec_type": "mean", "mix_layers": True}
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
    def __init__(self, model_name, init_vec_type, train_target_category_lst, propnoun_num_for_init_vec, model, tokenizer, seed, term_vec_type):

        self.model_name = model_name.split("/")[-1]  # "gemma-3-12b-it"のようなモデル名だけを取り出す
        self.init_vec_type = init_vec_type
        self.train_target_category_lst = train_target_category_lst
        self.propnoun_num_for_init_vec = propnoun_num_for_init_vec
        self.term_vec_type = term_vec_type    # single_last or mean,  inputsに対する隠れ状態を、term中の全subtokenに対して平均するか、term中の最後のtokenに対応する隠れ状態を使うか
        self.layer_to_globalHSMeanVec = {}
        self.category_to_layer_to_otherHSMeanVec = defaultdict(dict)  # category_to_layer_to_otherHSMeanVec[category][layer_idx] = other_hidden_mean_vec for that category and layer
        # self.global_primary_vec_by_mixed_layer = {} # layer代表vecはその前後との平均の主成分
        self.global_primary_vec_by_layer = {}       # layer代表vecはその単層の主成分
        self.num_propNouns_in_cat_for_globalHSMean = 100

        random.seed(seed)
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
        
        # # * debiase なし
        # elif init_vec_type in [
        #     'CatCent_by_HS', 'otherCatCent_by_HS', # [未実行]
        #     'CatCent_by_HSMixed', 'otherCatCent_by_HSMixed']: # [未実行]
        #     pass    # debiasなしの初期化方法は、global_hidden_meanを計算する必要がないため、ここでは何もしない

        # ** debias なし
        # * wikiのsummary入力時の隠れ状態で初期化vecを作成する方法. 固有名詞を構成する単語がカテゴリ特有の単語ではないことから、カテゴリ代表vec作成には固有名詞ではなく文脈から生まれる意味が必要なのではないか、という発想から。
        elif init_vec_type in [
            # wikiのsummary入力時の隠れ状態で初期化vecを作成する方法は、global_hidden_meanを計算する必要がないため、ここでは何もしない
            'CatCent_by_wikiSummary_HSMixed', 'otherCatCent_by_wikiSummary_HSMixed',
            # WIP memo
            'category_centroid_by_hidden_state_mean', 'other_category_centroid_by_hidden_state_mean',
            # WIP memo
            'category_COG', 'other_category_COG',
            ]:
            pass
        else:
            raise ValueError(f"Unknown init_vec_type: {init_vec_type}")




    # ====================================== debiase用のカテゴリ共通成分を計算する方法の種類 ============================
    def calculateGlobalHSMean_by_GlbPrimDebiasedHSMixed(self, model, tokenizer, n_components, mix_layers):
        # ****** (Type3) global_hidden_meanを、全カテゴリの主成分で計算する (各隠れ層の代表vecは、その前後の層との平均) ******
        print("Calculating global hidden state mean for debiasing by global primary components...")
        if mix_layers:
            save_globalPrimComp_dir = os.path.join(project_root, "data", "dbpedia", f"global_primary_components_mixedlayers_{self.model_name}_{self.term_vec_type}", "n_components_10") # os.path.join(project_root, "data", "dbpedia", "global_primary_components", f"n_components_{n_components}")
        else:
            save_globalPrimComp_dir = os.path.join(project_root, "data", "dbpedia", f"global_primary_components_singledlayer_{self.model_name}_{self.term_vec_type}", "n_components_10") # os.path.join(project_root, "data", "dbpedia", "global_primary_components_singledlayer", f"n_components_{n_components}")

        # dbpediaから収集した全ての固有名詞を収集
        propNoun_dir = os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")
        propNouns = []  # all_propNouns = []
        for category_file in os.listdir(propNoun_dir):
            if category_file.endswith(".csv"):
                df = pd.read_csv(os.path.join(propNoun_dir, category_file))
                # 全部追加すると多すぎたので、各カテゴリからランダムに100個だけ追加することにする
                propNouns += random.sample(df['label'].tolist(), min(self.num_propNouns_in_cat_for_globalHSMean, len(df)))

        try:
            E = model.model.embed_tokens.weight
            num_hidden_layers = model.config.num_hidden_layers
        except:
            E = model.model.language_model.embed_tokens.weight
            num_hidden_layers = model.config.text_config.num_hidden_layers

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
        for term in tqdm(propNouns, miniters=1000, desc="Initializing embeddings"):   #all_propNouns:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

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

                # *** 前後3層mixでterm_vecを作る場合 ***
                if mix_layers:
                    if layer_idx == -1 or layer_idx == num_hidden_layers:
                        mixd_layer = [-1, -2, -3]  # 最終層とその前の2層を平均する
                    elif layer_idx == 0:
                        mixd_layer = [0, 1, 2]  # 最初の層とその後の2層を平均する
                    else:
                        mixd_layer = [layer_idx-1, layer_idx, layer_idx+1]  # 指定層の前後3層を平均する

                    # *** 前後3層の隠れ状態を平均する: ***
                    layer_hs_mix = torch.stack(
                        [hs[lid] for lid in mixd_layer],
                        dim=0
                    )  # 指定層の出力 [3, 1, seq_len, d]

                    if self.term_vec_type == "single_last":
                        # ** term中の最後のtokenのみでterm_vecを作る場合:
                        last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                        term_vec = layer_hs_mix[:, 0, last_token_idx, :].mean(dim=0)   # [1, seq_len, d] -> [d] 前後3層の最後のtokenの隠れ状態を平均する.
                    elif self.term_vec_type == "mean":
                        # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                        seq_len = inputs["attention_mask"].sum().item()
                        term_vec = layer_hs_mix[:, 0, :seq_len, :].mean(dim=1).mean(dim=0)    # [3, 1, seq_len, d] -> [seq_len, d] -> [d]

                # ***** 単一層でterm_vecを作る場合 *****
                else:
                    if self.term_vec_type == "single_last":
                        # ** term中の最後のtokenのみでterm_vecを作る場合:
                        last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                        term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
                    elif self.term_vec_type == "mean":
                        # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                        seq_len = inputs["attention_mask"].sum().item()
                        term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [d]
                        
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
                    "term_vec_type": self.term_vec_type,
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
        propNoun_dir = os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")
        try:
            E = model.model.embed_tokens.weight
            num_hidden_layers = model.config.num_hidden_layers
        except:
            E = model.model.language_model.embed_tokens.weight
            num_hidden_layers = model.config.text_config.num_hidden_layers
        target_norm = E.norm(dim=1).median().item()     # 語彙中央値を目標ノルムとする
        

        # dbpediaから収集した固有名詞をモデルに入力し、全隠れ層における隠れ状態を平均して保持する
        for own_category in self.train_target_category_lst:
            print(f"Calculating global hidden state mean using prop nouns in other categories than '{own_category}'...")
            propNouns = []  # all_propNouns = []
            for category_file in os.listdir(propNoun_dir):
                if category_file.endswith(".csv"):
                    if own_category == category_file.split(".csv")[0].replace("_", " "):  # own_categoryのprop nounはglobal_hidden_meanの計算に使用しない
                        print(f"⭐️⭐️⭐️ Skipping prop nouns in own category '{own_category}' for global hidden state mean calculation.")
                        continue
                    df = pd.read_csv(os.path.join(propNoun_dir, category_file))
                    # 全部追加すると多すぎたので、各カテゴリからランダムに20個だけ追加することにする
                    propNouns += random.sample(df['label'].tolist(), min(self.num_propNouns_in_cat_for_globalHSMean, len(df)))


            # *** term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
            layer_to_hsSumVec = {}
            valid_init_term_count = 0   # ""でない有効なtermの数をカウント
            for term in tqdm(propNouns, miniters=100, desc="Initializing embeddings"):   #all_propNouns:
                if term.strip() == "":
                    continue
                valid_init_term_count += 1
                inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

                # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                    hs = out.hidden_states  # [layer_num, batch, seq_len, d]
                
                # *** 各層の隠れ状態を加算する ***
                for layer_idx, layer_hs in enumerate(hs): 
                    if layer_idx not in layer_to_hsSumVec.keys():
                        layer_to_hsSumVec[layer_idx] = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成

                    # ***** 前後3層mixでterm_vecを作る場合 *****
                    if mix_layers:
                        if layer_idx == -1 or layer_idx == num_hidden_layers:
                            mix_layers = [-1, -2, -3]  # 最終層とその前の2層を平均する
                        elif layer_idx == 0:
                            mix_layers = [0, 1, 2]  # 最初の層とその後の2層を平均する
                        else:
                            mix_layers = [layer_idx-1, layer_idx, layer_idx+1]  # 指定層の前後3層を平均する

                        layer_hs_mix = torch.stack(
                            [out.hidden_states[lid] for lid in mix_layers],
                            dim=0
                        )  # 指定層の出力 [3, 1, seq_len, d]

                        # *** 前後3層の隠れ状態を平均する: ***
                        if self.term_vec_type == "single_last":
                            # ** term中の最後のtokenのみでterm_vecを作る場合:
                            last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                            term_vec = layer_hs_mix[:, 0, last_token_idx, :].mean(dim=0)   # [1, seq_len, d] -> [d] 前後3層の最後のtokenの隠れ状態を平均する.
                        elif self.term_vec_type == "mean":
                            # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                            seq_len = inputs["attention_mask"].sum().item()
                            term_vec = layer_hs_mix[:, 0, :seq_len, :].mean(dim=1).mean(dim=0)    # [3, 1, seq_len, d] -> [seq_len, d] -> [d]


                    # ***** 単層でterm_vecを作る場合: *****
                    else:
                        if self.term_vec_type == "single_last":
                            # ** term中の最後のtokenのみでterm_vecを作る場合:
                            last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                            term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
                        elif self.term_vec_type == "mean":
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
                self.category_to_layer_to_otherHSMeanVec[own_category][layer_idx] = sumvec / valid_init_term_count
            
        print("Calculation of global hidden state mean is Finished.\n")





    def calculateGlobalHiddenStateMean(self, model, tokenizer):
        # ****** initVecWithMeanVecOfDebiasedTermHiddenStates 用に、hidden_stateの平均vecとしてglobal_hidden_meanを計算する ******
        print("Calculating global hidden state mean for debiasing...")
        # dbpediaから収集した全ての固有名詞をモデルに入力し、全隠れ層における隠れ状態を平均して保持する
        propNoun_dir = os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")
        propNouns = []  # all_propNouns = []
        for category_file in os.listdir(propNoun_dir):
            if category_file.endswith(".csv"):
                df = pd.read_csv(os.path.join(propNoun_dir, category_file))
                # 全部追加すると多すぎたので、各カテゴリからランダムに20個だけ追加することにする
                propNouns += random.sample(df['label'].tolist(), min(self.num_propNouns_in_cat_for_globalHSMean, len(df)))
                # all_propNouns += df['label'].tolist()

        try:
            E = model.model.embed_tokens.weight
        except:
            E = model.model.language_model.embed_tokens.weight

        # ノルムを語彙中央値
        target_norm = E.norm(dim=1).median().item() 

        # *** term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        layer_to_hsSumVec = {}
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        for term in tqdm(propNouns, miniters=100, desc="Initializing embeddings"):   #all_propNouns:
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
                
                if self.term_vec_type == "single_last":
                    # ** term中の最後のtokenのみでterm_vecを作る場合:
                    last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                    term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
                elif self.term_vec_type == "mean":
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











    # ============================================= 初期化vec作成用関数の実装 ===============================================

    # wikiのsummary入力時の隠れ状態で初期化vecを作成する方法は、global_hidden_meanを計算する必要がないため、ここでは何もしない
    # 'CatCent_by_wikiSummary_HSMixed', 'otherCatCent_by_wikiSummary_HSMixed',


    def initvec_by_uniform(self, model, train_token2tokenid):
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
        


        # ============================================================================================

        elif init_vec_type == 'category_centroid_plus_random':
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
        
        elif init_vec_type == 'other_category_centroid_plus_random':
            # [WIP] *** (Type1) category_centroid_plus_random の対。他のカテゴリの中心vec + 他のカテゴリのランダムvecで初期化. 新概念 apple の初期化にvehicleカテゴリの代表ベクトルを利用するなど ***
            pass
        




        # ============================================================================================

        elif init_vec_type == 'category_COG_by_simple_mean':       
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

        elif init_vec_type == 'other_category_COG_by_simple_mean':
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





        # ============================================================================================ 
        # 2026/03/19
        elif init_vec_type == 'category_centroid_by_hidden_state_mean':
            # *** (Type2) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. ***
            # Type2: 各prop nounをモデルに入力し、語句内の最終token位置における指定層の隠れ状態をその固有名詞のベクトルとする方法。
            # vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)

            for category, init_token_ids in category2initoken_ids.items():
                # カテゴリ固有のcentroid vec作成用固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                
                # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
                init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

                for init_token_id in init_token_ids:
                    # noise として、init_token毎にrandomな10個の固有名詞を選ぶ。これにより、同一カテゴリ内の初期化vec同士に差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    # 埋め込み層のinit_token_id行を、init_termsの指定層における隠れ状態の平均で初期化する
                    model = self.initVecWithMeanVecOfTermHiddenStates(model, tokenizer, init_terms, [init_token_id], layer_idx=layer_idx, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}).") 
            return model



        elif init_vec_type == 'other_category_centroid_by_hidden_state_mean':
            # *** (Type2) 他のカテゴリのCOGで初期化. category_centroid_by_hidden_state_mean の対。例えば、動物カテゴリの新規概念を、場所カテゴリの固有名詞のベクトルの平均で初期化するなど. ***
            # * カテゴリ毎のcentroid vec作成用の固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
            category_to_centroid_terms = {}
            for category, init_token_ids in category2initoken_ids.items():
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                category_to_centroid_terms[category] = init_terms_for_centroid

            for category, init_token_ids in category2initoken_ids.items():
                other_categories = [c for c in category_to_concepts_for_vec.keys() if c != category]
                print(f"Category '{category}' will be initialized with centroid of other categories: {other_categories[:5]}.")

                for init_token_id in init_token_ids:
                    # 他のカテゴリをランダムに選ぶ
                    other_category = random.choice(other_categories)
                    terms_in_other_category = category_to_concepts_for_vec[other_category]
                    init_terms_for_centroid = category_to_centroid_terms.get(other_category, [])    # configでother_categoryに属す固有名詞リストを含めていない場合はcategory_to_centroid_termsにother_categoryが存在しない可能性があるため、getで取得する

                    # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、他カテゴリのリストから centroid vec作成用の固有名詞を削除
                    init_terms_candidate = list(set(terms_in_other_category) - set(init_terms_for_centroid))
                    
                    # noise として、init_token毎にrandomな10個の固有名詞を選び、複数のinit_token_idにおいて他カテゴリとして同じカテゴリを選んだとしても、初期化vecにinit_token_id間で差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    model = self.initVecWithMeanVecOfTermHiddenStates(model, tokenizer, init_terms, [init_token_id], layer_idx=layer_idx, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
            return model
        



        # ============================================================================================ 
        # 2026/03/20
        elif init_vec_type == 'categoryCentroid_by_DebiasedHiddenState':
            """ (Type3) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. 
                更に、等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
            方法: 
                * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
                * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。(type2では、最後のsub-tokenの隠れ状態のみ)
            vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)
            """

            for category, init_token_ids in category2initoken_ids.items():
                # カテゴリ固有のcentroid vec作成用固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                
                # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
                init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

                for init_token_id in init_token_ids:
                    # noise として、init_token毎にrandomな10個の固有名詞を選ぶ。これにより、同一カテゴリ内の初期化vec同士に差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    # 埋め込み層のinit_token_id行を、init_termsの指定層における隠れ状態の平均で初期化する
                    model = self.initVecWithMeanVecOfDebiasedTermHiddenStates(model, tokenizer, init_terms, [init_token_id], layer_idx=layer_idx, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}).") 
            return model



        elif init_vec_type == 'otherCategoryCentroid_by_DebiasedHiddenState':
            # *** (Type3) 他のカテゴリのCOGで初期化. category_centroid_by_debiased_hidden_state の対。***
            category_to_centroid_terms = {}
            for category, init_token_ids in category2initoken_ids.items():
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                category_to_centroid_terms[category] = init_terms_for_centroid

            for category, init_token_ids in category2initoken_ids.items():
                other_categories = [c for c in category_to_concepts_for_vec.keys() if c != category]
                print(f"Category '{category}' will be initialized with centroid of other categories: {other_categories[:5]}.")

                for init_token_id in init_token_ids:
                    # 他のカテゴリをランダムに選ぶ
                    other_category = random.choice(other_categories)
                    terms_in_other_category = category_to_concepts_for_vec[other_category]
                    init_terms_for_centroid = category_to_centroid_terms.get(other_category, [])    # configでother_categoryに属す固有名詞リストを含めていない場合はcategory_to_centroid_termsにother_categoryが存在しない可能性があるため、getで取得する

                    # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、他カテゴリのリストから centroid vec作成用の固有名詞を削除
                    init_terms_candidate = list(set(terms_in_other_category) - set(init_terms_for_centroid))
                    
                    # noise として、init_token毎にrandomな10個の固有名詞を選び、複数のinit_token_idにおいて他カテゴリとして同じカテゴリを選んだとしても、初期化vecにinit_token_id間で差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    model = self.initVecWithMeanVecOfDebiasedTermHiddenStates(model, tokenizer, init_terms, [init_token_id], layer_idx=layer_idx, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
            return model
        



        # ============================================================================================ 
        # 2026/03/21
        elif init_vec_type == 'categoryCentroid_by_DebiasedHSMixed':
            """ (Type4) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. 
                更に、等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
            方法: 
                * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
                * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。
                * 指定した層だけでなく、その前後の層と平均したものをterm_vecとする。例えば、layer_idx=5を指定した場合、layer4, layer5, layer6の隠れ状態の平均をterm_vecとする。
            vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)
            """

            for category, init_token_ids in category2initoken_ids.items():
                # カテゴリ固有のcentroid vec作成用固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                
                # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
                init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

                for init_token_id in init_token_ids:
                    # noise として、init_token毎にrandomな10個の固有名詞を選ぶ。これにより、同一カテゴリ内の初期化vec同士に差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    # 埋め込み層のinit_token_id行を、init_termsの指定層における隠れ状態の平均で初期化する
                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, init_terms, [init_token_id], layer_idx=layer_idx, mix_layers=True, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}).") 
            return model



        elif init_vec_type == 'otherCategoryCentroid_by_DebiasedHSMixed':
            # *** (Type4) 他のカテゴリのCOGで初期化. category_centroid_by_debiased_hidden_state の対。***
            category_to_centroid_terms = {}
            for category, init_token_ids in category2initoken_ids.items():
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                category_to_centroid_terms[category] = init_terms_for_centroid

            for category, init_token_ids in category2initoken_ids.items():
                other_categories = [c for c in category_to_concepts_for_vec.keys() if c != category]
                print(f"Category '{category}' will be initialized with centroid of other categories: {other_categories[:5]}.")

                for init_token_id in init_token_ids:
                    # 他のカテゴリをランダムに選ぶ
                    other_category = random.choice(other_categories)
                    terms_in_other_category = category_to_concepts_for_vec[other_category]
                    init_terms_for_centroid = category_to_centroid_terms.get(other_category, [])    # configでother_categoryに属す固有名詞リストを含めていない場合はcategory_to_centroid_termsにother_categoryが存在しない可能性があるため、getで取得する

                    # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、他カテゴリのリストから centroid vec作成用の固有名詞を削除
                    init_terms_candidate = list(set(terms_in_other_category) - set(init_terms_for_centroid))
                    
                    # noise として、init_token毎にrandomな10個の固有名詞を選び、複数のinit_token_idにおいて他カテゴリとして同じカテゴリを選んだとしても、初期化vecにinit_token_id間で差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, init_terms, [init_token_id], layer_idx=layer_idx, lambda_=LAMBDA_, init_vec_type=init_vec_type, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
            return model
        
# mix_layers=True,  initVecWithGlobalVecDebiasedTermHS

        # ============================================================================================ 
        # 2026/03/21 -2
        elif init_vec_type == 'CatCentroid_by_OthCatDebiasedHSMixed':
            """ (Type5) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. 
                更に、等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
            方法: 
                * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
                    * global_hidden_mean は、他カテゴリ全部の平均で計算する。（Type4は自カテゴリも含めた平均）
                * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。
                * 指定した層だけでなく、その前後の層と平均したものをterm_vecとする。例えば、layer_idx=5を指定した場合、layer4, layer5, layer6の隠れ状態の平均をterm_vecとする。
            vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)
            """

            for own_category, init_token_ids in category2initoken_ids.items():
                # カテゴリ固有のcentroid vec作成用固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
                init_terms_candidate = category_to_concepts_for_vec[own_category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                
                # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
                init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

                for init_token_id in init_token_ids:
                    # noise として、init_token毎にrandomな10個の固有名詞を選ぶ。これにより、同一カテゴリ内の初期化vec同士に差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    # 埋め込み層のinit_token_id行を、init_termsの指定層における隠れ状態の平均で初期化する
                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, own_category, init_terms, [init_token_id], layer_idx=layer_idx, lambda_=0.1, init_vec_type=init_vec_type, mix_layers=True, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{own_category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}).") 
            return model


        elif init_vec_type == 'otherCatCentroid_by_OthCatDebiasedHSMixed':
            # *** (Type4) 他のカテゴリのCOGで初期化. category_centroid_by_debiased_hidden_state の対。***
            # 準備: 各カテゴリのcentroid vec作成用の固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成しておく
            category_to_centroid_terms = {}
            for category, init_token_ids in category2initoken_ids.items():
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                category_to_centroid_terms[category] = init_terms_for_centroid

            # main: 初期化対象token毎に毎回ランダムに選んだ他のカテゴリのCOGで初期化する
            for own_category, init_token_ids in category2initoken_ids.items():
                other_categories = [c for c in category_to_concepts_for_vec.keys() if c != own_category]
                print(f"Category '{own_category}' will be initialized with centroid of other categories: {other_categories[:5]}.")

                for init_token_id in init_token_ids:
                    # 他のカテゴリをランダムに選ぶ
                    other_category = random.choice(other_categories)
                    terms_in_other_category = category_to_concepts_for_vec[other_category]
                    init_terms_for_centroid = category_to_centroid_terms.get(other_category, [])    # configでother_categoryに属す固有名詞リストを含めていない場合はcategory_to_centroid_termsにother_categoryが存在しない可能性があるため、getで取得する

                    # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、他カテゴリのリストから centroid vec作成用の固有名詞を削除
                    init_terms_candidate = list(set(terms_in_other_category) - set(init_terms_for_centroid))
                    
                    # noise として、init_token毎にrandomな10個の固有名詞を選び、複数のinit_token_idにおいて他カテゴリとして同じカテゴリを選んだとしても、初期化vecにinit_token_id間で差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, own_category, init_terms, [init_token_id], layer_idx=layer_idx, lambda_=0.1, init_vec_type=init_vec_type, mix_layers=True, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{own_category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
            return model
        



        # ============================================================================================ 
        # 2026/03/21 -3
        elif init_vec_type == 'CatCent_by_GlbPrimDebiasedHSMixed':
            """ (Type5) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. 
                更に、等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
            方法: 
                * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
                    * global_hidden_mean は、他カテゴリ全部の平均で計算する。（Type4は自カテゴリも含めた平均）
                * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。
                * 指定した層だけでなく、その前後の層と平均したものをterm_vecとする。例えば、layer_idx=5を指定した場合、layer4, layer5, layer6の隠れ状態の平均をterm_vecとする。
            vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)
            """

            for own_category, init_token_ids in category2initoken_ids.items():
                # カテゴリ固有のcentroid vec作成用固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
                init_terms_candidate = category_to_concepts_for_vec[own_category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                
                # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
                init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

                for init_token_id in init_token_ids:
                    # noise として、init_token毎にrandomな10個の固有名詞を選ぶ。これにより、同一カテゴリ内の初期化vec同士に差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    # 埋め込み層のinit_token_id行を、init_termsの指定層における隠れ状態の平均で初期化する
                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, own_category, init_terms, [init_token_id], layer_idx=layer_idx, lambda_=0.1, init_vec_type=init_vec_type, mix_layers=True, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{own_category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}).") 
            return model


        elif init_vec_type == 'otherCatCent_by_GlbPrimDebiasedHSMixed':
            # *** (Type4) 他のカテゴリのCOGで初期化. category_centroid_by_debiased_hidden_state の対。***
            # 準備: 各カテゴリのcentroid vec作成用の固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成しておく
            category_to_centroid_terms = {}
            for category, init_token_ids in category2initoken_ids.items():
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                category_to_centroid_terms[category] = init_terms_for_centroid

            # main: 初期化対象token毎に毎回ランダムに選んだ他のカテゴリのCOGで初期化する
            for own_category, init_token_ids in category2initoken_ids.items():
                other_categories = [c for c in category_to_concepts_for_vec.keys() if c != own_category]
                print(f"Category '{own_category}' will be initialized with centroid of other categories: {other_categories[:5]}.")

                for init_token_id in init_token_ids:
                    # 他のカテゴリをランダムに選ぶ
                    other_category = random.choice(other_categories)
                    terms_in_other_category = category_to_concepts_for_vec[other_category]
                    init_terms_for_centroid = category_to_centroid_terms.get(other_category, [])    # configでother_categoryに属す固有名詞リストを含めていない場合はcategory_to_centroid_termsにother_categoryが存在しない可能性があるため、getで取得する

                    # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、他カテゴリのリストから centroid vec作成用の固有名詞を削除
                    init_terms_candidate = list(set(terms_in_other_category) - set(init_terms_for_centroid))
                    
                    # noise として、init_token毎にrandomな10個の固有名詞を選び、複数のinit_token_idにおいて他カテゴリとして同じカテゴリを選んだとしても、初期化vecにinit_token_id間で差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, own_category, init_terms, [init_token_id], layer_idx=layer_idx, lambda_=0.1, init_vec_type=init_vec_type, mix_layers=True, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{own_category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
            return model



        # ============================================================================================ 
        # 2026/03/21 -4
        elif init_vec_type == 'CatCent_by_GlbPrimDebiasedHS':
            """ (Type5) 各概念のベクトルを、同一カテゴリ内の固有名詞のベクトルの平均(カテゴリの重心と考える)で初期化する方法. 
                更に、等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
            方法: 
                * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
                    * global_hidden_mean は、他カテゴリ全部の平均で計算する。（Type4は自カテゴリも含めた平均）
                * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。
            vec_propnoun = h_last_token_in_propnoun -> 各カテゴリの初期化vec = mean(vec_propnoun_in_concept)
            """

            for own_category, init_token_ids in category2initoken_ids.items():
                # カテゴリ固有のcentroid vec作成用固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成
                init_terms_candidate = category_to_concepts_for_vec[own_category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                
                # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、カテゴリの概念リストから centroid vec作成用の固有名詞を削除
                init_terms_candidate = list(set(init_terms_candidate) - set(init_terms_for_centroid))

                for init_token_id in init_token_ids:
                    # noise として、init_token毎にrandomな10個の固有名詞を選ぶ。これにより、同一カテゴリ内の初期化vec同士に差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    # 埋め込み層のinit_token_id行を、init_termsの指定層における隠れ状態の平均で初期化する
                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, own_category, init_terms, [init_token_id], layer_idx=layer_idx, lambda_=0.1, init_vec_type=init_vec_type, mix_layers=False, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{own_category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}).") 
            return model


        elif init_vec_type == 'otherCatCent_by_GlbPrimDebiasedHS':
            # *** (Type4) 他のカテゴリのCOGで初期化. category_centroid_by_debiased_hidden_state の対。***
            # 準備: 各カテゴリのcentroid vec作成用の固有名詞リスト(propnoun_num_for_init_vec-10 個)を作成しておく
            category_to_centroid_terms = {}
            for category, init_token_ids in category2initoken_ids.items():
                init_terms_candidate = category_to_concepts_for_vec[category]
                init_terms_for_centroid = random.sample(init_terms_candidate, min(len(init_terms_candidate), self.propnoun_num_for_init_vec-10)) 
                category_to_centroid_terms[category] = init_terms_for_centroid

            # main: 初期化対象token毎に毎回ランダムに選んだ他のカテゴリのCOGで初期化する
            for own_category, init_token_ids in category2initoken_ids.items():
                other_categories = [c for c in category_to_concepts_for_vec.keys() if c != own_category]
                print(f"Category '{own_category}' will be initialized with centroid of other categories: {other_categories[:5]}.")

                for init_token_id in init_token_ids:
                    # 他のカテゴリをランダムに選ぶ
                    other_category = random.choice(other_categories)
                    terms_in_other_category = category_to_concepts_for_vec[other_category]
                    init_terms_for_centroid = category_to_centroid_terms.get(other_category, [])    # configでother_categoryに属す固有名詞リストを含めていない場合はcategory_to_centroid_termsにother_categoryが存在しない可能性があるため、getで取得する

                    # centroid vec作成用と、random vec作成用の固有名詞の重複を防ぐため、他カテゴリのリストから centroid vec作成用の固有名詞を削除
                    init_terms_candidate = list(set(terms_in_other_category) - set(init_terms_for_centroid))
                    
                    # noise として、init_token毎にrandomな10個の固有名詞を選び、複数のinit_token_idにおいて他カテゴリとして同じカテゴリを選んだとしても、初期化vecにinit_token_id間で差が生まれる。
                    init_terms_for_random = random.sample(init_terms_candidate, min(len(init_terms_candidate), 10)) 
                    init_terms = init_terms_for_centroid + init_terms_for_random # 中心vec用の固有名詞とランダムvec用の固有名詞を合わせたリストを初期化vec作成に使用

                    model = self.initVecWithGlobalVecDebiasedTermHS(model, tokenizer, own_category, init_terms, [init_token_id], layer_idx=layer_idx, lambda_=0.1, init_vec_type=init_vec_type,  mix_layers=False, print_flag=print_flag)
                    print(f"Initialized new token {tokenizer.decode(init_token_id)} in category '{own_category}' with layer {layer_idx}'s hidden state of {len(init_terms)} concepts: ... ({init_terms[-15:]}), from other category '{other_category}'.")
            return model


        # ============================================================================================
        else: 
            # ** 指定の語句で初期化 (句の場合は単純にmean poolingする) **
            init_terms = [init_vec_type]  # 'a chair' など
            model = self.initVecWithTokenVec(model, tokenizer, init_terms, trainTokenIds, print_flag=print_flag)
            return model
























    # def initVecWithGlobalVecDebiasedTermHS(
    #         self,
    #         model, 
    #         tokenizer, 
    #         own_category,
    #         init_terms, 
    #         init_target_ids, 
    #         layer_idx,
    #         lambda_,
    #         init_vec_type,
    #         print_flag=False
    #     ):
    #     """語句をモデルに入力し、語句中の最終tokenを入れた後の、モデル内の指定層における、中心化した隠れ状態をその語句のベクトルとし、
    #     埋め込み層の特定の行を、指定した語句の集合のベクトルの平均で初期化する関数.
    #     2026/03/21-3

    #     * 等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
    #     * 近傍層に特徴が分散している可能性があるので、前後3層の隠れ状態を平均する
    #     方法: 
    #         * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
    #             * global_hidden_mean は 全カテゴリで、PCAによる主成分で構成したベクトルを使用する
    #         * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。(type2では、最後のsub-tokenの隠れ状態のみ)
    #     Args:
    #         - model: HuggingFaceのモデルオブジェクト
    #         - tokenizer: HuggingFaceのトークナイザオブジェクト
    #         - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
    #         - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
    #         - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
    #         - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
    #     """
    #     try:
    #         E = model.model.embed_tokens.weight
    #         num_hidden_layers = model.config.num_hidden_layers
    #     except:
    #         E = model.model.language_model.embed_tokens.weight
    #         num_hidden_layers = model.config.text_config.num_hidden_layers
    #     if print_flag:
    #         print(f"⭐️num_hidden_layers: {num_hidden_layers}")

    #     # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
    #     valid_init_term_count = 0   # ""でない有効なtermの数をカウント
    #     sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成

    #     for term in init_terms:
    #         if term.strip() == "":
    #             continue
    #         valid_init_term_count += 1
    #         inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

    #         # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
    #         with torch.no_grad():
    #             out = model(**inputs, output_hidden_states=True)
    #         layer_hs = out.hidden_states[layer_idx]
            
    #         # *** : *** [⭐️WIP]        
    #         if self.term_vec_type == "single_last":
    #             # ** term中の最後のtokenのみでterm_vecを作る場合:
    #             last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
    #             term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
    #         elif self.term_vec_type == "mean":
    #             # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
    #             seq_len = inputs["attention_mask"].sum().item()
    #             term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [d]
    
    #         sum_vec += term_vec


    #     # 2. term間の平均vecを計算
    #     if sum_vec.norm().item() == 0.0 or valid_init_term_count == 0:
    #         raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
    #     own_centroid = sum_vec / valid_init_term_count


    #     # 3. 中心化: 隠れ層の平均vecを引き、カテゴリ代表vec間の方向の差を目立たせる
    #     if init_vec_type in ['CatCent_by_GlbPrimDebiasedHSMixed', 'otherCatCent_by_GlbPrimDebiasedHSMixed']:
    #         pcs = self.global_primary_vec_by_layer.get(layer_idx).to(
    #             device=own_centroid.device,
    #             dtype=own_centroid.dtype
    #         )
    #         # pcs = torch.stack(pcs, dim=0)   # [3, d]
    #         proj = (own_centroid @ pcs.T) @ pcs                  # [d]
    #         # init_src = own_centroid - lambda_ * proj
    #         global_vec = proj
    #     elif init_vec_type in ['CatCentroid_by_OthCatDebiasedHSMixed', 'otherCatCentroid_by_OthCatDebiasedHSMixed']:
    #         global_vec = self.category_to_layer_to_otherHSMeanVec[own_category][layer_idx]
    #     elif init_vec_type in ['categoryCentroid_by_DebiasedHSMixed', 'otherCategoryCentroid_by_DebiasedHSMixed']:
    #         global_vec = self.layer_to_globalHSMeanVec[layer_idx]

    #     global_vec = global_vec.to(device=own_centroid.device, dtype=own_centroid.dtype)

    #     init_src = own_centroid - lambda_ * global_vec
    #     # [memp これはだめな手法] own_centroid のノルムに対してlambda_倍した大きさのノルムの global_vec を引く. global_vec は、全カテゴリで、PCAによる主成分で構成したベクトル
    #     # init_src = own_centroid - (global_vec / global_vec.norm().clamp_min(1e-12) * own_centroid.norm().clamp_min(1e-12)) * lambda_   # global_vecをown_centroidのノルムに合わせてスケーリングしてから引く

    #     # 4. 微小ノイズを加える
    #     d = init_src.shape[0]

    #     # 微小ノイズを作る
    #     noise = torch.randn(d, device=E.device, dtype=E.dtype)

    #     # 各行をL2正規化して「方向だけランダム」にする
    #     eps = 1e-12
    #     noise = noise / noise.norm(p=2, dim=0, keepdim=True).clamp_min(eps)

    #     # ノイズの大きさを、重心ノルムのごく一部にする
    #     noise_scale = NOISE_SCALE   # まずは 1e-3 あたりから試す 1e-3だと少ししか改善しなかった, 1e-2だとother_category_COGの方がaccが高くなった 3e-3はいいかんじ。 2e-3はまだ試していないが後で試す
    #     init_norm = init_src.norm(p=2).clamp_min(eps)
    #     noise = noise * (init_norm * noise_scale)

    #     # 重心 + 微小ノイズ
    #     init_src = init_src + noise

    #     # 4. ノルムを語彙中央値に合わせる [memo] hidde stateのノルムは埋め込み層のノルムと大きく異なる可能性があるため、ノルムを合わせる
    #     target_norm = E.norm(dim=1).median().item()  # 埋め込み行のノルムの中央値をターゲットノルムとする
    #     init_src_norm = init_src.norm().item()
    #     if init_src_norm > 0:
    #         init_src = init_src / init_src_norm * target_norm  # ターゲットノルムに合わせてスケーリング

    #     # 5. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化.
    #     with torch.no_grad():
    #         init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
    #         src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
    #         E.index_copy_(dim=0, index=init_target_ids, source=src)
        
    #     return model




# initVecWithGlobalVecDebiasedTermHSsMixed
# mix_layers initVecWithGlobalVecDebiasedTermHS
    #[⭐️WIP]
    def initVecWithGlobalVecDebiasedTermHS(
            self,
            model, 
            tokenizer, 
            own_category,
            init_terms, 
            init_target_ids, 
            layer_idx,
            lambda_,
            init_vec_type,
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
        - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
        - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
        - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """
        try:
            E = model.model.embed_tokens.weight
            num_hidden_layers = model.config.num_hidden_layers
        except:
            E = model.model.language_model.embed_tokens.weight
            num_hidden_layers = model.config.text_config.num_hidden_layers
        if print_flag:
            print(f"⭐️num_hidden_layers: {num_hidden_layers}")

        # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        if mix_layers:
            if layer_idx == -1 or layer_idx == num_hidden_layers:
                mixed_layer = [-1, -2, -3]  # 最終層とその前の2層を平均する
            elif layer_idx == 0:
                mixed_layer = [0, 1, 2]  # 最初の層とその後の2層を平均する
            elif layer_idx != None:
                mixed_layer = [layer_idx-1, layer_idx, layer_idx+1]  # 指定層の前後3層を平均する
            else:
                raise ValueError(f"Invalid layer_idx: {layer_idx}. Must be -1, 0, or a positive integer less than num_hidden_layers.")


        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
        for term in init_terms:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            

            if mix_layers:
                # *** 前後3層の隠れ状態を平均する: ***
                layer_hs = torch.stack(
                    [out.hidden_states[lid] for lid in mixed_layer],
                    dim=0
                )  # 指定層の出力 [3, 1, seq_len, d]

                if self.term_vec_type == "single_last":
                    last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                    term_vec = layer_hs[:, 0, last_token_idx, :].mean(dim=0)   # [d] 前後3層の最後のtokenの隠れ状態を平均する. 
                    # sum_vec += term_vec
                    
                elif self.term_vec_type == "mean":
                    # *** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                    seq_len = inputs["attention_mask"].sum().item()
                    term_vec = layer_hs[:, 0, :seq_len, :].mean(dim=1).mean(dim=0)    # [3, 1, seq_len, d] -> [seq_len, d] -> [d]
                    # sum_vec += term_vec
            else:
                # 単層の場合
                layer_hs = out.hidden_states[layer_idx]
                
                # *** : *** [⭐️WIP]        
                if self.term_vec_type == "single_last":
                    # ** term中の最後のtokenのみでterm_vecを作る場合:
                    last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                    term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
                elif self.term_vec_type == "mean":
                    # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                    seq_len = inputs["attention_mask"].sum().item()
                    term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [d]
        
            sum_vec += term_vec

        # 2. term間の平均vecを計算
        if sum_vec.norm().item() == 0.0 or valid_init_term_count == 0:
            raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
        own_centroid = sum_vec / valid_init_term_count


        # 3. 中心化: 隠れ層の平均vecを引き、カテゴリ代表vec間の方向の差を目立たせる
        if init_vec_type in ['CatCent_by_GlbPrimDebiasedHSMixed', 'otherCatCent_by_GlbPrimDebiasedHSMixed',
                            'CatCent_by_GlbPrimDebiasedHS', 'otherCatCent_by_GlbPrimDebiasedHS']:
            pcs = self.global_primary_vec_by_layer.get(layer_idx).to(
                device=own_centroid.device,
                dtype=own_centroid.dtype
            )
            proj = (own_centroid @ pcs.T) @ pcs                  # [d]
            global_vec = proj
        
        # elif init_vec_type in ['CatCent_by_GlbPrimDebiasedHS', 'otherCatCent_by_GlbPrimDebiasedHS']:
        #     pcs = self.global_primary_vec_by_layer.get(layer_idx).to(
        #         device=own_centroid.device,
        #         dtype=own_centroid.dtype
        #     )
        #     proj = (own_centroid @ pcs.T) @ pcs                  # [d]
        #     global_vec = proj
        elif init_vec_type in ['CatCentroid_by_OthCatDebiasedHSMixed', 'otherCatCentroid_by_OthCatDebiasedHSMixed']:
            global_vec = self.category_to_layer_to_otherHSMeanVec[own_category][layer_idx]
        elif init_vec_type in ['categoryCentroid_by_DebiasedHSMixed', 'otherCategoryCentroid_by_DebiasedHSMixed']:
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

        # 5. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化.
        with torch.no_grad():
            init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
            src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
            E.index_copy_(dim=0, index=init_target_ids, source=src)
        
        return model




    def initVecWithOtherCatGlobalVecDebiasedTermHSsMixed(
            self,
            model, 
            tokenizer, 
            own_category,
            init_terms, 
            init_target_ids, 
            layer_idx,
            print_flag=False
        ):
        """語句をモデルに入力し、語句中の最終tokenを入れた後の、モデル内の指定層における、中心化した隠れ状態をその語句のベクトルとし、
        埋め込み層の特定の行を、指定した語句の集合のベクトルの平均で初期化する関数.
        2026/03/21-2

        * 等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
        * 近傍層に特徴が分散している可能性があるので、前後3層の隠れ状態を平均する
    方法: 
        * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
            * ただし、global_hidden_mean は、他カテゴリ全部の平均で計算する。（Type4は自カテゴリも含めた平均）
        * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。(type2では、最後のsub-tokenの隠れ状態のみ)
        Args:
        - model: HuggingFaceのモデルオブジェクト
        - tokenizer: HuggingFaceのトークナイザオブジェクト
        - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
        - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
        - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
        - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """
        try:
            E = model.model.embed_tokens.weight
            num_hidden_layers = model.config.num_hidden_layers
        except:
            E = model.model.language_model.embed_tokens.weight
            num_hidden_layers = model.config.text_config.num_hidden_layers
        if print_flag:
            print(f"⭐️num_hidden_layers: {num_hidden_layers}")

        # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成

        if layer_idx == -1 or layer_idx == num_hidden_layers:
            mix_layers = [-1, -2, -3]  # 最終層とその前の2層を平均する
        elif layer_idx == 0:
            mix_layers = [0, 1, 2]  # 最初の層とその後の2層を平均する
        else:
            mix_layers = [layer_idx-1, layer_idx, layer_idx+1]  # 指定層の前後3層を平均する

        for term in init_terms:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            
            # *** 前後3層の隠れ状態を平均する: ***
            layer_hs = torch.stack(
                [out.hidden_states[lid] for lid in mix_layers],
                dim=0
            )  # 指定層の出力 [3, 1, seq_len, d]

            if self.term_vec_type == "single_last":
                last_token_idx = inputs["attention_mask"].sum(dim=1) - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                term_vec = layer_hs[:, 0, last_token_idx, :].mean(dim=0)   # [d] 前後3層の最後のtokenの隠れ状態を平均する. 
                sum_vec += term_vec

            elif self.term_vec_type == "mean":
                # *** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                seq_len = inputs["attention_mask"].sum().item()
                term_vec = layer_hs[:, 0, :seq_len, :].mean(dim=1).mean(dim=0)    # [3, 1, seq_len, d] -> [seq_len, d] -> [d]
                sum_vec += term_vec



        # 2. term間の平均vecを計算
        if sum_vec.norm().item() == 0.0 or valid_init_term_count == 0:
            raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
        own_centroid = sum_vec / valid_init_term_count

        # 3. 中心化: 隠れ層の平均vecを引き、カテゴリ代表vec間の方向の差を目立たせる
        # own_centroid -= self.category_to_layer_to_otherHSMeanVec[own_category][layer_idx]
        lambda_ = 0.1
        other_mean = self.category_to_layer_to_otherHSMeanVec[own_category][layer_idx]
        init_src = own_centroid - lambda_ * other_mean


        # 4. 微小ノイズを加える
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


        # 4. ノルムを語彙中央値に合わせる [memo] hidde stateのノルムは埋め込み層のノルムと大きく異なる可能性があるため、ノルムを合わせる
        target_norm = E.norm(dim=1).median().item()  # 埋め込み行のノルムの中央値をターゲットノルムとする
        init_src_norm = init_src.norm().item()
        if init_src_norm > 0:
            init_src = init_src / init_src_norm * target_norm  # ターゲットノルムに合わせてスケーリング


        # 5. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化.
        with torch.no_grad():
            init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
            src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
            E.index_copy_(dim=0, index=init_target_ids, source=src)
        
        return model





    def initVecWithDebiasedTermHiddenStatesMixed(
            self,
            model, 
            tokenizer, 
            init_terms, 
            init_target_ids, 
            layer_idx,
            print_flag=False
        ):
        """語句をモデルに入力し、語句中の最終tokenを入れた後の、モデル内の指定層における、中心化した隠れ状態をその語句のベクトルとし、
        埋め込み層の特定の行を、指定した語句の集合のベクトルの平均で初期化する関数.
        2026/03/21

        * 等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
        * 近傍層に特徴が分散している可能性があるので、前後3層の隠れ状態を平均する
    方法: 
        * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
        * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。(type2では、最後のsub-tokenの隠れ状態のみ)
        Args:
        - model: HuggingFaceのモデルオブジェクト
        - tokenizer: HuggingFaceのトークナイザオブジェクト
        - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
        - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
        - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
        - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """
        try:
            E = model.model.embed_tokens.weight
            num_hidden_layers = model.config.num_hidden_layers
        except:
            E = model.model.language_model.embed_tokens.weight
            num_hidden_layers = model.config.text_config.num_hidden_layers

        # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成

        if layer_idx == -1 or layer_idx == num_hidden_layers:
            mix_layers = [-1, -2, -3]  # 最終層とその前の2層を平均する
        elif layer_idx == 0:
            mix_layers = [0, 1, 2]  # 最初の層とその後の2層を平均する
        else:
            mix_layers = [layer_idx-1, layer_idx, layer_idx+1]  # 指定層の前後3層を平均する

        for term in init_terms:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            
            # *** 前後3層の隠れ状態を平均する: ***
            layer_hs = torch.stack(
                [out.hidden_states[lid] for lid in mix_layers],
                dim=0
            )  # 指定層の出力 [3, 1, seq_len, d]
            # ** term中の最後のtokenのみでterm_vecを作る場合:
            ### shortに書くなら:
            # last_token_idx = inputs["attention_mask"].sum(dim=1) - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
            # h_last_token = layer_hs[:, 0, last_token_idx, :].mean(dim=0)   # [d] 前後3層の最後のtokenの隠れ状態を平均する. 
            ### forループで書くなら:
            # h_last_token = torch.zeros_like(E[0])
            # for idx in mix_layers:
            #     layer_hs = out.hidden_states[idx]   # 指定層の出力 [1(=batch_size), seq_len, d]

            #     # ** term中の最後のtokenのみでterm_vecを作る場合:
            #     last_token_idx = inputs["attention_mask"].sum(dim=1) - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
            #     h_last_token += layer_hs[0, last_token_idx.item()]           # [1, seq_len, d] -> [d]

            #     # # *** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
            #     # seq_len = inputs["attention_mask"].sum().item()
            #     # term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [1, seq_len, d] -> [seq_len, d] -> [d]
            # h_last_token /= len(mix_layers)   # 前後3層の平均を取る
            
            # sum_vec += h_last_token

            if self.term_vec_type == "single_last":
                last_token_idx = inputs["attention_mask"].sum(dim=1) - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                term_vec = layer_hs[:, 0, last_token_idx, :].mean(dim=0)   # [d] 前後3層の最後のtokenの隠れ状態を平均する. 
                sum_vec += term_vec

            elif self.term_vec_type == "mean":
                # *** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                seq_len = inputs["attention_mask"].sum().item()
                term_vec = layer_hs[:, 0, :seq_len, :].mean(dim=1).mean(dim=0)    # [3, 1, seq_len, d] -> [seq_len, d] -> [d]
                sum_vec += term_vec



        # 2. term間の平均vecを計算
        if sum_vec.norm().item() == 0.0 or valid_init_term_count == 0:
            raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
        init_src = sum_vec / valid_init_term_count

        # 3. 中心化: 隠れ層の平均vecを引き、カテゴリ代表vec間の方向の差を目立たせる
        init_src -= self.layer_to_globalHSMeanVec[layer_idx]


        # 4. 微小ノイズを加える
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


        # 4. ノルムを語彙中央値に合わせる [memo] hidde stateのノルムは埋め込み層のノルムと大きく異なる可能性があるため、ノルムを合わせる
        target_norm = E.norm(dim=1).median().item()  # 埋め込み行のノルムの中央値をターゲットノルムとする
        init_src_norm = init_src.norm().item()
        if init_src_norm > 0:
            init_src = init_src / init_src_norm * target_norm  # ターゲットノルムに合わせてスケーリング


        # 5. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化.
        with torch.no_grad():
            init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
            src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
            E.index_copy_(dim=0, index=init_target_ids, source=src)
        
        return model



    def initVecWithMeanVecOfDebiasedTermHiddenStates(
            self,
            model, 
            tokenizer, 
            init_terms, 
            init_target_ids, 
            layer_idx,
            print_flag=False
        ):
        """語句をモデルに入力し、語句中の最終tokenを入れた後の、モデル内の指定層における、中心化した隠れ状態をその語句のベクトルとし、
        埋め込み層の特定の行を、指定した語句の集合のベクトルの平均で初期化する関数.
        2026/03/20

        等方性(異方性?)を解消し、カテゴリ間のvecが類似することを防ぐため、生の hidden state 平均ではなく、中心化([WIP]・白色化)してから centroid を作る。
    方法: 
        * 収集した全固有名詞 の hidden state 平均(global_hidden_mean)を引く. term_vec = term_vec - global_hidden_mean
        * また、最後のtokenだけでなく、全sub-tokenにおける隠れ状態を平均する。(type2では、最後のsub-tokenの隠れ状態のみ)
        Args:
        - model: HuggingFaceのモデルオブジェクト
        - tokenizer: HuggingFaceのトークナイザオブジェクト
        - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
        - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
        - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
        - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """
        try:
            E = model.model.embed_tokens.weight
        except:
            E = model.model.language_model.embed_tokens.weight

        # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
        for term in init_terms:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            # inputs = tokenizer(term, return_tensors="pt").to(model.device)
            inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            layer_hs = out.hidden_states[layer_idx]   # 指定層の出力 [1(=batch_size), seq_len, d]

            if self.term_vec_type == "single_last":
                # ** term中の最後のtokenのみでterm_vecを作る場合:
                last_token_idx = inputs["attention_mask"].sum(dim=1) - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                h_last_token = layer_hs[0, last_token_idx.item()]           # [1, seq_len, d] -> [d]
                sum_vec += h_last_token
            elif self.term_vec_type == "mean":
                # *** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                seq_len = inputs["attention_mask"].sum().item()
                term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [1, seq_len, d] -> [seq_len, d] -> [d]
                sum_vec += term_vec


        # 2. term間の平均vecを計算
        if sum_vec.norm().item() == 0.0 or valid_init_term_count == 0:
            raise ValueError(f"All terms resulted in zero vectors. Cannot initialize with zero vector.")
        init_src = sum_vec / valid_init_term_count

        # 3. 中心化: 隠れ層の平均vecを引き、カテゴリ代表vec間の方向の差を目立たせる
        init_src -= self.layer_to_globalHSMeanVec[layer_idx]


        # 4. 微小ノイズを加える
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


        # 5. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化.
        with torch.no_grad():
            init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
            src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
            E.index_copy_(dim=0, index=init_target_ids, source=src)
        
        return model



    def initVecWithMeanVecOfTermHiddenStates(
            self,
            model, 
            tokenizer, 
            init_terms, 
            init_target_ids, 
            layer_idx,
            print_flag=False
        ):
        """語句をモデルに入力し、語句中の最終tokenを入れた後の、モデル内の指定層における隠れ状態をその語句のベクトルとし、
        埋め込み層の特定の行を、指定した語句の集合のベクトルの平均で初期化する関数.
        Args:
        - model: HuggingFaceのモデルオブジェクト
        - tokenizer: HuggingFaceのトークナイザオブジェクト
        - init_terms: 初期化に使用する語句のリスト (例: ['a chair', 'a table']など). 句の場合は単純にmean poolingする.
        - init_target_ids: 初期化したいtoken_idのリスト (例: [1000, 1001]など)
        - layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。
        - print_flag: 初期化の各ステップでベクトルの長さや値を表示するかどうか
        """
        try:
            E = model.model.embed_tokens.weight
        except:
            E = model.model.language_model.embed_tokens.weight

        # 1. term毎に、語句をモデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとし、そのベクトルを加算
        valid_init_term_count = 0   # ""でない有効なtermの数をカウント
        sum_vec = torch.zeros_like(E[0])  # (d,) ... E[0]と同じshapeとdtypeのゼロベクトルを作成
        for term in init_terms:
            if term.strip() == "":
                continue
            valid_init_term_count += 1
            # inputs = tokenizer(term, return_tensors="pt").to(model.device)
            inputs = tokenizer(term, return_tensors="pt", add_special_tokens=False).to(model.device)    # term内には<unused>が含まれないのでadd_special_tokens=FalseでOK. Trueの場合、last_token_idxで<EOS>の位置を取得してしまう

            # ** モデルに入力して、語句中の最終tokenを入れた後の、モデルの最後の隠れ状態をその語句のベクトルとする **
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            try:
                hs = out.hidden_states[layer_idx]   # 指定層の出力 [1, seq_len, d]
            except IndexError:
                raise ValueError(f"Layer index {layer_idx} is out of range for the model's hidden states. The model has {len(out.hidden_states)} layers of hidden states.")

            if self.term_vec_type == "single_last":
                # ** term中の最後のtokenのみでterm_vecを作る場合:
                last_token_idx = inputs["attention_mask"].sum(dim=1) - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                h_last_token = hs[0, last_token_idx.item()]        # [1, seq_len, d] -> [d]
                sum_vec += h_last_token
            elif self.term_vec_type == "mean":
                # *** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                seq_len = inputs["attention_mask"].sum().item()
                term_vec = hs[0, :seq_len, :].mean(dim=0)   # [1, seq_len, d] -> [seq_len, d] -> [d]
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


        # 5. 埋め込み層のinit_target_idsが指定した<unusedx>を、まとめてinit_srcで初期化.
        with torch.no_grad():
            init_target_ids = torch.as_tensor(init_target_ids, device=E.device, dtype=torch.long)
            src = init_src.unsqueeze(0).expand(len(init_target_ids), -1)   # (n, d) # 全トークンをカテゴリ重心で埋める
            E.index_copy_(dim=0, index=init_target_ids, source=src)
        
        return model
        
        



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
        try:
            E = model.model.embed_tokens.weight  # (vocab, d)
        except:
            E = model.model.language_model.embed_tokens.weight  # (vocab, d)


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
            try:
                E = model.model.embed_tokens.weight  # (vocab, d)
            except:
                E = model.model.language_model.embed_tokens.weight  # (vocab, d)
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

