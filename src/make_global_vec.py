
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
from transformers import AutoModelForCausalLM, logging as transformers_logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== Runtime config =====
transformers_logging.set_verbosity_error()

project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
# project_root = os.environ["HOME"] # [memo] genkaiを使う場合. "/singularity_home/project/EmbedNewConcept/src/trainMemVec_fromXvec_gemma.py"
sys.path.append(project_root)
print("Project root:", project_root)

N_COMPONENTS = 10


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




def calculateGlobalHSMean_by_GlbPrimDebiasedHSMixed(model_name, term_vec_type, model, tokenizer, n_components, mix_layers=False):
    # ****** (Type3) global_hidden_meanを、全カテゴリの主成分で計算する (各隠れ層の代表vecは、その前後の層との平均) ******
    print("Calculating global hidden state mean for debiasing by global primary components...")
    if mix_layers:
        save_globalPrimComp_dir = os.path.join(project_root, "data", "dbpedia", f"global_primary_components_mixedlayers_{model_name.split('/')[-1]}_{term_vec_type}", "n_components_10") # os.path.join(project_root, "data", "dbpedia", "global_primary_components", f"n_components_{n_components}")
    else:
        save_globalPrimComp_dir = os.path.join(project_root, "data", "dbpedia", f"global_primary_components_singledlayer_{model_name.split('/')[-1]}_{term_vec_type}", "n_components_10") # os.path.join(project_root, "data", "dbpedia", "global_primary_components_singledlayer", f"n_components_{n_components}")

    # dbpediaから収集した全ての固有名詞を収集
    propNoun_dir = os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")
    propNouns = []  # all_propNouns = []
    for category_file in os.listdir(propNoun_dir):
        if category_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(propNoun_dir, category_file))
            # 全部追加すると多すぎたので、各カテゴリからランダムに100個だけ追加することにする
            propNouns += random.sample(df['label'].tolist(), min(100, len(df)))

    try:
        E = model.model.embed_tokens.weight
        num_hidden_layers = model.config.num_hidden_layers
    except:
        E = model.model.language_model.embed_tokens.weight
        num_hidden_layers = model.config.text_config.num_hidden_layers

    # *** 既に保存されている主成分があれば読み込む ***
    global_primary_vec_by_layer = {}
    if n_components <= 10:
        # 最初にn_components=10で計算済みのため、n_componentsが10より少ない場合は10で保存されているファイルから必要なn_components分だけ切り取って使う
        for layer_idx in range(num_hidden_layers+1):
            save_globalPrimComp_path = os.path.join(save_globalPrimComp_dir, f"global_primary_components_layer_{layer_idx}.pt")
            if os.path.exists(save_globalPrimComp_path):
                global_mean_vec, pcs, explained_ratio, meta = load_pca_components(save_globalPrimComp_path)
                global_primary_vec_by_layer[layer_idx] = pcs[:n_components]

        # if n_components != 10:
        #     # もし指定されたn_componentsが10でない場合は、保存されている10成分のファイルから新たにn_components成分のファイルを作る
        #     for layer_idx in range(num_hidden_layers+1):

        # 全ての層の主成分が既に保存されていれば、計算せずに終了する
        if all(layer_idx in global_primary_vec_by_layer for layer_idx in range(num_hidden_layers + 1)):
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
            if layer_idx in global_primary_vec_by_layer.keys():
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

                if term_vec_type == "single_last":
                    # ** term中の最後のtokenのみでterm_vecを作る場合:
                    last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                    term_vec = layer_hs_mix[:, 0, last_token_idx, :].mean(dim=0)   # [1, seq_len, d] -> [d] 前後3層の最後のtokenの隠れ状態を平均する.
                elif term_vec_type == "mean":
                    # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                    seq_len = inputs["attention_mask"].sum().item()
                    term_vec = layer_hs_mix[:, 0, :seq_len, :].mean(dim=1).mean(dim=0)    # [3, 1, seq_len, d] -> [seq_len, d] -> [d]

            # ***** 単一層でterm_vecを作る場合 *****
            else:
                if term_vec_type == "single_last":
                    # ** term中の最後のtokenのみでterm_vecを作る場合:
                    last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1    # 入力語句の最後のtokenのindex ({attention_maskの1の数}-1で計算)
                    term_vec = layer_hs[0, last_token_idx, :]      # [1, seq_len, d] -> [d]
                elif term_vec_type == "mean":
                    # ** term中の全てのsubtokenにおける状態の平均をterm_vecとする場合:
                    seq_len = inputs["attention_mask"].sum().item()
                    term_vec = layer_hs[0, :seq_len, :].mean(dim=0)   # [d]
                    
            layer_to_hsVecs[layer_idx].append(term_vec.cpu())  # GPUからCPUに移してリストに追加


    # *** layer毎に主成分を計算する ***
    for layer_idx, hsVecs in layer_to_hsVecs.items():
        if layer_idx in global_primary_vec_by_layer.keys():
            # 既に生成されたlayerはskipする
            continue
        save_globalPrimComp_path = os.path.join(save_globalPrimComp_dir, f"global_primary_components_layer_{layer_idx}.pt")
        X = torch.stack(hsVecs)
        # PCAで主成分を計算する
        global_mean_vec, pcs, explained_ratio = compute_pca_components(X, n_components)
        global_primary_vec_by_layer[layer_idx] = pcs[:n_components]
        
        # 保存する
        save_pca_components(
            save_path=save_globalPrimComp_path,
            mean_vec=global_mean_vec,
            pcs=pcs,
            explained_ratio=explained_ratio,
            meta={
                "layer_idx": layer_idx,
                "term_vec_type": term_vec_type,
                "mix_layers": mix_layers,
                "num_samples": X.shape[0],
                "hidden_dim": X.shape[1],
            }
        )
        print(global_mean_vec.shape)
        print(pcs.shape)
        print(explained_ratio)


def main():

    # modelとtokenizerのロード
    n_components = N_COMPONENTS
    model_name = f"google/gemma-3-12b-it"  
    term_vec_type = 'single_last'  # "single_last" or "mean"
    mix_layers=True

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")#, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name) #, token=access_token)

    calculateGlobalHSMean_by_GlbPrimDebiasedHSMixed(model_name, term_vec_type, model, tokenizer, n_components, mix_layers=mix_layers)



if __name__ == "__main__":
    main()

"""
nohup uv run src/make_global_vec.py &
3423120
"""
# global_primary_components_mixedlayers_gemma-3-12b-it_single_last
# global_primary_components_singledlayer_gemma-3-12b-it_single_last
