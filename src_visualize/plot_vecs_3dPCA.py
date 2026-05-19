"""
src/generate_goal_embeddings.py 等で生成したvecをpcaで次元削減して、3次元でプロットするコード。

uv run python src_visualize/plot_vecs_3dPCA.py \
    --vector_file "/work04/toko/EmbedNewConcept-20260305/goal_embeddings/gemma-3-4B/goal_embeddings_4B_target_concepts_mini_13_mean_pool_layerall.npz"

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from utils.handle_text_utils import get_year_if_it_is_year, normalize_PublishedIn_facts, create_test_prompt




def main(args):

    vector_file = args.vector_file

    # ベクトルの読み込み
    data = np.load(vector_file, allow_pickle=True) # (N, D) or (N, H, D)
    vecs = data["vectors"]
    concept_names = data["concept_names"]
    model_size = data["model_size"]
    pool_hs_type = data["pool_hs_type"]
    layer_index = data["layer_index"]
    target_concepts_filename = data["target_concepts_filename"]

    print(f"vecs.shape: {vecs.shape}")
    print(f"concept_names: {len(concept_names)}, {concept_names}")
    print(f"layer_index: {layer_index}")


    layer_indices = []
    if layer_index == "all":
        layer_indices = list(range(vecs.shape[1])) # 0からH-1までの層のインデックス
    elif type(layer_index) == list:
        layer_indices = [int(idx) for idx in layer_index] # 指定された層のインデックスのリスト
    else:
        layer_indices = [int(layer_index)] # 指定された層のインデックス

    flattened_vecs = vecs.reshape(-1, vecs.shape[-1]) # 次元Dを固定. (N*H, D) または (N, D)

    concept_names_for_hover = [concept_name for concept_name in concept_names for layer_id in layer_indices]
    concept_name_and_layer_index = [f"{concept_name}_layer{layer_id}" for concept_name in concept_names for layer_id in layer_indices]


    print(f"vecs.shape: {vecs.shape}, flattened_vecs.shape: {flattened_vecs.shape}")
    print(f"concept_names_for_hover: {len(concept_names_for_hover)}")
    print(f"concept_name_and_layer_index: {len(concept_name_and_layer_index)}")

    # PCAで次元削減
    pca = PCA(n_components=3)
    coords = pca.fit_transform(flattened_vecs) # (N, 3)

    # DataFrame化
    df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "PC3": coords[:, 2],
        "concept_names_for_hover": concept_names_for_hover,
        "concept_name_and_layer_index": concept_name_and_layer_index,
    })

    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="concept_names_for_hover",
        hover_name="concept_name_and_layer_index",
        hover_data={"PC1": ':.3f', "PC2": ':.3f', "PC3": ':.3f'},
        title=f"3D PCA of EOS hidden states ({model_size}B, layer={layer_index}, pool_hs_type={pool_hs_type})",
    )
    fig.update_traces(marker=dict(size=6))


    # 保存
    output_dir = os.path.join(project_root, "src_visualize", "output", "pca_plot")
    output_path = os.path.join(output_dir, f"{model_size}B_layer{layer_index}_{target_concepts_filename}_poolHStype_{pool_hs_type}_pca_3d.html")
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(output_path)
    print(f"Plot saved to: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument("--vector_file", type=str, required=True, help="Path to the .npy file containing the vectors to be plotted.")
    args = parser.parse_args()

    main(args)


