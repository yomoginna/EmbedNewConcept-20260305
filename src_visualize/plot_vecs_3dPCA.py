"""
src/generate_goal_embeddings.py で生成したvecをpcaで次元削減して、3次元でプロットするコード。

uv run python src_visualize/plot_vecs_3dPCA.py \
    --vector_file "/work04/toko/EmbedNewConcept-20260305/goal_embeddings/gemma-3-4B/goal_embeddings_4B_target_concepts_mini_13_mean_pool_layerall.npz"

uv run python src_visualize/plot_vecs_3dPCA.py \
    --vector_file "/work04/toko/EmbedNewConcept-20260305/goal_embeddings/gemma-3-12B/goal_embeddings_12B_target_concepts_mini_13_mean_pool_layerall.npz"


特徴
- 3次元PCAでプロット
- 概念毎に色分け
- ホバーで、概念名と層のインデックスを表示
- 色の濃さは層のインデックスに基づいて変化
    - 層のインデックスが大きいほど濃い色

実行後:
- .htmlファイルが出力されるので、ブラウザで開いてプロットを確認する。
    - まずファイルをDownload
    - ターミナルで、 `open <file_name>.html` を実行してブラウザで開く


結果
- 傾向としては、初期層と最終層が近い位置にplot
- 中間層は、概念間で同じように移動
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

    # ======================
    # ベクトルの読み込み
    # ======================
    data = np.load(vector_file, allow_pickle=True) # (N, D) or (N, H, D)
    vecs = data["vectors"]
    concept_names = data["concept_names"]
    model_size = data["model_size"]
    pool_hs_type = data["pool_hs_type"]
    layer_index = data["layer_index"]
    target_concepts_filename = str(data["target_concepts_filename"])

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

    concept_names_for_hover = [concept_name for concept_name in concept_names for layer_id in layer_indices]
    concept_name_and_layer_index = [f"{concept_name}_layer{layer_id}" for concept_name in concept_names for layer_id in layer_indices]
    color_intensity = np.linspace(0, 1, len(layer_indices))                  # 層のインデックスに基づいて色の強さを決定
    color_intensity_norm = (color_intensity - color_intensity.min()) / (color_intensity.max() - color_intensity.min()) # 0〜1に正規化
    color_intensity_norm = np.tile(color_intensity_norm, len(concept_names)) # 各概念について、層の数だけ色の強さを繰り返す

    print(f"concept_names_for_hover: {len(concept_names_for_hover)}")
    print(f"concept_name_and_layer_index: {len(concept_name_and_layer_index)}")
    print(f"color_intensity_norm: {len(color_intensity_norm)}")

    # ======================
    # PCAで次元削減
    # ======================
    flattened_vecs = vecs.reshape(-1, vecs.shape[-1]) # 次元Dを固定. (N*H, D) または (N, D)
    print(f"vecs.shape: {vecs.shape}, flattened_vecs.shape: {flattened_vecs.shape}")

    pca = PCA(n_components=3)
    coords = pca.fit_transform(flattened_vecs) # (N, 3)

    # ======================
    # DataFrame化
    # ======================
    df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "PC3": coords[:, 2],
        "concept_names_for_hover": concept_names_for_hover,
        "concept_name_and_layer_index": concept_name_and_layer_index,
        "color_intensity_norm": color_intensity_norm,
    })

    # ======================
    # 描画準備
    # ======================
    # base_palette = px.colors.qualitative.Plotly     # 例: ['#636EFA', '#EF553B', '#00CC96', ...]
    base_palette = px.colors.qualitative.Alphabet
    base_colors = {
        concept_name: base_palette[i % len(base_palette)]
        for i, concept_name in enumerate(concept_names)
    }
    # intensity を 0.25〜1.0 に正規化すると、薄すぎる点を避けられる
    df["alpha_like"] = 0.25 + 0.75 * df["color_intensity_norm"]
    df["point_color"] = [
        blend_with_white(base_colors[concept_name], amount)
        for concept_name, amount in zip(df["concept_names_for_hover"], df["alpha_like"])
    ]
    
    title=f"3D PCA of EOS hidden states ({model_size}B, layer={layer_index}, pool_hs_type={pool_hs_type})"


    # ======================
    # 3D散布図の描画
    # ======================
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="concept_names_for_hover",
        hover_name="concept_name_and_layer_index",
        hover_data={"PC1": ':.3f', "PC2": ':.3f', "PC3": ':.3f'},
        title=title,
    )
    fig.update_traces(marker=dict(size=6))

    for trace in fig.data:
        concept_name = trace.name
        sub_df = df[df["concept_names_for_hover"] == concept_name]

        trace.marker.color = sub_df["point_color"].tolist()
        trace.marker.size = 5
        trace.marker.opacity = 1.0

    fig.update_layout(
        title=title,
        legend_title_text="Concept",
    )


    # 保存
    output_dir = os.path.join(project_root, "src_visualize", "output", "pca_plot")
    output_path = os.path.join(output_dir, f"{model_size}B_layer{layer_index}_{target_concepts_filename.split('.')[0]}_poolHStype_{pool_hs_type}_pca_3d.html")
    
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(output_path)
    print(f"Plot saved to: {output_path}")


# **** 白と基本色を混ぜて、薄い色〜濃い色を作る関数 ****
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    ])
def blend_with_white(hex_color, amount):
    """
    amount = 0 -> white
    amount = 1 -> original color
    """
    rgb = hex_to_rgb(hex_color)
    white = np.array([255, 255, 255])
    mixed = white * (1 - amount) + rgb * amount
    mixed = mixed.astype(int)
    return f"rgb({mixed[0]}, {mixed[1]}, {mixed[2]})"




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument("--vector_file", type=str, required=True, help="Path to the .npy file containing the vectors to be plotted.")
    args = parser.parse_args()

    main(args)


