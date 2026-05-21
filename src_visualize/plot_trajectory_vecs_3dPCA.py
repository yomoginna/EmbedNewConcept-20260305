"""
src/generate_goal_embeddings.py と src/generate_trajectory_embeddings.py で生成したvecの差分vecをpcaで次元削減して、3次元でプロットするコード。

uv run python src_visualize/plot_trajectory_vecs_3dPCA.py \
...

uv run python src_visualize/plot_trajectory_vecs_3dPCA.py \
    --goal_vector_file "/work04/toko/EmbedNewConcept-20260305/goal_embeddings/gemma-3-12B/goal_embeddings_12B_target_concepts_mini_13_mean_pool_layerall.npz"
    # --trajectory_vector_dir "/work04/toko/EmbedNewConcept-20260305/trajectory_embeddings/gemma-3-12B/repeat_mean_pool/trajectory_embeddings_12B_target_concepts_mini_13_initvecwithCatCent_by_WikiSummaryRepeatHSMixed_vislayerall_epoch10.npz"


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

    goal_vector_file = args.goal_vector_file
    # trajectory_vector_file = args.trajectory_vector_file

    # ======================
    # ベクトルの読み込み
    # ======================
    goal_data = np.load(goal_vector_file, allow_pickle=True) # (N, D) or (N, H, D)
    # trajectory_data = np.load(trajectory_vector_file, allow_pickle=True) # (N, D) or (N, H, D)

    goal_vecs = goal_data["vectors"]
    # trajectory_vecs = trajectory_data["vectors"]
    concept_names = goal_data["concept_names"]
    model_size = goal_data["model_size"]
    pool_hs_type = goal_data["pool_hs_type"]
    layer_index = goal_data["layer_index"]
    target_concepts_filename = str(goal_data["target_concepts_filename"])

    print(f"goal_vecs.shape: {goal_vecs.shape}")
    print(f"concept_names: {len(concept_names)}, {concept_names}")
    print(f"layer_index: {layer_index}")

    layer_indices = []
    if layer_index == "all":
        layer_indices = list(range(goal_vecs.shape[1])) # 0からH-1までの層のインデックス
    elif type(layer_index) == list:
        layer_indices = [int(idx) for idx in layer_index] # 指定された層のインデックスのリスト
    else:
        layer_indices = [int(layer_index)] # 指定された層のインデックス

    # [WIP] plotのhoverやcolorがまだ設定できていない!!
    

    concept_names_for_hover = [concept_name for concept_name in concept_names for layer_id in layer_indices]
    concept_name_and_layer_index = [f"{concept_name}_layer{layer_id}" for concept_name in concept_names for layer_id in layer_indices]
    color_intensity = np.linspace(0, 1, len(layer_indices))                  # 層のインデックスに基づいて色の強さを決定
    color_intensity_norm = (color_intensity - color_intensity.min()) / (color_intensity.max() - color_intensity.min()) # 0〜1に正規化
    color_intensity_norm = np.tile(color_intensity_norm, len(concept_names)) # 各概念について、層の数だけ色の強さを繰り返す

    print(f"concept_names_for_hover: {len(concept_names_for_hover)}")
    print(f"concept_name_and_layer_index: {len(concept_name_and_layer_index)}")
    print(f"color_intensity_norm: {len(color_intensity_norm)}")


    trajectory_vector_dir = "/work04/toko/EmbedNewConcept-20260305/trajectory_embeddings/gemma-3-12B/repeat_mean_pool"
    trajectory_vec_file_format = "trajectory_embeddings_12B_target_concepts_mini_13_initvecwithCatCent_by_WikiSummaryRepeatHSMixed_vislayerall_epoch<epoch>.npz"
    epoch_to_trajectory_vecs = {}
    for epoch in range(11):
        trajectory_vec_file = trajectory_vec_file_format.replace("<epoch>", f"{epoch}")
        trajectory_vec_path = os.path.join(trajectory_vector_dir, trajectory_vec_file)
        if os.path.exists(trajectory_vec_path):
            trajectory_data = np.load(trajectory_vec_path, allow_pickle=True)
            trajectory_vecs = trajectory_data["vectors"]
            print(f"Loaded trajectory vectors from {trajectory_vec_path}, shape: {trajectory_vecs.shape}")
            epoch_to_trajectory_vecs[epoch] = trajectory_vecs
        else:
            print(f"Trajectory vector file not found for epoch {epoch}: {trajectory_vec_path}")

    trajectory_epoch_list = sorted(epoch_to_trajectory_vecs.keys())

    # ======================
    # ベクトルの差分を計算
    # ======================
    vecs = []
    for epoch, trajectory_vecs in epoch_to_trajectory_vecs.items():
        if goal_vecs.ndim == 3 and trajectory_vecs.ndim == 3:
            # (N, H, D) の場合、層ごとに差分を計算して、(N, H, D)の形状を維持
            diff_vecs = - (goal_vecs - trajectory_vecs) # (N, H, D)
        elif goal_vecs.ndim == 2 and trajectory_vecs.ndim == 2:
            # (N, D) の場合、そのまま差分を計算
            diff_vecs = - (goal_vecs - trajectory_vecs) # (N, D)
        else:
            raise ValueError("goal_vecs and trajectory_vecs must have the same number of dimensions (both should be either 2D or 3D).")
        vecs.append(diff_vecs)
    
    print(f"shape of vecs before stacking: {[v.shape for v in vecs]}") # 各epochのvecの形状を表示
    vecs = np.stack(vecs, axis=1) # before: list of (N, H, D) or (N, D) -> after: (N, num_epochs, H, D) or (N, num_epochs, D), 最初の次元が概念の数Nとなるように変換するため、axis=1
    print(f"shape of vecs after stacking: {vecs.shape}")

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
        "concept_name_and_init_layer_index_and_epoch": concept_name_and_layer_index,
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
    parser.add_argument("--goal_vector_file", type=str, required=True, help="Path to the .npy file containing the vectors to be plotted.")
    parser.add_argument("--trajectory_vector_file", type=str, required=True, help="Path to the .npy file containing the vectors to be plotted.")
    args = parser.parse_args()

    main(args)


