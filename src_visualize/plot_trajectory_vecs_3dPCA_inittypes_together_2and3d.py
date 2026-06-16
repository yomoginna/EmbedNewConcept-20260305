"""
src/generate_goal_embeddings.py と src/generate_trajectory_embeddings.py で生成したvecの差分vecをpcaで次元削減して、3次元でプロットするコード。
plot_trajectory_vecs_3dPCA.pyとは違い、指定した複数の初期化手法によるvecsの軌跡を一緒にPCAし、同じプロット上に描画するコード。

plot_trajectory_vecs_3dPCA_inittypes_togetherのコードがごちゃごちゃになりすぎたので、単純化する

特徴
- 3次元PCAでプロット+2次元も
- 原点(=目標ベクトルの位置)に最も近い点に行くまでの軌跡を、epochの順番に線で結ぶ
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


memo2026/05/29
- 3dの描画も追加したいので書いているところ。現状では、2dの描画コードをそのまま3dパートにコピペしただけなので、直す。
"""

import os
import sys
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from utils.gemma_train_and_test_utils import get_gemma_model_version

mem_dir = "/work04/toko/EmbedNewConcept-20260305/"


def main(args):

    model_size = args.model_size
    target_concepts_filename = args.target_concepts_filename
    pool_hs_type = args.pool_hs_type
    init_vec_type_list = args.init_vec_type_list
    lr = args.lr
    trained_date = args.trained_date
    init_layer_index = args.init_layer_index
    seed = args.seed
    visualize_layer_index = args.visualize_layer_index
    embed_vec_type = args.embed_vec_type

    model_version = get_gemma_model_version(model_size)

    goal_vector_file = os.path.join(
        mem_dir, "goal_embeddings", embed_vec_type, f"gemma-{model_version}-{model_size}B", pool_hs_type, 
        f"{target_concepts_filename.split('.')[0]}_layerall.npz"
    )
    trajectory_vec_dir = os.path.join(
        mem_dir, 
        "trajectory_embeddings", 
        embed_vec_type, 
        f"gemma-{model_version}-{model_size}B", 
        pool_hs_type
    )
    output_dir = os.path.join(
        project_root, "src_visualize", "output", 
        "pca_plot_diffvecs", 
        f"{model_size}B", 
        embed_vec_type,
        str(pool_hs_type)
    )
    os.makedirs(output_dir, exist_ok=True)

    # ======================
    # 目標ベクトルの読み込み
    # ======================
    goal_data = np.load(goal_vector_file, allow_pickle=True) # (N, D) or (N, H, D)
    goal_vecs = goal_data["vectors"]
    concept_names = goal_data["concept_names"]
    # model_size = goal_data["model_size"]
    # pool_hs_type = goal_data["pool_hs_type"]
    layer_index = goal_data["layer_index"]
    target_concepts_filename = str(goal_data["target_concepts_filename"])
    print(f"goal_vecs.shape: {goal_vecs.shape}")
    print(f"concept_names: {len(concept_names)}, {concept_names}")
    print(f"layer_index: {layer_index}")

    
    # visualize_layer_index を all / list[int] に正規化
    if visualize_layer_index == "all":
        visualize_layer_index_norm = "all"
    elif isinstance(visualize_layer_index, int):
        visualize_layer_index_norm = [visualize_layer_index]
    elif isinstance(visualize_layer_index, str):
        if "," in visualize_layer_index:
            visualize_layer_index_norm = [
                int(v.strip()) for v in visualize_layer_index.split(",")
                if v.strip() != ""
            ]
        else:
            visualize_layer_index_norm = [int(visualize_layer_index)]
    elif isinstance(visualize_layer_index, (list, tuple)):
        visualize_layer_index_norm = [int(v) for v in visualize_layer_index]
    else:
        raise ValueError(f"Invalid visualize_layer_index: {visualize_layer_index}")

    # 描画対象 layer を決定
    if goal_vecs.ndim == 3: # (N, H, D) の場合、
        num_layers = goal_vecs.shape[1]

        if visualize_layer_index_norm == "all":
            visualize_layer_indices = list(range(num_layers))
        else:
            visualize_layer_indices = visualize_layer_index_norm

        for layer_id in visualize_layer_indices:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"visualize_layer_index={layer_id} is out of range. "
                    f"Available layers: 0..{num_layers - 1}"
                )

    elif goal_vecs.ndim == 2:
        if visualize_layer_index_norm == "all":
            visualize_layer_indices = [int(layer_index)] if str(layer_index) != "all" else [-1]
        else:
            visualize_layer_indices = visualize_layer_index_norm
    else:
        raise ValueError(f"goal_vecs must be 2D or 3D, but got shape {goal_vecs.shape}")

    print(f"visualize_layer_indices: {visualize_layer_indices}")



    # ==================================
    # 初期化手法毎の学習過程ベクトルの読み込み
    # ==================================
    initvectype_to_epoch_to_trajectory_vecs = {} # {init_vec_type: {epoch: [conceptname's trajectory_vecs of each layer]}} or {init_vec_type: {epoch: [conceptname's trajectory_vec at one layer]}} 
    for init_vec_type in init_vec_type_list:
        # ** trajectory_vecsのファイル名のフォーマットを作成 **
        need_layer_flag = 'HS' in init_vec_type or 'HiddenState' in init_vec_type   # 初期化方法名に'隠れ層'が含まれれば、layer_idxの指定が必要な初期化方法とみなす
        if not need_layer_flag:
            # HSを初期vec作成に使わない場合は、layer_indexは指定されていないものとして扱う
            trajectory_vec_filename_format = f"{target_concepts_filename.split('.')[0]}_{trained_date}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayerall_epoch<epoch>.npz"
        else:
            trajectory_vec_filename_format = f"{target_concepts_filename.split('.')[0]}_{trained_date}_initlayer{init_layer_index}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayerall_epoch<epoch>.npz"


        # ** epoch毎のtrajectory_vecsのファイルを読み込んで、epoch_to_trajectory_vecsに保存 **
        epoch_to_trajectory_vecs = {}
        for epoch in range(100):  # maxのepochを100と仮定してループするが、実際には存在するepochのファイルのみ読み込むようにする
            # 読み込み
            trajectory_vec_file = trajectory_vec_filename_format.replace("<epoch>", f"{epoch}")
            trajectory_vec_path = os.path.join(trajectory_vec_dir, trajectory_vec_file)
            if os.path.exists(trajectory_vec_path):
                trajectory_data = np.load(trajectory_vec_path, allow_pickle=True)
                trajectory_vecs = trajectory_data["vectors"]
                trajectory_concept_names = trajectory_data["concept_names"]
                print(f"Loaded trajectory vectors from {trajectory_vec_path}, shape: {trajectory_vecs.shape}")

                # ** trajectory_vecsを、goalのconcept_namesの順に並び替える **
                trajectory_vecs = check_concept_name_match_and_sort(
                    concept_names,              # 並び替えたい目標順の名前リスト
                    trajectory_vecs,            # 並び替え対象のリスト
                    trajectory_concept_names    # trajectory_vecs の現在の名前リスト
                )
                epoch_to_trajectory_vecs[epoch] = trajectory_vecs
                print(f"After sorting, trajectory_vecs shape: {trajectory_vecs.shape}")

            else:
                print(f"Trajectory vector file not found for epoch {epoch}: {trajectory_vec_path}")
                # そのepochのファイルがない場合は、それ以降のepochについても学習が行われていないとしてループを抜ける
                break

        initvectype_to_epoch_to_trajectory_vecs[init_vec_type] = epoch_to_trajectory_vecs
        trajectory_epoch_list = sorted(epoch_to_trajectory_vecs.keys())
        print(f"Loaded trajectory vectors for epochs: {trajectory_epoch_list}")
    

    # ======================
    # 描画準備
    # ======================
    # ** 初期化手法毎にplot点の形を変えるためのリストを作成 **
    # scatter3d で使用可能な marker symbol
    raw_symbols = [
        "circle",
        "x",
        "diamond",
        "cross",
        "square",
        "square-open",
        "diamond-open",
        "circle-open",
    ]
    initvectype_to_marker_symbol = {
        init_vec_type: raw_symbols[i % len(raw_symbols)]
        for i, init_vec_type in enumerate(init_vec_type_list)
    }

    point_symbols = []   # 各点のシンボルを格納するリスト. 初期化手法の名前に基づいてシンボルを割り当てる. visualize_layer_indexが複数ある場合は、layer_idに基づいて色の濃さを変え、単一の場合はepochに基づいて色の濃さを変える.
    for concept_name in concept_names:
        for init_vec_type in init_vec_type_list:
            trajectory_epoch_list = sorted(initvectype_to_epoch_to_trajectory_vecs[init_vec_type].keys())
            for epoch in trajectory_epoch_list:
                for layer_id in visualize_layer_indices:
                    point_symbols.append(initvectype_to_marker_symbol[init_vec_type])

    # ** conceptname毎にplot点の色を変えるためのリストを作成 **
    base_palette = px.colors.qualitative.Alphabet     # 例: ['#636EFA', '#EF553B', '#00CC96', ...]
    base_colors = {
        concept_name: base_palette[i % len(base_palette)]
        for i, concept_name in enumerate(concept_names)
    }

    # ** 初期化手法毎に点間の線のスタイルを変えるためのリストを作成 **
    line_dash_list = [
    "solid",
    "dash",
    "longdashdot",
    "dashdot",
    "longdash",
    "longdashdot",
    "dot",
    ]
    initvectype_to_line_dash = {
        init_vec_type: line_dash_list[i % len(line_dash_list)]
        for i, init_vec_type in enumerate(init_vec_type_list)
    }


    # ========================================================================================
    # 初期化手法毎に、目標vec と 学習過程vec の差分を計算しつつ、metadata も同じ順番で作る
    # ========================================================================================
    flattened_diffvecs = []
    concept_names_for_hover = []
    concept_name_and_epoch_and_layer_index = []
    point_symbols = []
    point_colors = []
    max_trajectory_epoch_list = []
    init_vec_types_for_hover = []
    epochs_for_hover = []
    layer_ids_for_hover = []

    for init_vec_type in init_vec_type_list:
        epoch_to_trajectory_vecs = initvectype_to_epoch_to_trajectory_vecs[init_vec_type]
        trajectory_epoch_list = sorted(epoch_to_trajectory_vecs.keys())
        max_trajectory_epoch_list = trajectory_epoch_list if len(trajectory_epoch_list) > len(max_trajectory_epoch_list) else max_trajectory_epoch_list

        if len(trajectory_epoch_list) == 0:
            print(f"No trajectory vectors found for init_vec_type={init_vec_type}")
            continue

        for epoch_pos, epoch in enumerate(trajectory_epoch_list):
            trajectory_vecs = epoch_to_trajectory_vecs[epoch]

            if goal_vecs.ndim == 3:
                if trajectory_vecs.ndim != 3:
                    raise ValueError(
                        f"trajectory_vecs must be 3D when goal_vecs is 3D. "
                        f"Got trajectory_vecs.shape={trajectory_vecs.shape}"
                    )
                for concept_idx, concept_name in enumerate(concept_names):
                    for layer_pos, layer_id in enumerate(visualize_layer_indices):
                        # ** trajectory_vecs と goal_vecs の差分を計算 **
                        diff_vec = (
                            trajectory_vecs[concept_idx, layer_id, :]
                            - goal_vecs[concept_idx, layer_id, :]
                        )
                        flattened_diffvecs.append(diff_vec)

                        concept_names_for_hover.append(concept_name)
                        concept_name_and_epoch_and_layer_index.append(
                            f"{concept_name}, init_vec_type: {init_vec_type}, "
                            f"epoch: {epoch}, layer: {layer_id}"
                        )

                        init_vec_types_for_hover.append(init_vec_type)
                        epochs_for_hover.append(epoch)
                        layer_ids_for_hover.append(layer_id)

                        # ** plot点の形を決める
                        point_symbols.append(initvectype_to_marker_symbol[init_vec_type])

                        # ** plot colorを決める
                        if len(trajectory_epoch_list) <= 1:
                            color_strength = 0.8
                        else:
                            color_strength = 0.25 + 0.75 * (
                                epoch_pos / (len(trajectory_epoch_list) - 1)
                            )
                        point_colors.append(
                            blend_with_white(base_colors[concept_name], color_strength)
                        )

            elif goal_vecs.ndim == 2:
                if trajectory_vecs.ndim != 2:
                    raise ValueError(
                        f"trajectory_vecs must be 2D when goal_vecs is 2D. "
                        f"Got trajectory_vecs.shape={trajectory_vecs.shape}"
                    )
                
                for concept_idx, concept_name in enumerate(concept_names):
                    diff_vec = trajectory_vecs[concept_idx, :] - goal_vecs[concept_idx, :]

                    flattened_diffvecs.append(diff_vec)
                    concept_names_for_hover.append(concept_name)
                    concept_name_and_epoch_and_layer_index.append(
                        f"{concept_name}, init_vec_type: {init_vec_type}, "
                        f"epoch: {epoch}, layer: {visualize_layer_indices[0]}"
                    )
                    point_symbols.append(initvectype_to_marker_symbol[init_vec_type])
                    init_vec_types_for_hover.append(init_vec_type)
                    epochs_for_hover.append(epoch)
                    layer_ids_for_hover.append(visualize_layer_indices[0])

                    if len(trajectory_epoch_list) <= 1:
                        color_strength = 0.8
                    else:
                        color_strength = 0.25 + 0.75 * (
                            epoch_pos / (len(trajectory_epoch_list) - 1)
                        )

                    point_colors.append(
                        blend_with_white(base_colors[concept_name], color_strength)
                    )

    flattened_diffvecs = np.asarray(flattened_diffvecs)

    print(f"flattened_diffvecs.shape: {flattened_diffvecs.shape}")
    print(f"concept_names_for_hover: {len(concept_names_for_hover)}")
    print(f"concept_name_and_epoch_and_layer_index: {len(concept_name_and_epoch_and_layer_index)}")
    print(f"point_symbols: {len(point_symbols)}")
    print(f"point_colors: {len(point_colors)}")

    if flattened_diffvecs.shape[0] == 0:
        raise ValueError("No vectors were loaded. Please check trajectory vector files.")

    if flattened_diffvecs.shape[0] < 3:
        raise ValueError(
            f"PCA(n_components=3) requires at least 3 samples, "
            f"but got {flattened_diffvecs.shape[0]} samples."
        )
    
    # ======================
    # 各 diff_vec の大きさ L2 norm を計算.  ! 元のflattened_diffvecsで距離を計算する。PCA後ではなく前のvec.
    # ======================
    diff_norms = np.linalg.norm(flattened_diffvecs, axis=1)

    print("diff_norms shape:", diff_norms.shape)
    print("diff_norm min:", diff_norms.min())
    print("diff_norm mean:", diff_norms.mean())
    print("diff_norm max:", diff_norms.max())
    # df["diff_norm"] = diff_norms
    
    
    # ======================
    # PCAで次元削減
    # ======================
    pca = PCA(n_components=3)
    coords = pca.fit_transform(flattened_diffvecs) # (N, 3)
    # 寄与率を取得
    explained = pca.explained_variance_ratio_

    

    # ======================
    # DataFrame化
    # ======================
    
    df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "PC3": coords[:, 2],
        "diff_norm": diff_norms,
        "concept_names_for_hover": concept_names_for_hover,
        "init_vec_type": init_vec_types_for_hover,
        "epoch": epochs_for_hover,
        "layer_id": layer_ids_for_hover,
        "concept_name_and_epoch_and_layer_index": concept_name_and_epoch_and_layer_index,
        "point_symbol": point_symbols,
        "point_color": point_colors,
    })

    assert len(df) == len(flattened_diffvecs)
    assert len(df) == len(concept_names_for_hover)
    assert len(df) == len(concept_name_and_epoch_and_layer_index)
    assert len(df) == len(point_symbols)
    assert len(df) == len(point_colors)
    assert len(df) == len(init_vec_types_for_hover)
    assert len(df) == len(epochs_for_hover)
    assert len(df) == len(layer_ids_for_hover)







    # ==================================================================
    # 3D散布図の描画・保存
    # ==================================================================
    marker_size = 3

    title_3d = (
        f"3D PCA of hidden states "
        f"({model_size}B, layer={layer_index}, "
        f"pool_hs_type={pool_hs_type}, init_vec_types=multi)"
    )

    # ======================
    # 3D散布図の描画
    # ======================

    fig_3d = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="concept_names_for_hover",   # conceptname の凡例を出す
        hover_name="concept_name_and_epoch_and_layer_index",
        hover_data={
            "PC1": ':.3f',
            "PC2": ':.3f',
            "PC3": ':.3f',
            "diff_norm": ':.3f',
            "concept_names_for_hover": True,
            "point_symbol": False,
            "point_color": False,
        },
        title=title_3d,
    )

    # concept ごとの trace に対して、対応する symbol/color を設定
    concept_name_set = set(df["concept_names_for_hover"])

    for trace in fig_3d.data:
        concept_name = trace.name

        if concept_name not in concept_name_set:
            continue

        sub_df = df[df["concept_names_for_hover"] == concept_name]

        trace.marker.size = marker_size
        trace.marker.symbol = sub_df["point_symbol"].tolist()
        trace.marker.color = sub_df["point_color"].tolist()
        trace.marker.opacity = 1.0

        trace.legendgroup = concept_name


    # ** 軸ラベルとレイアウト **
    # PCA の各軸がどれくらい情報を持っているか(寄与率)を表示し、3D グラフの X/Y/Z 軸名を設定する
    fig_3d.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({explained[0] * 100:.2f}%)",
            yaxis_title=f"PC2 ({explained[1] * 100:.2f}%)",
            zaxis_title=f"PC3 ({explained[2] * 100:.2f}%)",
        ),
        # width=1300,
        # height=900,
        title=title_3d,
        legend_title_text="Concept",
    )
    x_range = df["PC1"].max() - df["PC1"].min()
    y_range = df["PC2"].max() - df["PC2"].min()
    z_range = df["PC3"].max() - df["PC3"].min()

    ranges = np.array([x_range, y_range, z_range], dtype=float)

    # 0除算対策
    max_range = ranges.max()
    if max_range == 0:
        aspect = np.array([1.0, 1.0, 1.0])
    else:
        aspect = ranges / max_range

    # 細くなりすぎるのを防ぐ
    min_aspect = 0.35
    aspect = np.clip(aspect, min_aspect, 1.0)

    fig_3d.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(
                x=aspect[0],
                y=aspect[1],
                z=aspect[2],
            ),
        )
    )


    # 原点を追加
    fig_3d.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(
                size=marker_size * 1.5,
                color="black",
                opacity=0.5,
            ),
            name="origin",
            showlegend=False,
        )
    )


    # ======================
    # epoch 0 の点を大きく強調表示
    # これがないとepoch0の点が他の点と重なって見つけられなかったため
    # ======================
    epoch0_df = df[df["epoch"] == 0].copy()

    for concept_name, concept_df in epoch0_df.groupby("concept_names_for_hover"):
        fig_3d.add_trace(
            go.Scatter3d(
                x=concept_df["PC1"],
                y=concept_df["PC2"],
                z=concept_df["PC3"],
                mode="markers",
                marker=dict(
                    size=marker_size * 1.5,
                    color=concept_df["point_color"],
                    opacity=1.0,
                    symbol="circle-open",
                    line=dict(
                        width=3,
                        color=base_colors[concept_name],
                    ),
                ),## 最初の描画時点でhoverは作成され表示されているため、ここでもhoverすると両方表示されてしまった。そのためskip
                hoverinfo="skip",
                ## もしhoverを表示させたい場合は、上の１行をコメントアウトし、したのコメントアウトを外す
                # text=concept_df["concept_name_and_epoch_and_layer_index"],
                # customdata=concept_df["diff_norm"].to_numpy(),
                # hovertemplate=(
                #     "<b>%{text}</b><br><br>"
                #     "PC1=%{x:.3f}<br>"
                #     "PC2=%{y:.3f}<br>"
                #     "PC3=%{z:.3f}<br>"
                #     "diff_norm=%{customdata:.3f}<br>"
                #     "<extra></extra>"
                # ),
                name=f"epoch 0: {concept_name}",
                legendgroup=concept_name,
                showlegend=False,
            )
        )


    # ======================
    # 初期化手法ごとのマーカー凡例を追加
    # ======================
    for init_vec_type, marker_symbol in initvectype_to_marker_symbol.items():
        fig_3d.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(
                    size=8,
                    symbol=marker_symbol,
                    color="gray",
                    opacity=1.0,
                ),
                name=f"init: {init_vec_type}",
                showlegend=True,
                visible="legendonly",
                legendgroup="init_vec_type",
            )
        )

    # ======================
    # 各 init_vec_type・concept ごとに、
    # 原点に一番近い点までの軌跡線を追加
    # ======================
    init_vec_type_to_closest_diff_norm = defaultdict(list)
    init_vec_type_to_closest_epoch = defaultdict(list)
    for (init_vec_type, concept_name, layer_id), group_df in df.groupby(
        ["init_vec_type", "concept_names_for_hover", "layer_id"]
    ):
        group_df = group_df.sort_values("epoch").reset_index(drop=True)

        if len(group_df) < 2:
            continue

        closest_pos = group_df["diff_norm"].idxmin()
        traj_df = group_df.iloc[:closest_pos + 1]
        init_vec_type_to_closest_diff_norm[init_vec_type].append(group_df.loc[closest_pos, 'diff_norm'])
        init_vec_type_to_closest_epoch[init_vec_type].append(group_df.loc[closest_pos, 'epoch'])

        print(f"concept_name: {concept_name}, init_vec_type: {init_vec_type}, closest_pos: {closest_pos}, closest_diff_norm: {group_df.loc[closest_pos, 'diff_norm']}, pca2d_coords of closest point: ({group_df.loc[closest_pos, 'PC1']:.3f}, {group_df.loc[closest_pos, 'PC2']:.3f}), layer_id: {layer_id}")
        print(f"\tdiff_norms: {group_df['diff_norm'].tolist()}")
        print(f"\tPC1 coords: {group_df['PC1'].tolist()}")
        if len(traj_df) < 2:
            continue

        fig_3d.add_trace(
            go.Scatter3d(
                x=traj_df["PC1"],
                y=traj_df["PC2"],
                z=traj_df["PC3"],
                mode="lines",
                line=dict(
                    color=hex_to_rgba(base_colors[concept_name], 0.8),
                    width=2,
                    dash=initvectype_to_line_dash[init_vec_type],
                ),
                name=f"path: {concept_name}, {init_vec_type}, layer={layer_id}",
                legendgroup=concept_name,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig_3d.update_layout(
        legend=dict(
            groupclick="togglegroup"
        )
    )


    save_filename_3d = (
        f"{model_size}B"
        f"_{target_concepts_filename.split('.')[0]}"
        f"_initlayer{init_layer_index}"
        f"_seed{seed}"
        "_multi_init_vis"
        f"_vislayer{visualize_layer_index}"
        f"_pca_3d.html"
    )

    output_path_3d = os.path.join(output_dir, save_filename_3d)

    fig_3d.write_html(output_path_3d)
    print(f"3D Plot saved to: {output_path_3d}")






    # ==================================================================
    # 2D散布図の描画・保存
    # ==================================================================
    marker_size = 10
    title_2d = (
        f"2D PCA of hidden states "
        f"({model_size}B, layer={layer_index}, "
        f"pool_hs_type={pool_hs_type}, init_vec_types=multi)"
    )


    # ======================
    # 2D散布図の描画
    # ======================
    fig_2d = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="concept_names_for_hover",   # conceptname の凡例を出す
        hover_name="concept_name_and_epoch_and_layer_index",
        hover_data={
            "PC1": ':.3f',
            "PC2": ':.3f',
            "PC3": ':.3f',
            "diff_norm": ':.3f',
            "concept_names_for_hover": True,
            "point_symbol": False,
            "point_color": False,
        },
        title=title_2d,
    )

    # concept ごとの trace に対して、対応する symbol/color を設定
    concept_name_set = set(df["concept_names_for_hover"])

    for trace in fig_2d.data:
        concept_name = trace.name

        if concept_name not in concept_name_set:
            continue

        sub_df = df[df["concept_names_for_hover"] == concept_name]

        trace.marker.size = marker_size
        trace.marker.symbol = sub_df["point_symbol"].tolist()
        trace.marker.color = sub_df["point_color"].tolist()
        trace.marker.opacity = 1.0

        trace.legendgroup = concept_name


    # ** 軸ラベルとレイアウト **
    x_range = df["PC1"].max() - df["PC1"].min()
    y_range = df["PC2"].max() - df["PC2"].min()

    base_height = 900

    width = int(base_height * x_range / y_range)
    width = max(700, min(width, 1800))

    # PCA の各軸がどれくらい情報を持っているか(寄与率)を表示し、3D グラフの X/Y/Z 軸名を設定する
    fig_2d.update_layout(
        xaxis_title=f"PC1 ({explained[0] * 100:.2f}%)",
        yaxis_title=f"PC2 ({explained[1] * 100:.2f}%)",
        width=width,    # 1300,
        height=base_height,  #900,
        title=title_2d,
        legend_title_text="Concept",
    )

    # 原点を追加
    fig_2d.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(
                size=marker_size * 1.5,
                color="black",
                opacity=0.5,
            ),

            name="origin",
            showlegend=False,
        )
    )

    # ======================
    # epoch 0 の点を大きく強調表示
    # これがないとepoch0の点が他の点と重なって見つけられなかったため
    # ======================
    epoch0_df = df[df["epoch"] == 0].copy()

    for concept_name, concept_df in epoch0_df.groupby("concept_names_for_hover"):
        fig_2d.add_trace(
            go.Scatter(
                x=concept_df["PC1"],
                y=concept_df["PC2"],
                mode="markers",
                marker=dict(
                    size=18,
                    color=concept_df["point_color"],
                    opacity=1.0,
                    symbol="circle-open",
                    line=dict(
                        width=3,
                        color=base_colors[concept_name],
                    ),
                ),
                ## 最初の描画時点でhoverは作成され表示されているため、ここでもhoverすると両方表示されてしまった。そのためskip
                hoverinfo="skip",
                ## もしhoverを表示させたい場合は、上の１行をコメントアウトし、したのコメントアウトを外す
                # text=concept_df["concept_name_and_epoch_and_layer_index"],
                # customdata=concept_df["diff_norm"].to_numpy(),
                # hovertemplate=(
                #     "<b>%{text}</b><br><br>"
                #     "PC1=%{x:.3f}<br>"
                #     "PC2=%{y:.3f}<br>"
                #     "diff_norm=%{customdata:.3f}<br>"
                #     "<extra></extra>"
                # ),
                name=f"epoch 0: {concept_name}",
                legendgroup=concept_name,
                showlegend=False,
            )
        )



    # ======================
    # 初期化手法ごとのマーカー凡例を追加
    # ======================
    for init_vec_type, marker_symbol in initvectype_to_marker_symbol.items():
        fig_2d.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(
                    size=8,
                    symbol=marker_symbol,
                    color="gray",
                    opacity=1.0,
                ),
                name=f"init: {init_vec_type}",
                showlegend=True,
                visible="legendonly",
                legendgroup="init_vec_type",
            )
        )

    # ======================
    # 各 init_vec_type・concept ごとに、
    # 原点に一番近い点までの軌跡線を追加
    # ======================
    for (init_vec_type, concept_name, layer_id), group_df in df.groupby(
        ["init_vec_type", "concept_names_for_hover", "layer_id"]
    ):
        group_df = group_df.sort_values("epoch").reset_index(drop=True)

        if len(group_df) < 2:
            continue

        closest_pos = group_df["diff_norm"].idxmin()
        traj_df = group_df.iloc[:closest_pos + 1]

        print(f"concept_name: {concept_name}, init_vec_type: {init_vec_type}, closest_pos: {closest_pos}, closest_diff_norm: {group_df.loc[closest_pos, 'diff_norm']}, pca2d_coords of closest point: ({group_df.loc[closest_pos, 'PC1']:.3f}, {group_df.loc[closest_pos, 'PC2']:.3f}), layer_id: {layer_id}")
        print(f"\tdiff_norms: {group_df['diff_norm'].tolist()}")
        print(f"\tPC1 coords: {group_df['PC1'].tolist()}")
        if len(traj_df) < 2:
            continue

        fig_2d.add_trace(
            go.Scatter(
                x=traj_df["PC1"],
                y=traj_df["PC2"],
                mode="lines",
                line=dict(
                    color=hex_to_rgba(base_colors[concept_name], 0.8),
                    width=2,
                    dash=initvectype_to_line_dash[init_vec_type],
                ),
                name=f"path: {concept_name}, {init_vec_type}, layer={layer_id}",
                legendgroup=concept_name,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig_2d.update_layout(
        legend=dict(
            groupclick="togglegroup"
        )
    )


    save_filename_2d = (
        f"{model_size}B"
        f"_{target_concepts_filename.split('.')[0]}"
        f"_initlayer{init_layer_index}"
        f"_seed{seed}"
        "_multi_init_vis"
        f"_vislayer{visualize_layer_index}"
        f"_pca_2d.html"
    )

    output_path_2d = os.path.join(output_dir, save_filename_2d)

    fig_2d.write_html(output_path_2d)
    print(f"2D Plot saved to: {output_path_2d}")


    # ======================
    # 統計情報の表示
    # ======================
    print("\n=== Statistics of closest points to origin ===")
    
    # 最も原点に近づいた初期化手法ランキングを作る
    closest_init_vec_type_list = []
    for i in range(len(init_vec_type_to_closest_diff_norm.get(init_vec_type_list[0], []))):
        min_closest_diff_norm = float('inf')
        min_closest_diff_init_vec_type = None
        for init_vec_type in init_vec_type_list:
            closest_diff_norm = init_vec_type_to_closest_diff_norm[init_vec_type][i]
            if closest_diff_norm < min_closest_diff_norm:
                min_closest_diff_norm = closest_diff_norm
                min_closest_diff_init_vec_type = init_vec_type
        closest_init_vec_type_list.append(min_closest_diff_init_vec_type)

    init_vec_type_ranking = pd.Series(closest_init_vec_type_list).value_counts().reset_index()
    init_vec_type_ranking.columns = ["init_vec_type", "count"]
    print(init_vec_type_ranking)

    return 0






# **** 白と基本色を混ぜて、薄い色〜濃い色を作る関数 ****
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    ])
def blend_with_white(hex_color, color_strength):
    """
    color_strength = 0 -> white
    color_strength = 1 -> original color
    """
    rgb = hex_to_rgb(hex_color)
    white = np.array([255, 255, 255])
    mixed = white * (1 - color_strength) + rgb * color_strength
    mixed = mixed.astype(int)
    return f"rgb({mixed[0]}, {mixed[1]}, {mixed[2]})"

def hex_to_rgba(hex_color, alpha):
    """
    hex color -> rgba string
    alpha = 0.0〜1.0
    """
    rgb = hex_to_rgb(hex_color)
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


def check_concept_name_match_and_sort(
    goal_names_order,                 # 並べ替えの目標とする順番に並んだ名前リスト
    sort_target_list,           # 並び替えたいリスト (例: trajectory_vecs)
    sort_target_current_names_order   # 並び替えたいリストの各要素の現在の名前リスト (例: trajectory_concept_names)
    ):
    """
    sort_target_current_names_orderを、goal_names_orderの順番に並び替える。
     - 例えば、goal_names_orderが ['concept1', 'concept2', 'concept3'] で、
        sort_target_current_names_orderが ['concept2', 'concept3', 'concept1'] だった場合、
        sort_target_current_names_orderが ['concept1', 'concept2', 'concept3'] の順番になるよう、
        sort_target_current_names_orderとsort_target_listを同時に並び替える。
     - 並び替えの方法は、sort_target_current_names_orderの各要素がgoal_names_orderのどこにあるかを見て、そのインデックスに基づいてsort_target_listを並び替える。
     - 並び替え後、sort_target_current_names_orderがgoal_names_orderと同じ順番になっていることを確認する。
    """
    if not np.array_equal(goal_names_order, sort_target_current_names_order):
        print(f"Warning: concept names in goal_vecs and trajectory_vecs do not match. Sorting trajectory_vecs to match the order of concept_names in goal_vecs.")
            
        current_name_to_idx = {
            name: idx for idx, name in enumerate(sort_target_list)
        }
        missing_concepts = [
            name for name in goal_names_order
            if name not in current_name_to_idx
        ]
        if len(missing_concepts) > 0:
            raise ValueError(
                "Some concept_names are missing in sort_target_list: "
                f"{missing_concepts}"
            )
        sorted_indices = [
            current_name_to_idx[name]
            for name in goal_names_order
        ]
        sort_target_list = [sort_target_list[i] for i in sorted_indices]
        sort_target_current_names_order = [sort_target_current_names_order[i] for i in sorted_indices]
        print("Sorted trajectory vectors to match concept_names order.")

    assert np.array_equal(goal_names_order, sort_target_current_names_order), (
        "trajectory_concept_names still does not match concept_names after sorting."
    )
    return sort_target_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate goal embeddings for target concepts using a specified Gemma model.")
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"') # *** 🟠 
    parser.add_argument("--model_size", type=int, default=3, help="Size of the Gemma model in billions (e.g., 3 for Gemma-3B).")
    parser.add_argument("--pool_hs_type", type=str, default="mean_pool", help='hidden stateのpooling方法. "eos": 最後のEOSトークンの隠れ状態を使用. "last_token": 最後のトークンの隠れ状態を使用. "mean_pool": テキスト全体の隠れ状態の平均を使用. "repeat_mean_pool": テキストを2回繰り返した内の後のtextの隠れ状態の平均を使用. "dot": テキスト全体の隠れ状態を平均したものと、最後のトークンの隠れ状態を連結して使用.')

    parser.add_argument('--embed_vec_type', type=str, default="by_conceptname", help='conceptname毎の可視化用vectorの作成方法.')
    parser.add_argument('--init_vec_type_list', type=str, nargs='+', default="CatCent_by_WikiSummaryRepeatHSMixed", help='memory vectorの初期化方法')
    parser.add_argument('--lr', type=float, default=0.003, help='学習率. 例: 3e-3')
    parser.add_argument('--trained_date',  type=str, default="", help='学習した日付. 例: "20260427"')
    parser.add_argument('--init_layer_index', type=int, default=12, help='学習時に訓練対象token_vecの初期vecとして使用した層のインデックス. 例: 12')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード. 例: 42')
    parser.add_argument('--visualize_layer_index', type=int, default=None, help='プロットする際の層のインデックス. 例: 12')

    args = parser.parse_args()

    if args.visualize_layer_index is None:
        args.visualize_layer_index = 'all'
    
    main(args)



"""
TARGET_CONCEPTS_FILENAME="target_concepts_mini_13.json"
MODEL_SIZE=12
LR=0.003
INIT_LAYER_INDEX=12
TRAINED_DATE="20260529"
SEED=0

VISUALIZE_LAYER_INDEX=12    # 'all' にすると全層プロットするが、プロットが見づらくなる可能性があるので、特定の層のインデックスを指定した方が良い


INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "farCatCent_by_WikiSummaryRepeatHSMixed" "norm_rand_vocab") 

INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "farCatCent_by_WikiSummaryRepeatHSMixed" "norm_rand_vocab") 

INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "farCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab")


## by_embed_conceptname_in_wikisummaryの場合
EMBED_VEC_TYPE="by_embed_conceptname_in_wikisummary"
POOL_HS_TYPE="target_seq_repeat_mean_pool"

## by_conceptnameの場合
EMBED_VEC_TYPE="by_conceptname"
POOL_HS_TYPE="mean_pool"

## by_embed_conceptname_in_testの場合
EMBED_VEC_TYPE="by_embed_conceptname_in_test"
POOL_HS_TYPE="target_seq_last_token"


nohup uv run python src_visualize/plot_trajectory_vecs_3dPCA_inittypes_together_2and3d.py \
    --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
    --model_size ${MODEL_SIZE} \
    --pool_hs_type ${POOL_HS_TYPE} \
    --embed_vec_type ${EMBED_VEC_TYPE} \
    --init_vec_type_list ${INIT_VEC_TYPE_LIST} \
    --lr ${LR} \
    --trained_date ${TRAINED_DATE} \
    --init_layer_index ${INIT_LAYER_INDEX} \
    --seed ${SEED} \
    --visualize_layer_index ${VISUALIZE_LAYER_INDEX} \
    > log_plot_trajectory_vecs_3dPCA_gemma-${MODEL_SIZE}B_${TARGET_CONCEPTS_FILENAME}.log 2>&1 &


"""