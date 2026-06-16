"""
src/generate_goal_embeddings.py と src/generate_trajectory_embeddings.py で生成したvecの差分vecをpcaで次元削減して、3次元でプロットするコード。
plot_trajectory_vecs_3dPCA.pyとは違い、指定した複数の初期化手法によるvecsの軌跡を一緒にPCAし、同じプロット上に描画するコード。


uv run python src_visualize/plot_trajectory_vecs_3dPCA_inittypes_together.py \


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
import plotly.graph_objects as go
from plotly.validator_cache import ValidatorCache
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

    model_version = get_gemma_model_version(model_size)


    # goal_vector_file = os.path.join(mem_dir, "goal_embeddings", f"gemma-{model_version}-{model_size}B", f"{target_concepts_filename.split('.')[0]}_{pool_hs_type}_layer{visualize_layer_index}.npz")
    goal_vector_file = os.path.join(
        mem_dir, "goal_embeddings", args.embed_vec_type, f"gemma-{model_version}-{model_size}B", pool_hs_type, 
        f"{target_concepts_filename.split('.')[0]}_layerall.npz"
    )
    trajectory_vec_dir = os.path.join(mem_dir, "trajectory_embeddings", args.embed_vec_type, f"gemma-{model_version}-{model_size}B", pool_hs_type)


    # ======================
    # 目標ベクトルの読み込み
    # ======================
    goal_data = np.load(goal_vector_file, allow_pickle=True) # (N, D) or (N, H, D)
    goal_vecs = goal_data["vectors"]
    concept_names = goal_data["concept_names"]
    # model_size = goal_data["model_size"]
    pool_hs_type = goal_data["pool_hs_type"]
    layer_index = goal_data["layer_index"]
    target_concepts_filename = str(goal_data["target_concepts_filename"])
    print(f"goal_vecs.shape: {goal_vecs.shape}")
    print(f"concept_names: {len(concept_names)}, {concept_names}")
    print(f"layer_index: {layer_index}")


    # 全visualize_layer_indexをplotしたら多すぎたので、引数visualize_layer_indexで指定した層のインデックスのみplotするように変更したい
    if visualize_layer_index != 'all':
        if goal_vecs.ndim == 3: # (N, H, D) の場合、指定した層のインデックスに基づいてgoal_vecsをスライスして (N, D) の形状にする
            goal_vecs = goal_vecs[:, visualize_layer_index, :]
        # trajectory_vecsも同様にスライスして (N, num_epochs, D) の形状にする必要があるが、trajectory_vecsはまだ読み込んでいないので、後で読み込む際にスライスするようにする

    visualize_layer_indices = []
    if layer_index == "all" and visualize_layer_index == 'all': # if layer_index == "all":
        visualize_layer_indices = list(range(goal_vecs.shape[1])) # 0からH-1までの層のインデックス
    elif type(visualize_layer_index) == list:
        visualize_layer_indices = [int(idx) for idx in visualize_layer_index] # 指定された層のインデックスのリスト
    else:
        visualize_layer_indices = [int(visualize_layer_index)] # 指定された層のインデックス


    # ==================================
    # 初期化手法毎の学習過程ベクトルの読み込み
    # ==================================
    initvectype_to_epoch_to_trajectory_vecs = {} # {init_vec_type: {epoch: [conceptname's trajectory_vecs of each layer]}} or {init_vec_type: {epoch: [conceptname's trajectory_vec at one layer]}} 
    for init_vec_type in init_vec_type_list:
        # ** trajectory_vecsのファイル名のフォーマットを作成 **
        need_layer_flag = 'HS' in init_vec_type or 'HiddenState' in init_vec_type   # 初期化方法名に'隠れ層'が含まれれば、layer_idxの指定が必要な初期化方法とみなす
        if not need_layer_flag:
            # HSを初期vec作成に使わない場合は、layer_indexは指定されていないものとして扱う
            # trajectory_vec_filename_format = f"{target_concepts_filename.split('.')[0]}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_epoch<epoch>.npz"
            trajectory_vec_filename_format = f"{target_concepts_filename.split('.')[0]}_{trained_date}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayerall_epoch<epoch>.npz"
        else:
            # trajectory_vec_filename_format = f"{target_concepts_filename.split('.')[0]}_initlayer{init_layer_index}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_epoch<epoch>.npz"
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
                print(f"After sorting, trajectory_vecs shape: {trajectory_vecs.shape}")

                # ** visualize_layer_indexに基づいてtrajectory_vecsをスライスして (N, H, D) の場合は (N, D) の形状にする **
                if visualize_layer_index != 'all' and trajectory_vecs.ndim == 3:
                    trajectory_vecs = trajectory_vecs[:, visualize_layer_index, :]
                    print(f"Sliced trajectory_vecs to visualize layer {visualize_layer_index}, new shape: {trajectory_vecs.shape}")

                epoch_to_trajectory_vecs[epoch] = trajectory_vecs
                # trajectory_vecs に含まれるvecの数だけ、concept_name を繰り返して追加する. 
                for concept_name in concept_names:
                    concept_names_for_hover.extend([concept_name] * len(trajectory_vecs) * len(visualize_layer_indices))
                concept_name_and_epoch_and_layer_index.extend(
                    [f"{concept_name}, init_vec_type: {init_vec_type}, epoch: {epoch}, layer: {layer_id}" 
                     for concept_name in concept_names 
                     for layer_id in visualize_layer_indices]
                )

            else:
                print(f"Trajectory vector file not found for epoch {epoch}: {trajectory_vec_path}")
                # そのepochのファイルがない場合は、それ以降のepochについても学習が行われていないとしてループを抜ける
                break

        initvectype_to_epoch_to_trajectory_vecs[init_vec_type] = epoch_to_trajectory_vecs
        trajectory_epoch_list = sorted(epoch_to_trajectory_vecs.keys())
        print(f"Loaded trajectory vectors for epochs: {trajectory_epoch_list}")
    



    # ** epoch毎に色の濃さを変えるためのリストを作成 **
    if len(visualize_layer_indices) == 1:
        # color_intensity_norm = np.array([0.8])  # 単一の層の場合、色の強さを固定
        # 単一層をplotする場合は、epochが大きいほど濃い色にする
        color_intensity = np.linspace(0.2, 1, len(trajectory_epoch_list)) # epochの数に基づいて色の強さを決定
        color_intensity_norm = (color_intensity - color_intensity.min()) / (color_intensity.max() - color_intensity.min()) # 0〜1に正規化
        color_intensity_norm = np.tile(color_intensity_norm, len(concept_names)) # epochの数だけ色の強さを、各概念の数だけ繰り返す
    else:
        color_intensity = np.linspace(0, 1, len(visualize_layer_indices))                  # 層のインデックスに基づいて色の強さを決定
        color_intensity_norm = (color_intensity - color_intensity.min()) / (color_intensity.max() - color_intensity.min()) # 0〜1に正規化
        # intensity を 0.25〜1.0にスケーリングし、薄すぎる点を避ける
        color_intensity_norm = 0.25 + color_intensity_norm * (1.0 - 0.25)
        color_intensity_norm = np.tile(color_intensity_norm, len(concept_names) * len(trajectory_epoch_list)) # 層の数だけ色の強さを、各概念とepochの数だけ繰り返す

    print(f"concept_names_for_hover: {len(concept_names_for_hover)}")
    print(f"concept_name_and_epoch_and_layer_index: {len(concept_name_and_epoch_and_layer_index)}")
    # print(f"color_intensity_norm: {len(color_intensity_norm)}")

    # ** 初期化手法毎にplot点の形を変えるためのリストを作成 **
    # plotlyの散布図で使用可能なマークリストを取得
    SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")  
    raw_symbols = SymbolValidator.values
    initvectype_to_marker_symbol = {}
    for i, init_vec_type in enumerate(init_vec_type_list):
        initvectype_to_marker_symbol[init_vec_type] = raw_symbols[i % len(raw_symbols)] # 初期化手法の数だけ、plotlyのマークリストから順番にシンボルを割り当てる
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
    point_colors = []   # 各点の色を格納するリスト. concept名に基づいて基本色を決定し、さらに層のインデックスに基づいて色の濃さを変える. visualize_layer_indexが複数ある場合は、layer_idに基づいて色の濃さを変え、単一の場合はepochに基づいて色の濃さを変える.
    for concept_name in concept_names:
        for init_vec_type in init_vec_type_list:
            trajectory_epoch_list = sorted(initvectype_to_epoch_to_trajectory_vecs[init_vec_type].keys())
            for epoch in trajectory_epoch_list:
                for layer_id in visualize_layer_indices:
                    base_color = base_colors[concept_name]
                    intensity_amount = color_intensity_norm[layer_id] if len(visualize_layer_indices) > 1 else 0.8 # 単一層の場合は、色の強さを固定
                    point_color = blend_with_white(base_color, intensity_amount)
                    point_colors.append(point_color)

    print(f"point_symbols: {len(point_symbols)}, point_colors: {len(point_colors)}")
    # return 0

    # ======================
    # ベクトルの差分を計算
    # ======================
    # vecs: (initvectype1-epoch1-concept1-layer1, initvectype1-epoch1-concept1-layer2, ..., initvectype1-epoch1-concept2-layer1, initvectype1-epoch1-concept2-layer2, ......., initvectypeN-epochM-conceptX-layerY, ....) の順番でベクトルを格納するリスト. visualize_layer_indexが複数ある場合は、layer_idに基づいて色の濃さを変え、単一の場合はepochに基づいて色の濃さを変える.
    vecs = []   # (初期化手法数 * epoch数, 描画対象の層の数, 隠れ状態の次元) or (初期化手法数 * epoch数, 隠れ状態の次元) の形状の配列になる
    for init_vec_type, epoch_to_trajectory_vecs in initvectype_to_epoch_to_trajectory_vecs.items():
        for epoch, trajectory_vecs in epoch_to_trajectory_vecs.items():
            if goal_vecs.ndim == 3 and trajectory_vecs.ndim == 3:
                # (N, H, D) の場合、層ごとに差分を計算して、(N, H, D)の形状を維持
                diff_vecs = trajectory_vecs - goal_vecs # (N, H, D)
            elif goal_vecs.ndim == 2 and trajectory_vecs.ndim == 2:
                # (N, D) の場合、そのまま差分を計算
                diff_vecs = trajectory_vecs - goal_vecs # (N, D)
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
        "concept_name_and_epoch_and_layer_index": concept_name_and_epoch_and_layer_index,
        # "color_intensity_norm": color_intensity_norm,
        "point_symbol": point_symbols,
        "point_color": point_colors,
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
    # # intensity を 0.25〜1.0 に正規化すると、薄すぎる点を避けられる
    # df["alpha_like"] = 0.25 + 0.75 * df["color_intensity_norm"]
    # df["point_color"] = [
    #     blend_with_white(base_colors[concept_name], amount)
    #     for concept_name, amount in zip(df["concept_names_for_hover"], df["alpha_like"])
    # ]
    
    title=f"3D PCA of hidden states ({model_size}B, layer={layer_index}, pool_hs_type={pool_hs_type}, init_vec_type={init_vec_type})"


    # ======================
    # 3D散布図の描画
    # ======================
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        hover_name="concept_name_and_epoch_and_layer_index",
        hover_data={"PC1": ':.3f', "PC2": ':.3f', "PC3": ':.3f'},
        title=title,
        legend_title_text="Concept",
    )
    fig.update_traces(
        marker=dict(
            size=6,
            symbol=df["point_symbol"],
            color=df["point_color"],
            opacity=0.85,
        )
    )

    # ** 軸ラベルとレイアウト **
    # PCA の各軸がどれくらい情報を持っているか(寄与率)を表示し、3D グラフの X/Y/Z 軸名を設定する
    explained = pca.explained_variance_ratio_
    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({explained[0] * 100:.2f}%)",
            yaxis_title=f"PC2 ({explained[1] * 100:.2f}%)",
            zaxis_title=f"PC3 ({explained[2] * 100:.2f}%)",
        ),
        width=1000,
        height=800,
    )

    # ** タイトルと凡例の更新 **
    fig.update_layout(
        title=title,
        legend_title_text="Concept",
    )
    # (0,0,0)の点を打つ
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(
                size=5,
                color="black",
                opacity=0.5,
            ),
            name="origin",
            showlegend=False,
        )
    )


    # 保存
    output_dir = os.path.join(project_root, "src_visualize", "output", "pca_plot_diffvecs", f"{model_size}B", str(pool_hs_type))
    os.makedirs(output_dir, exist_ok=True)
    save_filename = (
        f"{model_size}B"
        f"_{target_concepts_filename.split('.')[0]}"
        f"_initlayer{init_layer_index}"
        f"_seed{seed}"
        f"_initvecwith{init_vec_type.replace(' ', '_')}"
        f"_vislayer{visualize_layer_index}"
        f"_pca_3d.html"
        
    )
    output_path = os.path.join(
        output_dir, 
        save_filename
        # f"{model_size}B_{target_concepts_filename.split('.')[0]}_initlayer{init_layer_index}_seed{seed}_initvecwith{init_vec_type.replace(' ', '_')}_vislayer{visualize_layer_index}_pca_3d.html")
    )
    
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
NUM_OPTIONS=3
INIT_LAYER_INDEX=12
TRAINED_DATE="20260523"
SEED=0

EMBED_VEC_TYPE="by_conceptname"   # "by_conceptname" 
POOL_HS_TYPE="mean_pool"   # "target_seq_mean_pool" # "repeat_mean_pool" "eos" "mean_pool"
INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "farCatCent_by_WikiSummaryRepeatHSMixed" "norm_rand_vocab") 

INIT_VEC_TYPE_LIST=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "zero" "norm_rand_vocab")

VISUALIZE_LAYER_INDEX=12    # 'all' にすると全層プロットするが、プロットが見づらくなる可能性があるので、特定の層のインデックスを指定した方が良い

nohup uv run python src_visualize/plot_trajectory_vecs_3dPCA_inittypes_together.py \
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