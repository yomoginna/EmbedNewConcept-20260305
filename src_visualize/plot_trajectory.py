import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
from transformers import AutoTokenizer


# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
sys.path.append(project_root)

from utils.gemma_train_and_test_utils import get_gemma_model_version


def main(args):

    print(f"project_root: {project_root}")

    model_size = args.model_size
    lr = args.lr
    target_concepts_filename = args.target_concepts_filename # 🟠 修正後
    trained_date = args.trained_date
    seeds = args.seeds
    init_vec_types = args.init_vec_types
    # layer_indices = args.layer_indices

    # ****** 全てのベクトルを読み込む -> vectors に全て格納 ******
    print("Loading tracked vectors...")
    initvectype_order = []
    vectors = []
    steps = []
    token_ids = []

    for init_vec_type in args.init_vec_types:
        initvectype_order.append(init_vec_type) # init_vec_typesの順番を記録しておく。これをplotの色分けに使用する
        for seed in args.seeds:
            layer_indices = args.layer_indices                                          # init_vec_typeによってlayer_indices=[None]になることもあるので、init_vec_typeループ毎に取得し直す
            need_layer_flag = 'HS' in init_vec_type or 'HiddenState' in init_vec_type   # 初期化方法名に'隠れ層'が含まれれば、layer_idxの指定が必要な初期化方法とみなす
            if len(layer_indices) < 1 or not need_layer_flag:
                layer_indices = [None]  # layer_idxが不要の初期化方法の場合は、layer_indicesを[None]にして、1回だけループするようにする

            for layer_idx in layer_indices:
                # print(f"\n\n=== Processing seed: {seed}, init_vec_type: {init_vec_type}, layer_idx: {layer_idx} ===")

                # * 読み込むディレクトリのパスを指定 *
                model_version = get_gemma_model_version(model_size)
                model_name_for_dirname = f"gemma-{model_version}-{model_size}B-lr{lr}-{trained_date}"
                if layer_idx is not None:
                    model_name_for_dirname += f"-hidden_layer{layer_idx}"
                model_name_for_dirname += f"-seed{seed}"

                # mem_dir = os.path.join(project_root, "memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")
                # "work04"が大規模データ保存用のストレージ
                mem_dir = os.path.join("/work04/toko/EmbedNewConcept-20260305/memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")

                # * 保存先ディレクトリのパスを指定
                output_dir = os.path.join(project_root, "src_visualize", "output", f"trajectory_pca_plots_{args.pca_components}d")
                os.makedirs(output_dir, exist_ok=True)


                # *** 読み込み ***
                tracked_vectors_path = os.path.join(mem_dir, "tracked_embeddings.npz")
                data = np.load(tracked_vectors_path)

                token_assignment_path = os.path.join(mem_dir, "token_assignment.json")
                with open(token_assignment_path, 'r') as f:
                    conceptForFict2token_map = json.load(f)


                # tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-{model_version}-{model_size}b-it")
                # token_names = [tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in data["token_ids"]]
                
                # *** データの内容を確認 ***
                # token_ids = data["token_ids"]
                steps_tmp = data["steps"]
                # vectors = data["vectors"]
                num_tokens, num_steps, v_dim = data["vectors"].shape

                vectors.append(data["vectors"])



                if num_steps != len(steps_tmp):
                    steps_tmp = np.insert(steps_tmp, 0, 0) # 学習前の埋め込みの状態をdata["vectors"]の冒頭に追加しているが、2026/04/27学習時にはstepsに0を追加し忘れていたため入っていない。その場合は最初に 0 を追加する。これで、stepsは [0, step1, step2, ...] となる。
                steps.append(steps_tmp)
                # print(data["vectors"].shape)
                # print(f"num_tokens: {num_tokens}, num_steps: {num_steps}, v_dim: {v_dim}")

    vectors = np.array(vectors)     # list -> array. shape: (num_initvectype, num_tokens, num_steps, v_dim)
    steps = np.array(steps)         # list -> array. shape: (num_initvectype, num_steps)
    print(f"vectors shape (num_initvectype, num_tokens, num_steps, v_dim): {vectors.shape}")


    # *** PCAで次元削減してプロット ***
    print("Performing PCA and plotting trajectories...")
    num_initvectype, num_tokens, num_steps, v_dim = vectors.shape
    X = vectors.reshape(-1, v_dim)     # 全部のベクトルを1列に並べる
    Y = pca_trajectory(X, n_components=args.pca_components)  # shape: [num_tokens, num_steps, args.pca_components
    Y = Y.reshape(num_initvectype, num_tokens, num_steps, args.pca_components)      # 元の構造(num_tokens, num_steps, n_components)に戻す

    plot_dic = {
        "markersize": 1,
        "linewidth": 1,
        "alpha": 0.8,
        "color": {initvectype: plt.cm.tab20.colors[i % 20] for i, initvectype in enumerate(initvectype_order)}, # token_nameごとに異なる色を割り当てる
        "title": f"Trajectory of embedding vectors during training (model size={model_size}, seed={seed}, layer_idx={layer_idx})",
    }

    # print(len(plot_dic['color']), plot_dic['color'])

    output_path = os.path.join(output_dir, f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}")
    if args.pca_components == 2:
        output_path += ".png"
        plot_trajectory_2d(
            Y, 
            # steps, 
            initvectype_order, 
            output_path, 
            plot_dic, 
            xlim=(-2, 2), 
            ylim=(-2, 2)
        )
    elif args.pca_components == 3:
        output_path += ".html"
        plot_trajectory_3d(
            Y, 
            steps, 
            initvectype_order, 
            output_path, 
            plot_dic
        )
    else:
        raise ValueError(f"Invalid pca_components: {args.pca_components}. Must be 2 or 3.")


    print(f"Trajectory plot is saved to: {output_path}")


def pca_trajectory(vectors, n_components):
    pca = PCA(n_components=n_components)                        # 各ベクトルがn_components次元座標になった
    Y = pca.fit_transform(vectors)                                    
    return Y


def plot_trajectory_2d(
        vectors_2d, 
        # steps, 
        initvectype_order, 
        output_path, 
        plot_dic, 
        xlim=None, 
        ylim=None
    ):
    # vectors.shape: (num_initvectype, num_tokens, num_steps, 2)

    plt.figure(figsize=(10, 8))

    for i, init_vec_type in enumerate(initvectype_order):
        color = plot_dic["color"][init_vec_type]
        print(f"Shape of vectors for {init_vec_type}: {vectors_2d[i].shape} (num_tokens, num_steps, 2)")

        # ** token vec 毎に学習の軌跡を描画する **
        for j, token_traj in enumerate(vectors_2d[i]):

            plt.plot(
                token_traj[:, 0], token_traj[:, 1], # あるtokenの全てのstepにおけるvec軌跡を描画. (x_list, y_list)の形で渡す必要があるため、traj[:, 0], traj[:, 1]
                marker='o', 
                markersize=plot_dic["markersize"], 
                linewidth=plot_dic["linewidth"], 
                alpha=plot_dic["alpha"],
                label=init_vec_type if j == 0 else "", # 最初のtokenの軌跡にだけlabelをつける。これで、legendには各初期化方法が1回ずつ表示される。
                color=color,
            )

            # step番号も点の近くに表示するなら
            # for k, step in enumerate(steps):
            #     plt.text(token_traj[k, 0], token_traj[k, 1], str(step), fontsize=8)
        
            # 始点と終点
            plt.scatter(token_traj[0, 0], token_traj[0, 1], marker="s", s=plot_dic["markersize"]**2)
            plt.scatter(token_traj[-1, 0], token_traj[-1, 1], marker="x", s=plot_dic["markersize"]**2)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(plot_dic["title"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.savefig(output_path)


def rgb_to_str(c):
    return f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"

def plot_trajectory_3d(
        vectors_3d, 
        steps, 
        initvectype_order, 
        output_path, 
        plot_dic
    ):
    # vectors_3d.shape: (num_initvectype, num_tokens, num_steps, 3)
    num_initvectype, num_tokens, num_steps, _ = vectors_3d.shape
    # =========================
    # DataFrame化
    # =========================
    print("PC1:", vectors_3d[:, :, :, 0].reshape(-1).shape)
    df = pd.DataFrame({
        "PC1": vectors_3d[:, :, :, 0].reshape(-1), # (num_initvectype, num_tokens, num_steps, 3) -> (num_initvectype*num_tokens*num_steps, 3) -> (num_initvectype*num_tokens*num_steps)
        "PC2": vectors_3d[:, :, :, 1].reshape(-1),
        "PC3": vectors_3d[:, :, :, 2].reshape(-1),
        "init_vec_type": np.repeat(
                    # 各init_vec_type名を、num_tokens*num_steps回ずつ繰り返す。(vectors_3dの行数は num_initvectype*num_tokens*num_stepsなので, (num_tokens*num_steps) * num_initvectype )
                    initvectype_order, 
                    num_tokens*num_steps
                    # vectors_3d.shape[0] // len(initvectype_order) * vectors_3d.shape[1]
                ), # num_tokens*num_stepsを初期化方法の数で割った数だけ、各初期化方法の名前を繰り返す。これで、vectors_3dの各行に対応する初期化方法の名前が得られる。
        "step": np.tile(steps.reshape(-1), num_tokens) # stepsは (num_initvectype, num_steps) の形をしているので、これをflat化し、 num_tokens 回繰り返す。これで、vectors_3dの各行に対応するstep番号が得られる。
        # "color": np.repeat(
        #             [plot_dic["color"][init_vec_type] for init_vec_type in initvectype_order], 
        #             num_tokens*num_steps,
        #             axis=0  # colorの一要素はrgbの3要素タプルなので、axisを指定しないとrgbの各値も複製されてしまう
        #         ),
    })
    color_map = {
        f"{t}": rgb_to_str(plot_dic["color"][t])
        for t in initvectype_order
    }


    # =========================
    # Plot
    # =========================
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="init_vec_type",
        color_discrete_map=color_map,
        hover_name="step",
        # hover_data={"category": True, "PC1": ':.3f', "PC2": ':.3f', "PC3": ':.3f'},
        title=plot_dic["title"],
    )
    # fig.update_traces(marker=dict(size=6))
    fig.update_traces(
        marker=dict(
            size=plot_dic["markersize"], 
            opacity=plot_dic["alpha"]
        )
    )

    # 保存
    fig.write_html(output_path)
    print(f"Plot saved to: {output_path}")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"')
    parser.add_argument('--trained_date', type=str, default='20260427', help='モデルを訓練した日付 (例: "20260305")')
    parser.add_argument('--model_size', type=str, default='12', help='モデルサイズ (例: 4, 9, 12)')
    parser.add_argument('--lr', type=float, default=0.01, help='学習率')
    parser.add_argument('--init_vec_types', type=str, nargs='+', default=['zero', 'uniform', 'norm_rand'], help='memory vectorの初期化方法のリスト. ')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='軌跡をtrackする対象シード値のリスト')
    parser.add_argument("--pool_hs_type", type=str, default="eos", choices=["eos", "last_token", "mean_pool"], help="隠れ状態のプーリング方法。")
    parser.add_argument('--layer_indices', type=int, nargs='*', default=None, help='隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。init_vec_typeが \'category_centroid_by_hidden_state_mean\' の場合に使用')
    parser.add_argument('--pca_components', type=int, default=2, choices=[2, 3], help='PCAで削減する次元数。デフォルトは2で、2次元プロットを作成する。3に設定すると3次元プロットを作成する。')
    args = parser.parse_args()

    main(args)


"""
```sh

TRAINED_DATE=20260427
MODEL_SIZE=12
LR=0.003
TARGET_CONCEPTS_FILENAME="target_concepts_mini_13.json"
SEEDS=(0)
LAYER_INDICES=(12)
POOL_HS_TYPE="mean_pool"
INIT_VEC_TYPES=("CatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "norm_rand_vocab" "zero")
INIT_VEC_TYPES=("CatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed")
PCA_COMPONENTS=3

nohup uv --no-progress run python src_visualize/plot_trajectory.py \
    --target_concepts_filename $TARGET_CONCEPTS_FILENAME \
    --trained_date $TRAINED_DATE \
    --model_size $MODEL_SIZE \
    --lr $LR \
    --init_vec_types ${INIT_VEC_TYPES[@]} \
    --seeds ${SEEDS[@]} \
    --pool_hs_type $POOL_HS_TYPE \
    --layer_indices ${LAYER_INDICES[@]} \
    --pca_components $PCA_COMPONENTS \
    > logs/plot_trajectory_${MODEL_SIZE}B.log 2>&1 &
    
```
"""