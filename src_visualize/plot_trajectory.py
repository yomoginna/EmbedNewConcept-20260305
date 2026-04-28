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

    seed = args.seed
    init_vec_type = args.init_vec_type
    layer_idx = args.layer_idx

    # *** 読み込むディレクトリのパスを指定 ***
    global model_name_for_dirname
    model_version = get_gemma_model_version(model_size)

    # [WIP] 'it'と'pt'のどちらが良いかは未検証. とりあえず'it'で統一.
    model_name = f"google/gemma-{model_version}-{model_size}b-it" # [memo] 'gemma-'部分は変えないこと!! -を消すとモデルがloadできない．さらにそのエラーメッセージは，"huggingface-cli login"をして，という関係ないmessageになるので注意!
    model_name_for_dirname = f"gemma-{model_version}-{model_size}B-lr{lr}-{trained_date}"
    if layer_idx is not None:
        model_name_for_dirname += f"-hidden_layer{layer_idx}"
    model_name_for_dirname += f"-seed{seed}"

    # mem_dir = os.path.join(project_root, "memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")
    # "work04"が大規模データ保存用のストレージ
    mem_dir = os.path.join("/work04/toko/EmbedNewConcept-20260305/memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")

    output_dir = os.path.join(project_root, "src_visualize", "output", f"trajectory_pca_plots_{args.pca_components}d")
    os.makedirs(output_dir, exist_ok=True)


    # *** 読み込み ***
    tracked_vectors_path = os.path.join(mem_dir, "tracked_embeddings.npz")
    data = np.load(tracked_vectors_path)

    token_assignment_path = os.path.join(mem_dir, "token_assignment.json")
    with open(token_assignment_path, 'r') as f:
        conceptForFict2token_map = json.load(f)

    
    # *** データの内容を確認 ***
    token_ids = data["token_ids"]
    steps = data["steps"]
    vectors = data["vectors"]

    num_tokens, num_steps, hidden_dim = vectors.shape
    if num_steps != len(steps):
        steps = np.insert(steps, 0, 0) # 学習前の埋め込みの状態をdata["vectors"]の冒頭に追加しているが、2026/04/27学習時にはstepsに0を追加し忘れていたため入っていない。その場合は最初に 0 を追加する。これで、stepsは [0, step1, step2, ...] となる。
    print(vectors.shape)
    print(f"num_tokens: {num_tokens}, num_steps: {num_steps}, hidden_dim: {hidden_dim}")

    # *** PCAで次元削減してプロット ***
    Y = pca_trajectory(vectors, n_components=args.pca_components)  # shape: [num_tokens, num_steps, args.pca_components]

    plot_dic = {
        "markersize": 1,
        "linewidth": 1,
        "alpha": 0.8,
        "color": {token_id: plt.cm.tab20.colors[i % 20] for i, token_id in enumerate(token_ids)}, # token_idごとに異なる色を割り当てる
        "title": f"Trajectory of embedding vectors during training (model size={model_size}, seed={seed}, init_vec_type={init_vec_type}, layer_idx={layer_idx})",
    }

    print(len(plot_dic['color']), plot_dic['color'])

    output_path = os.path.join(output_dir, f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")
    if args.pca_components == 2:
        output_path += ".png"
        plot_trajectory_2d(Y, token_ids, steps, conceptForFict2token_map, output_path, plot_dic)
    elif args.pca_components == 3:
        output_path += ".html"
        plot_trajectory_3d(Y, token_ids, steps, conceptForFict2token_map, output_path, plot_dic)
    else:
        raise ValueError(f"Invalid pca_components: {args.pca_components}. Must be 2 or 3.")


    print(f"Trajectory plot is saved to: {output_path}")


def pca_trajectory(vectors, n_components):

    num_tokens, num_steps, hidden_dim = vectors.shape

    X = vectors.reshape(num_tokens * num_steps, hidden_dim)     # 全部のベクトルを1列に並べる
    pca = PCA(n_components=n_components)                        # 各ベクトルがn_components次元座標になった
    Y = pca.fit_transform(X)                                    # 元の構造(num_tokens, num_steps, n_components)に戻す
    Y = Y.reshape(num_tokens, num_steps, n_components)
    return Y


def plot_trajectory_2d(vectors_2d, token_ids, steps, conceptForFict2token_map, output_path, plot_dic):
    plt.figure(figsize=(10, 8))

    for i, token_id in enumerate(token_ids):
        traj = vectors_2d[i]  # shape: [num_steps, 2]

        # * もしそのtoken_idに埋め込んだ架空の概念の元の概念名を知りたければ、tokenizerを使って、conceptForFict2token_mapから探す
        # for concept, assigned_token in conceptForFict2token_map.items():
        #     assigned_token_id = tokenizer.convert_tokens_to_ids(assigned_token)
        #     if assigned_token_id == token_id:
        #         break

        plt.plot(
            traj[:, 0], traj[:, 1], 
            marker='o', 
            markersize=plot_dic["markersize"], 
            linewidth=plot_dic["linewidth"], 
            alpha=plot_dic["alpha"],
            label=f"token {token_id}",   # f"token {token_id} (originally {concept})")
            color=plot_dic["color"][token_id]
        )

        # step番号も点の近くに表示するなら
        # for j, step in enumerate(steps):
        #     plt.text(traj[j, 0], traj[j, 1], str(step), fontsize=8)
    
        # 始点と終点
        plt.scatter(traj[0, 0], traj[0, 1], marker="s", s=plot_dic["markersize"]**2)
        plt.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=plot_dic["markersize"]**2)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(plot_dic["title"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path)
    

def rgb_to_str(c):
    return f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"

def plot_trajectory_3d(vectors_3d, token_ids, steps, conceptForFict2token_map, output_path, plot_dic):
    # =========================
    # DataFrame化
    # =========================
    token_labels = np.repeat([f"token {t}" for t in token_ids], len(steps))
    print("PC1:", vectors_3d[:, :, 0].reshape(-1).shape)
    df = pd.DataFrame({
        "PC1": vectors_3d[:, :, 0].reshape(-1), # num_tokens, num_steps, 3 -> num_tokens*num_steps, 3 -> num_tokens*num_steps
        "PC2": vectors_3d[:, :, 1].reshape(-1),
        "PC3": vectors_3d[:, :, 2].reshape(-1),
        "token": np.repeat([f"token {t}" for t in token_ids], len(steps)),
        "step": np.tile(steps, len(token_ids)),
        # "color": np.repeat(
        #             [plot_dic["color"][t] for t in token_ids], 
        #             len(steps),
        #             axis=0  # colorの一要素はrgbの3要素タプルなので、axisを指定しないとrgbの各値も複製されてしまう
        #         ),
            
        # "category": valid_categories,
    })
    color_map = {
        f"token {t}": rgb_to_str(plot_dic["color"][t])
        for t in token_ids
    }


    # =========================
    # Plot
    # =========================
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        # color="category",
        color="token",
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


    task_id = -1
    for seed in args.seeds:

        for init_vec_type in args.init_vec_types:

            layer_indices = args.layer_indices                                          # init_vec_typeによってlayer_indices=[None]になることもあるので、init_vec_typeループ毎に取得し直す
            need_layer_flag = 'HS' in init_vec_type or 'HiddenState' in init_vec_type   # 初期化方法名に'隠れ層'が含まれれば、layer_idxの指定が必要な初期化方法とみなす
            if len(layer_indices) < 1 or not need_layer_flag:
                # layer_idxが不要の初期化方法の場合は、layer_indicesを[None]にして、1回だけループするようにする
                layer_indices = [None]


            for layer_idx in layer_indices:
                task_id += 1

                print(f"\n\n=== Training with seed: {seed}, init_vec_type: {init_vec_type}, layer_idx: {layer_idx} ===")
                args.seed = seed
                args.init_vec_type = str(init_vec_type)
                args.layer_idx = layer_idx

                ## [memo] 今後複数プロセスを並列したくなった時用
                # if int(task_id % processNum) != int(args.thread_id):
                #     # 複数process同時に実行する場合, thread_idに応じてtask_idが偶数or奇数の設定のみを実行する
                #     print(f"Skipping task_id {task_id} for thread_id {args.thread_id}")
                #     continue

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
INIT_VEC_TYPES=("CatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "norm_rand_vocab" "zero")
INIT_VEC_TYPES=("CatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed" "nearCatCent_by_WikiSummaryRepeatHSMixed")
PCA_COMPONENTS=2

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