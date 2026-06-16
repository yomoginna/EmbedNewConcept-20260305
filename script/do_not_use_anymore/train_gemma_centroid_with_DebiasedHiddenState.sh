
# ⭐️ source script/train_gemma.sh
# zao: nvidia-smi で空きgpuチェック

# ** 4B **
# 44713MiB 使用 (batch-size 32の時)
# 全部一気に実行する場合(夜間実行用)
MODEL_SIZE=4


# ** 12B **
# 全部一気に実行する場合(夜間実行用)
# batchsize 4の時 93337MiB 使用
# MAX_EPOCHS=10
# TARGET_CONCEPTS_FILENAME="target_concepts.json"
# MODEL_SIZE=12
# LR=0.01
# PROCESS_NUM=2
# SEED_NUM=2 # 10

MODEL_SIZE=12
MAX_EPOCHS=4
TARGET_CONCEPTS_FILENAME="target_concepts_mini.json"
LR=0.01
PROCESS_NUM=3
SEED_NUM=1 
INIT_VEC_TYPES=("category_centroid_by_debiased_hidden_state_mean" "other_category_centroid_by_debiased_hidden_state_mean" "norm_rand_vocab")
LAYER_INDICES=(9 10 11 12)
# -1は最終層、0以上の整数はその層の隠れ状態を使用.(0層は埋め込み層の出力) 12B: 48層
# 全体の層を大まかに調べる: (0 1 8 12 24 36 40 -1)


THREAD_ID=0
CUDA_VISIBLE_DEVICES=0

THREAD_ID=1
CUDA_VISIBLE_DEVICES=1

THREAD_ID=2
CUDA_VISIBLE_DEVICES=3

# THREAD_ID=3
# CUDA_VISIBLE_DEVICES=4


nohup uv --no-progress run python src/trainMemVec_fromXvec_gemma_wholeRun.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --max_epochs ${MAX_EPOCHS} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --init_vec_types ${INIT_VEC_TYPES[@]} \
        --layer_indices ${LAYER_INDICES[@]} \
        --thread_id ${THREAD_ID} \
        --process_num ${PROCESS_NUM} \
        --seed_num ${SEED_NUM} \
        > log_TrainMemVec_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}.log 2>&1 &

THREAD_ID=0: 1383576
THREAD_ID=1: 1384718
THREAD_ID=2: 1385682
THREAD_ID=3: -




# ** 9B **
# 75942MiB 以下の使用 (batch-size 16の時)
# 全部一気に実行する場合(夜間実行用)


for d in gemma-3-12B-lr0.01-mini-20260319*; do
  [ -d "$d" ] || continue
  mv -- "$d" "${d/#gemma-3-12B-lr0.01-mini-20260319/gemma-3-12B-lr0.01-20260319}"
done