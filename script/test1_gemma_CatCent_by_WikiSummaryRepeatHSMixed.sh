# ⭐️ source script/test1_gemma.sh
# nohupで実行中のjobは pgrep -a uv で確認可能



# ** 4B **
# 17933MiB 使用
MODEL_SIZE=4
LR=0.01
TARGET_CONCEPTS_FILENAME="target_concepts.json"
PROCESS_NUM=2
SEED_NUM=1
NUM_OPTIONS=3





# ** 12B **=
# * 12B, wholeRunで全通りを一気に実行 *
# processNumを対応する数に変更する必要あり
# 利用メモリ: 24415MiB
MODEL_SIZE=12
LR=0.003
TARGET_CONCEPTS_FILENAME="target_concepts_mini_13.json"
PROCESS_NUM=3
SEED_NUM=20
NUM_OPTIONS=3
LAYER_INDICES=(12 32) # 12 40)

INIT_VEC_TYPES=("CatCent_by_WikiSummRepeatHSMix_noRand" "otherCatCent_by_WikiSummRepeatHSMix_noRand")
INIT_VEC_TYPES=("CatCent_by_WikiSummaryRepeatHSMixed" "otherCatCent_by_WikiSummaryRepeatHSMixed")
INIT_VEC_TYPES=("otherCatCent_by_WikiSummaryRepeatHSMixed")
# LAYER_INDICES=(1 4 8 10 12 16 20 24 28 32 36 38 40 44 -1)
# INIT_VEC_TYPES=("CatCent_by_WikiSummaryHS" "otherCatCent_by_WikiSummaryHS" "norm_rand_vocab")
# -1は最終層、0以上の整数はその層の隠れ状態を使用.(0層は埋め込み層の出力) 12B: 48層
# 全体の層を大まかに調べる: (0 1 8 12 24 36 40 -1)

THREAD_ID=0
CUDA_VISIBLE_DEVICES=3

THREAD_ID=1
CUDA_VISIBLE_DEVICES=3

THREAD_ID=2
CUDA_VISIBLE_DEVICES=3

THREAD_ID=3
CUDA_VISIBLE_DEVICES=4

THREAD_ID=4
CUDA_VISIBLE_DEVICES=4

THREAD_ID=5
CUDA_VISIBLE_DEVICES=4


nohup uv --no-progress run python src/test1_gemma_wholeRun.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --thread_id ${THREAD_ID} \
        --process_num ${PROCESS_NUM} \
        --seed_num ${SEED_NUM} \
        --num_options ${NUM_OPTIONS} \
        --init_vec_types ${INIT_VEC_TYPES[@]} \
        --layer_indices ${LAYER_INDICES[@]} \
        > log_Test1_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}_2.log 2>&1 &

# thread0: 2208805, 4epoch以降: -4011226
# thread1: 2209729, 4epoch以降: -(まだ) 
# thread2: 2210630, 4epoch以降: -(まだ)
# thread3: 2211510
# thread4: 2212466
# thread5: 2213489

THREAD_ID=0
CUDA_VISIBLE_DEVICES=2
2261138

THREAD_ID=1
CUDA_VISIBLE_DEVICES=2
2262073

THREAD_ID=2
CUDA_VISIBLE_DEVICES=2
2262978

# * 9B, wholeRunで全通りを一気に実行 *
MODEL_SIZE=9
LR=0.01
TARGET_CONCEPTS_FILENAME="target_concepts_20260117.json"
PROCESS_NUM=4

THREAD_ID=0
CUDA_VISIBLE_DEVICES=3

THREAD_ID=1
CUDA_VISIBLE_DEVICES=4

THREAD_ID=2
CUDA_VISIBLE_DEVICES=3

THREAD_ID=3
CUDA_VISIBLE_DEVICES=4

nohup uv --no-progress run python src/test1_gemma_wholeRun.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --thread_id ${THREAD_ID} \
        --process_num ${PROCESS_NUM} \
        > log_Test1_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}.log 2>&1 &

# thread0: -
# thread1: -
# thread2: -
# thread3: -