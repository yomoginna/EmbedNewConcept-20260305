# ⭐️ source script/test1_gemma.sh
# nohupで実行中のjobは pgrep -a uv で確認可能



# ** 4B **
# 17933MiB 使用
MODEL_SIZE=4
LR=0.01
TARGET_CONCEPTS_FILENAME="target_concept.json"

CUDA_VISIBLE_DEVICES=3
THREAD_ID=1

CUDA_VISIBLE_DEVICES=4
THREAD_ID=0

nohup uv --no-progress run python src/test1_gemma_wholeRun.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --thread_id ${THREAD_ID} \
        > log_Test1_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}.log 2>&1 &

# thread1: 183802
# thread0: 184549





# ** 12B **=
# * 12B, wholeRunで全通りを一気に実行 *
# processNumを対応する数に変更する必要あり
# 利用メモリ: 24415MiB
MODEL_SIZE=12
LR=0.01
TARGET_CONCEPTS_FILENAME="target_concepts.json"
PROCESS_NUM=2
SEED_NUM=1
NUM_OPTIONS=3

THREAD_ID=0
CUDA_VISIBLE_DEVICES=1

THREAD_ID=1
CUDA_VISIBLE_DEVICES=1

# THREAD_ID=2
# CUDA_VISIBLE_DEVICES=4

# THREAD_ID=3
# CUDA_VISIBLE_DEVICES=4

nohup uv --no-progress run python src/test1_gemma_wholeRun_oneTrainOneToken.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --thread_id ${THREAD_ID} \
        --process_num ${PROCESS_NUM} \
        --seed_num ${SEED_NUM} \
        --num_options ${NUM_OPTIONS} \
        > log_Test1_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}.log 2>&1 &

# thread0: 3832858, 4epoch以降: -
# thread1: -, 4epoch以降: -(まだ)
# thread2: -
# thread3: -




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