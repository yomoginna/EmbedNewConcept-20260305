
# ⭐️ source script/train_gemma.sh
# zao: nvidia-smi で空きgpuチェック

# ** 4B **
# 44713MiB 使用 (batch-size 32の時)
# 全部一気に実行する場合(夜間実行用)
MAX_EPOCHS=50
TARGET_CONCEPTS_FILENAME="target_concepts_20260117.json"
MODEL_SIZE=4
LR=0.01
PROCESS_NUM=4

THREAD_ID=0
CUDA_VISIBLE_DEVICES=0

THREAD_ID=1
CUDA_VISIBLE_DEVICES=1

THREAD_ID=2
CUDA_VISIBLE_DEVICES=0

THREAD_ID=3
CUDA_VISIBLE_DEVICES=1


nohup uv --no-progress run python src/trainMemVec_fromXvec_gemma_wholeRun.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --max_epochs ${MAX_EPOCHS} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --thread_id ${THREAD_ID} \
        --process_num ${PROCESS_NUM} \
        > log_TrainMemVec_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}_2.log 2>&1 &

THREAD_ID=0: 1247936
THREAD_ID=1: 1248814
THREAD_ID=2: 1249846
THREAD_ID=3: 1250651



# ** 12B **
# 全部一気に実行する場合(夜間実行用)
# batchsize 4の時 64401MiB 使用
# MAX_EPOCHS=10
# TARGET_CONCEPTS_FILENAME="target_concepts.json"
# MODEL_SIZE=12
# LR=0.01
# PROCESS_NUM=2
# SEED_NUM=2 # 10

MAX_EPOCHS=5
TARGET_CONCEPTS_FILENAME="target_concepts.json"
MODEL_SIZE=12
LR=0.01
PROCESS_NUM=2
SEED_NUM=1 

THREAD_ID=0
CUDA_VISIBLE_DEVICES=0

THREAD_ID=1
CUDA_VISIBLE_DEVICES=1

# THREAD_ID=2
# CUDA_VISIBLE_DEVICES=3

# THREAD_ID=3
# CUDA_VISIBLE_DEVICES=4


nohup uv --no-progress run python src/trainMemVec_fromXvec_gemma_wholeRun_oneTrainOneToken.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --max_epochs ${MAX_EPOCHS} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --thread_id ${THREAD_ID} \
        --process_num ${PROCESS_NUM} \
        --seed_num ${SEED_NUM} \
        > log_TrainMemVec_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}.log 2>&1 &

THREAD_ID=0: 3813873
THREAD_ID=1: -
THREAD_ID=2: -
THREAD_ID=3: -




# ** 9B **
# 75942MiB 以下の使用 (batch-size 16の時)
# 全部一気に実行する場合(夜間実行用)
MAX_EPOCHS=50
TARGET_CONCEPTS_FILENAME="target_concepts_20260117.json"
MODEL_SIZE=9
LR=0.01

PROCESS_NUM=2

THREAD_ID=0
CUDA_VISIBLE_DEVICES=3

THREAD_ID=1
CUDA_VISIBLE_DEVICES=4

# THREAD_ID=2
# CUDA_VISIBLE_DEVICES=3

# THREAD_ID=3
# CUDA_VISIBLE_DEVICES=1

nohup uv --no-progress run python src/trainMemVec_fromXvec_gemma_wholeRun.py \
        --target_concepts_filename ${TARGET_CONCEPTS_FILENAME} \
        --model_size ${MODEL_SIZE} \
        --lr ${LR} \
        --max_epochs ${MAX_EPOCHS} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --thread_id ${THREAD_ID} \
        > log_TrainMemVec_gemma-${MODEL_SIZE}B_lr${LR}_wholeRun${THREAD_ID}.log 2>&1 &

# THREAD_ID=0: 
# THREAD_ID=1: 
# THREAD_ID=2: -