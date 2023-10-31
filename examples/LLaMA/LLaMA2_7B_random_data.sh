#!/bin/bash

TP_SIZE=2
PP_SIZE=1
WORLD_SIZE=4
MICRO_BATCH_SIZE=2
# The int is the number of micro steps of gradient accumulation
# GLOBAL_BATCH_SIZE=$(($WORLD_SIZE * $MICRO_BATCH_SIZE))
GLOBAL_BATCH_SIZE=$((($WORLD_SIZE * $MICRO_BATCH_SIZE) / ($TP_SIZE * $PP_SIZE) * 1))
# GLOBAL_BATCH_SIZE=128
LAYERS=32

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
JOB_NAME="megatron_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}_layers${LAYERS}_code_sync_${TIMESTAMP}"

LOAD_CHECKPOINT_PATH="/mnt/vepfs/lcxyc/model/Llama-2-7b-hf/"
SAVE_CHECKPOINT_PATH="/mnt/vepfs/lcxpt/model/Llama-2-7b-mgt/interval"
TOKENIZER_PATH="/mnt/vepfs/lcxyc/model/Llama-2-7b-hf/"
TENSORBOARD_DIR="/mnt/vepfs/lcxpt/tensorboard/llama/colossal/${JOB_NAME}"

TRAIN_ITERS=10
EVAL_ITERS=0
EVAL_INTERVAL=5
SAVE_INTERVAL=0
LOG_INTERVAL=1


# Setting --tensorboard-queue-size to 1 significantly slows down the training
options=" \
    --sequence-parallel \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers ${LAYERS} \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --no-position-embedding \
        --swiglu \
        --ffn-hidden-size 11008\
        --disable-bias-linear \
        --RMSNorm \
        --layernorm-epsilon 1e-6 \
        --causal-lm \
    --init-method-std 0.01 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 6.0e-5 \
        --lr-decay-iters 10 \
        --lr-warmup-iters 5 \
        --min-lr 6.0e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --use-distributed-optimizer \
        --overlapped-distributed-optimizer \
        --reduce-bucket-size=2e8 \
        --no-gradient-accumulation-fusion \
    --dataloader-type single \
        --data-path ${DATASET} \
        --split 98,2,0 \
    --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
    --load ${LOAD_CHECKPOINT_PATH} \
        --no-load-optim \
    --log-interval ${LOG_INTERVAL} \
    --job-name ${JOB_NAME} \
    --recompute-activations \
        --recompute-granularity selective \
    --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --make-vocab-size-divisible-by 1 \
    --bf16 \
    --save-interval ${SAVE_INTERVAL} \
    --save ${SAVE_CHECKPOINT_PATH} \
    --use-flash-attn \
    --tensorboard-dir ${TENSORBOARD_DIR} \
        --tensorboard-queue-size 1000 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
    " \

# changes from LLaMA2_7B_standalone.sh
#  --data-impl mmap \
#  --finetune
#  --no-persist-layer-norm \
# --use-rotary-position-embeddings


CUDA_VISIBLE_DEVICES=3,4,5,7 torchrun --nproc_per_node=$WORLD_SIZE ~/share/Megatron-LLaMA/pretrain_llama.py ${options}
