AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-1.5}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
# New tensor parallelism size for the resumed training run
TP="${1:-2}"
PP="${2:-1}"
SEQ_LEN="${3:-4096}"
MBS="${4:-1}"
# Adjust GBS so that GBS / (TP * PP) matches the original DP degree.
# Example: original DP=8 TP=1 GBS=64 => new DP=4 TP=2 GBS=32
GBS="${5:-32}"
NSTEP="${6:-40}"
DATA_PATH=${DATA_PATH:-"/workspace/dataset/LLaVA-NeXT-780k-webdataset"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-stage0"}
# Path to the original torch_dist checkpoint (e.g. saved with TP=1)
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/LLaVA-OneVision-1.5/original_tp1_checkpoint"}
# Optional: load from a specific iteration instead of the latest one.
# Leave empty to use the latest checkpoint (default behaviour).
CKPT_STEP=${CKPT_STEP:-""}

#! /bin/bash
# Resume training with a different tensor-parallel degree from a torch_dist checkpoint.
#
# Typical use case: an earlier run used {DP,TP,PP}={8,1,1} and saved checkpoints in
# torch_dist format.  This script resumes that run with {DP,TP,PP}={4,2,1}, which
# requires resharding the model and optimizer state across the new TP ranks.
#
# Prerequisites
# -------------
# * The original checkpoint must have been saved with --ckpt-format torch_dist.
#   (torch_dist / fully_sharded_model_space is the only format that supports
#   resharding of the distributed optimizer when TP/PP changes.)
# * The new GBS must be divisible by (TP * PP * num_nodes * GPUS_PER_NODE).
#
# The script needs to be run on at least 1 node.

# --- Multi-node configuration ---
# List of IP addresses for the nodes in the training cluster
declare -a list_ip=(
    "localhost"
)

# Get the primary IP of the current node
CURRENT_IP=$(hostname -I | awk '{print $1}')

if [ -z "$CURRENT_IP" ]; then
    CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

# Dynamically determine NNODES, MASTER_ADDR
NNODES=${#list_ip[@]}
MASTER_ADDR=${list_ip[0]}

if [[ $SINGLE_NODE -eq 1 ]]; then
    NNODES=1
    MASTER_ADDR=127.0.0.1
    NODE_RANK=0
    echo "--- Single-node mode ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
else
    # Find the rank of the current node
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        if [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done

    # Exit if the current IP is not in the list
    if [ "$NODE_RANK" -eq -1 ]; then
        echo "Error: Current IP ($CURRENT_IP) not found in the IP list."
        echo "Please run this script on a node with an IP in list_ip."
        exit 1
    fi

    echo "--- Running on ${NNODES} nodes ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
fi
# --- End of Multi-node configuration ---


SAVE_CKPT_PATH=$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"${list_ip[0]}"}
MASTER_PORT=${MASTER_PORT:-"26000"}

if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
    )
else
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
fi

MODEL_ARGS=(
    --model-name llava-ov-1.5-4b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 1
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --image-resolution 60000
    --min-pixels 3136
    --max-pixels 12845056
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 3.0e-6
    --min-lr 2.0e-7
    --clip-grad 0.5
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    --initial-loss-scale 65536
    --bf16
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 20
    # Save in torch_dist format so that future TP/PP changes remain possible.
    --ckpt-format torch_dist
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
    # Auto-detect the format of the checkpoint being loaded (handles both torch and torch_dist).
    --auto-detect-ckpt-format
    # Enable parallel loading across DP ranks for faster checkpoint loading.
    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

# Optional: load from a specific checkpoint step instead of the latest one.
# This is useful when the checkpoint directory contains multiple saved steps and
# you want to resume from a particular one (e.g. mid-training checkpoint).
if [[ -n "$CKPT_STEP" ]]; then
    TRAINING_ARGS+=(--ckpt-step "$CKPT_STEP")
fi

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export OFFLINE_PACKED_DATA='0'
export OFFLINE_PACKING_VQA='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72

PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
