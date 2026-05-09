#!/bin/bash

set -e

echo "========================================="
echo "MiniMind-VLA Training Script"
echo "========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_ROOT/model}"
export VISION_MODEL_PATH="${VISION_MODEL_PATH:-$PROJECT_ROOT/model/siglip2-base-p32-256-ve}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/out/vla}"
export DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/dataset/eb_alfred_env_anchored.json}"

LEARNING_RATE="${LEARNING_RATE:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
USE_MOE="${USE_MOE:-0}"
DEEPSPEED="${DEEPSPEED:-0}"

echo "Configuration:"
echo "  TOKENIZER_PATH: $TOKENIZER_PATH"
echo "  VISION_MODEL_PATH: $VISION_MODEL_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  DATA_PATH: $DATA_PATH"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  EPOCHS: $EPOCHS"
echo "  MAX_SEQ_LEN: $MAX_SEQ_LEN"
echo "  USE_MOE: $USE_MOE"
echo "  DEEPSPEED: $DEEPSPEED"

mkdir -p "$OUTPUT_DIR"

if [ "$DEEPSPEED" = "1" ]; then
    echo "Launching with DeepSpeed..."
    torchrun \
        --master_port 29500 \
        --nproc_per_node 8 \
        --nnodes 1 \
        trainer/train_vla.py \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --vision_model_path "$VISION_MODEL_PATH" \
        --learning_rate "$LEARNING_RATE" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --use_moe "$USE_MOE" \
        --deepspeed \
        --use_tb
else
    echo "Launching without DeepSpeed (single GPU)..."
    CUDA_VISIBLE_DEVICES=0 python trainer/train_vla.py \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --vision_model_path "$VISION_MODEL_PATH" \
        --learning_rate "$LEARNING_RATE" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --use_moe "$USE_MOE" \
        --use_tb
fi

echo "Training complete!"
