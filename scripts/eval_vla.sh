#!/bin/bash

set -e

echo "========================================="
echo "MiniMind-VLA Inference Script"
echo "========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/out/vla/final}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-$PROJECT_ROOT/dataset/eb_alfred_env_anchored.json}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_ROOT/model}"
export VISION_MODEL_PATH="${VISION_MODEL_PATH:-$PROJECT_ROOT/model/siglip2-base-p32-256-ve}"
export OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/eval_results.json}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
USE_MOE="${USE_MOE:-0}"

echo "Configuration:"
echo "  CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "  EVAL_DATA_PATH: $EVAL_DATA_PATH"
echo "  TOKENIZER_PATH: $TOKENIZER_PATH"
echo "  VISION_MODEL_PATH: $VISION_MODEL_PATH"
echo "  OUTPUT_PATH: $OUTPUT_PATH"
echo "  MAX_SEQ_LEN: $MAX_SEQ_LEN"
echo "  USE_MOE: $USE_MOE"

CUDA_VISIBLE_DEVICES=0 python eval_vla.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --vision_model_path "$VISION_MODEL_PATH" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_path "$OUTPUT_PATH" \
    --use_moe "$USE_MOE"

echo "Inference complete!"
