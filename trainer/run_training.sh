#!/bin/bash
# MiniMind-VLA Training Script - Optimized for 8x H20 GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Environment
source /opt/conda/etc/profile.d/conda.sh transf457

# Paths
export TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_ROOT/model}"
export VISION_MODEL_PATH="${VISION_MODEL_PATH:-$PROJECT_ROOT/model/siglip2-base-p32-256-ve}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/out/vla}"
export DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/dataset/robot_data/train_trajectories.json}"

# Hyperparameters - optimized for small VLA on H20
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
MAX_HISTORY_STEPS="${MAX_HISTORY_STEPS:-10}"
USE_MOE="${USE_MOE:-0}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

echo "========================================="
echo "MiniMind-VLA Training"
echo "========================================="
echo "Configuration:"
echo "  TOKENIZER_PATH: $TOKENIZER_PATH"
echo "  VISION_MODEL_PATH: $VISION_MODEL_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  DATA_PATH: $DATA_PATH"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  EPOCHS: $EPOCHS"
echo "  MAX_SEQ_LEN: $MAX_SEQ_LEN"
echo "  MAX_HISTORY_STEPS: $MAX_HISTORY_STEPS"
echo "  USE_MOE: $USE_MOE"
echo "========================================="

mkdir -p "$OUTPUT_DIR"

# Single GPU training (for now)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --master_port 29500 \
    --nproc_per_node 1 \
    trainer/train_vla.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --vision_model_path "$VISION_MODEL_PATH" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --max_history_steps "$MAX_HISTORY_STEPS" \
    --use_moe "$USE_MOE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --weight_decay "$WEIGHT_DECAY" \
    --use_tb \
    2>&1 | tee "$OUTPUT_DIR/training_log.txt"

echo "Training complete!"
echo "Checkpoint saved to: $OUTPUT_DIR/final"