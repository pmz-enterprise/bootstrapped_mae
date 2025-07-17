#!/bin/bash

# 基础配置
export CUDA_VISIBLE_DEVICES=0  # 使用单GPU
DATA_PATH="./data"
PRETRAINED="./output/output_mae_pretrain/checkpoints/checkpoint-199.pth"
OUTPUT_DIR="./output/output_mae_linprobe_dir"

# 自动创建目录结构
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/logs"

python main_linprobe.py \
    --finetune "${PRETRAINED}" \
    --config "./configs/linprobe_config.yaml" \
    --batch_size 256 \
    --model vit_tiny_patch4 \
    --epochs 100 \
    --accum_iter 1 \
    --blr 0.01 \
    --weight_decay 0.05 \
    --warmup_epochs 20 \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}/checkpoints" \
    --log_dir "${OUTPUT_DIR}/logs" \
    --global_pool \
    --nb_classes 10