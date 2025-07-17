#!/bin/bash

# 基础配置
export CUDA_VISIBLE_DEVICES=1  
DATA_PATH="./data"
PRETRAINED="./output/output_bkmae_pretrain/stage_4/checkpoint-49.pth"
OUTPUT_DIR="./output/output_bkmae_linprobe_dir"

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
    --blr 0.1 \
    --weight_decay 0.05 \
    --warmup_epochs 20 \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}/checkpoints" \
    --log_dir "${OUTPUT_DIR}/logs" \
    --global_pool \
    --nb_classes 10