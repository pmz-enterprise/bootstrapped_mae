#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p output_dir/mae

# 训练MAE模型
python main_pretrain.py \
    --model mae_diet_tiny_patch4 \
    --batch_size 256 \
    --epochs 200 \
    --accum_iter 1 \
    --input_size 32 \
    --mask_ratio 0.75 \
    --data_path ./data/ \
    --output_dir ./output_dir/mae \
    --log_dir ./output_dir/mae \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 40 \
    --num_workers 4 