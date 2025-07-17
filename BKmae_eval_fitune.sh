# 设置默认参数
export CUDA_VISIBLE_DEVICES=1 # 使用单GPU
DATA_PATH="./data"
PRETRAINED="./output/output_bkmae_pretrain/stage_4/checkpoint-49.pth"

OUTPUT_DIR="./output/output_bkmae_finetune_dir"
# Create checkpoints and logs directories for this experiment inside the specific run folder
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/logs"


python main_finetune.py \
    --finetune "${PRETRAINED}" \
    --config ./configs/finetune_config.yaml \
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
