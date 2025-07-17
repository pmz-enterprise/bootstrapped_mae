# The script is used to pretrain the bootstrap MAE model with EMA
# 设置默认参数
RUN_OUTPUT_DIR="./output/output_bootstrap_ema_pretrain"
# Create checkpoints and logs directories for this experiment inside the specific run folder
CHECKPOINTS_DIR="${RUN_OUTPUT_DIR}/checkpoints"
LOGS_DIR="${RUN_OUTPUT_DIR}/logs"
mkdir -p $CHECKPOINTS_DIR
mkdir -p $LOGS_DIR

export CUDA_VISIBLE_DEVICES=2

python main_bmae_pretrain.py \
    --model bmae_ema_tiny_patch4 \
    --input_size 32 \
    --mask_ratio 0.75 \
    --bootstrap_ratio 0.5 \
    --norm_pix_loss \
    --normalize_target \
    --data_path ./data \
    --batch_size 128 \
    --epochs 200 \
    --accum_iter 1 \
    --num_workers 16 \
    --pin_mem \
    --start_epoch 0 \
    --ema_alpha 0.999 \
    --ema_start_epoch 50 \
    --ema_decay_warmup_epochs 10 \
    --weight_decay 0.05 \
    --blr 0.0001 \
    --min_lr 0.0 \
    --warmup_epochs 40 \
    --device cuda \
    --seed 42 \
    --output_dir $CHECKPOINTS_DIR \
    --log_dir $LOGS_DIR


