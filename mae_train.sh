RUN_OUTPUT_DIR="./output/output_mae_pretrain"
# Create checkpoints and logs directories for this experiment inside the specific run folder
CHECKPOINTS_DIR="${RUN_OUTPUT_DIR}/checkpoints"
LOGS_DIR="${RUN_OUTPUT_DIR}/logs"
mkdir -p $CHECKPOINTS_DIR
mkdir -p $LOGS_DIR

export CUDA_VISIBLE_DEVICES=0
python main_pretrain.py \
    --model mae_diet_tiny_patch4 \
    --input_size 32 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --data_path ./data \
    --batch_size 512 \
    --epochs 200 \
    --accum_iter 1 \
    --num_workers 16 \
    --pin_mem \
    --weight_decay 0.05 \
    --blr 0.0001 \
    --min_lr 0.0 \
    --warmup_epochs 40 \
    --device cuda \
    --seed 0 \
    --output_dir $CHECKPOINTS_DIR \
    --log_dir $LOGS_DIR
