K=3
TOTAL_EPOCHS=200 # Total compute budget
EPOCHS_PER_STAGE=$((TOTAL_EPOCHS / K))
export CUDA_VISIBLE_DEVICES=1  # 使用单GPU
# Root directory for this experiment's output (matching your log)
ROOT_OUTPUT_DIR="./output/output_bkmae_pretrain"
STAGE_NUM=2
# Update the teacher checkpoint for the next iteration


STAGE_NUM=3
 
STAGE_DIR="./output/output_bkmae_pretrain/stage_3"
TEACHER_CKPT="./output/output_bkmae_pretrain/stage_2/checkpoint-65.pth"
  # Note: Now we provide --teacher_checkpoint_path and --resume
python main_bootstrap_step.py \
      --model bmae_diet_tiny_patch4 \
      --epochs $EPOCHS_PER_STAGE \
      --output_dir $STAGE_DIR \
      --teacher_checkpoint_path $TEACHER_CKPT \
      --resume $TEACHER_CKPT \
      --batch_size 512 \
      --blr 1.5e-4 \
      --weight_decay 0.05 \
      --mask_ratio 0.75 \
      --warmup_epochs 10 \
      --data_path ./data
      
