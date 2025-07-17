#!/bin/bash
#SBATCH --job-name=bmae_k_cifar10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=10
#SBATCH --output=bmae_k_%j.out
#SBATCH --error=bmae_k_%j.err
#SBATCH --partition=your_partition
export CUDA_VISIBLE_DEVICES=1  # 使用单GPU
# --- Configuration ---
K=4 # Total number of bootstrap iterations.
TOTAL_EPOCHS=200 # Total compute budget
EPOCHS_PER_STAGE=$((TOTAL_EPOCHS / K))

# Root directory for this experiment's output (matching your log)
ROOT_OUTPUT_DIR="./output/output_bkmae_pretrain"
mkdir -p $ROOT_OUTPUT_DIR # Ensure root directory exists

# --- Stage 1: Standard MAE (Pixel Reconstruction) ---
STAGE_NUM=1
echo "--- Starting Bootstrap Stage ${STAGE_NUM}/${K} (Pixel Reconstruction) ---"
STAGE_DIR="${ROOT_OUTPUT_DIR}/stage_${STAGE_NUM}"

# # Note: No --teacher_checkpoint_path, so it runs in pixel mode
python main_bootstrap_step.py \
    --model mae_diet_tiny_patch4 \
    --epochs $EPOCHS_PER_STAGE \
    --output_dir $STAGE_DIR \
    --batch_size 512 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --warmup_epochs 10 \
    --data_path ./data

# Path to the checkpoint that will become the teacher for the next stage
TEACHER_CKPT="${STAGE_DIR}/checkpoint-$((EPOCHS_PER_STAGE - 1)).pth"

# --- Stages 2 to K: Bootstrapped MAE (Feature Reconstruction) ---
for STAGE_NUM in $(seq 2 $K)
do
  echo "--- Starting Bootstrap Stage ${STAGE_NUM}/${K} (Feature Reconstruction) ---"
  STAGE_DIR="${ROOT_OUTPUT_DIR}/stage_${STAGE_NUM}"
  
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
      
  # Update the teacher checkpoint for the next iteration
  TEACHER_CKPT="${STAGE_DIR}/checkpoint-$((EPOCHS_PER_STAGE - 1)).pth"
done

echo "--- Bootstrapping complete. Final model is at $TEACHER_CKPT ---"
echo "You can now use this checkpoint for linear probing and fine-tuning."

