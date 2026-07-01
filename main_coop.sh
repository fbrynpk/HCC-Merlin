#!/bin/bash

temperatures=0.07
tunings="full"
lr=1e-5
n_ctx=(2 4 8 16)

CHECKPOINT_DIR="/media/ryan/TOSHIBA2/checkpoint_merlin"

# Loop n_ctx values
for ctx in "${n_ctx[@]}"; do
    echo "========================================="
    echo "Running with n_ctx = $ctx"
    echo "========================================="
    SAVE_PATH="${CHECKPOINT_DIR}/Interpolate_HCC_Negative_GradAccumulation8_Venous_50Shot_CoCoOp_${ctx}CTX_lr1e-5_bs18_temp007_epoch50_lldr_512_nowarmup.pth"

    python train.py \
        --phase "venous" \
        --epochs 50 \
        --batch_size 18 \
        --tuning_mode "full" \
        --learning_rate "$lr" \
        --temperature 0.07 \
        --model_save_path "$SAVE_PATH" \
        --evaluate \
        --use_wandb \
        --use_coop \
        --n_ctx "$ctx"
done