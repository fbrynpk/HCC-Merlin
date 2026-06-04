#!/bin/bash

temperatures=(0.07 0.5 1.0 10.0)
tunings=("lora" "full")

# CHECKPOINT_DIR="/media/ryan/T500/checkpoint"
CHECKPOINT_DIR="/media/ryan/TOSHIBA2/checkpoint_merlin"

for tuning in "${tunings[@]}"; do

    if [ "$tuning" = "lora" ]; then
        lr="1e-3"
        model_tag="OfficialConvLoRAa2r2+LoRAa16r32"
    else
        lr="1e-4"
        model_tag="Full"
    fi

    for temp in "${temperatures[@]}"; do

        # Remove decimal for filename style: 0.07 → 007, 0.5 → 05, 1.0 → 10, 10.0 → 100
        temp_tag=$(echo "$temp" | sed 's/\.//g')

        SAVE_PATH="${CHECKPOINT_DIR}/\
Interpolate_HCC_Negative_GradAccumulation8_\
Venous_${model_tag}_\
lr${lr}_\
bs18_\
temp${temp_tag}_\
epoch300_\
noWD_cgmh_final_no_leakage_lldr_512_warmup_valloss.pth"

        echo "========================================="
        echo "Running:"
        echo " tuning = $tuning"
        echo " temp   = $temp"
        echo " lr     = $lr"
        echo " save   = $SAVE_PATH"
        echo "========================================="

        python train.py \
            --phase "venous" \
            --epochs 300 \
            --batch_size 18 \
            --tuning "$tuning" \
            --learning_rate "$lr" \
            --temperature "$temp" \
            --model_save_path "$SAVE_PATH" \
            --evaluate \
            --use_wandb

    done
done