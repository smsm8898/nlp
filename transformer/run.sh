#!/bin/bash

# Transformer Training Script with Model Size Presets
# Usage ./run.sh [tiny|small|base|big]

# Default to tiny if no argument
MODEL_SIZE=${1:-tiny}


# Data Location(preprocessed)
data_dir="transformer/data"

# Common Settings
label_smoothing=0.1
max_len=100
num_workers=0
accelerator="mps"
gradient_clip_val=1.0

# Logging and Checkpoint
log_dir="transformer/logs"
checkpoint_dir="transformer/checkpoints"

# Model Size Configurations
case $MODEL_SIZE in
    tiny)
        echo "Training TINY model"
        d_model=128
        num_heads=4
        num_encoder_layers=3
        num_decoder_layers=3
        d_ff=512
        dropout=0.1
        batch_size=128
        num_epochs=15
        warmup_steps=1000
        precision="16-mixed"
        experiment_name="Transformer-Tiny-WMT14"
        ;;
    
    small)
        echo "Training SMALL model"
        d_model=256
        num_heads=8
        num_encoder_layers=4
        num_decoder_layers=4
        d_ff=1024
        dropout=0.1
        batch_size=96
        num_epochs=20
        warmup_steps=2000
        precision="16-mixed"
        experiment_name="Transformer-Small-WMT14"
        ;;

    base)
        echo "Training BASE model"
        d_model=512
        num_heads=8
        num_encoder_layers=6
        num_decoder_layers=6
        d_ff=2048
        dropout=0.1
        batch_size=64
        num_epochs=20
        warmup_steps=4000
        precision="16-mixed"  # Still use mixed precision for speed
        experiment_name="Transformer-Base-WMT14"
        ;;

    big)
        echo "Training BIG model"
        d_model=1024
        num_heads=16
        num_encoder_layers=6
        num_decoder_layers=6
        d_ff=4096
        dropout=0.3  # Higher dropout for big model
        batch_size=32  # Smaller batch due to memory
        num_epochs=20
        warmup_steps=4000
        precision="16-mixed"
        experiment_name="Transformer-Big-WMT14"
        ;;
    *)
        echo "❌ Invalid model size: $MODEL_SIZE"
        echo "Usage: $0 [tiny|small|base|big]"
        echo ""
        echo "Available options:"
        echo "  tiny  - Tiny (12-18h), 4M params, BLEU ~20-23"
        echo "  small - Small (1.5-2d), 16M params, BLEU ~24-26"
        echo "  base  - Paper baseline (5-8d), 65M params, BLEU ~27-28"
        echo "  big   - Paper big (10-14d), 213M params, BLEU ~28-29"
        exit 1
        ;;
esac

python -m transformer.train \
    --data_dir $data_dir \
    --d_model $d_model \
    --num_heads $num_heads \
    --num_encoder_layers $num_encoder_layers \
    --num_decoder_layers $num_decoder_layers \
    --d_ff $d_ff \
    --dropout $dropout \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --warmup_steps $warmup_steps \
    --label_smoothing $label_smoothing \
    --max_len $max_len \
    --num_workers $num_workers \
    --log_dir $log_dir \
    --checkpoint_dir $checkpoint_dir \
    --experiment_name $experiment_name \
    --accelerator $accelerator \
    --precision $precision \
    --gradient_clip_val $gradient_clip_val