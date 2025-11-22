# Data Location(preprocessed)
data_dir="transformer/data"

# Model
d_model=512
num_heads=8
num_encoder_layers=6
num_decoder_layers=6
d_ff=2048
dropout=0.1

# Training
batch_size=64
num_epochs=2
warmup_steps=4000
label_smoothing=0.1
max_len=100 # fixed
num_workers=0 # mps

# Logging and Checkpoint
log_dir="transformer/log"
checkpoint_dir="transformer/checkpoints"
experiment_name="Transformer-WMT14-EN-DE"

# Hardware
accelerator="mps" # cpu, mps
precision=32
gradient_clip_val=1.0


# Baseline
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

