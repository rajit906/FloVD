#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5b-I2V"
    --model_name "cogvideox-flovd-omsm"  # ["cogvideox-i2v"]
    --model_type "i2vFlow" # ["t2v", "i2v", "i2vFlow"]
    --training_type "lora"
    --additional_save_blocks "patch_embed" # additional blocks to update when using lora. e.g., "patch_embed,text_proj"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "absolute/path/to/output"
    --report_to "wandb"
    --run_name "CogVideoX_OMSM"
)

# Data Configuration
DATA_ARGS=(
    --data_root "absolute/path/to/whole_data"
    --caption_column "prompt.txt" # Do not need
    --video_column "videos.txt" # Do not need
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
)

# Lora Configuration
LORA_ARGS=(
    --target_modules "to_q" "to_k" "to_v" "to_out.0" "norm1.linear" "norm2.linear" "ff.net.2"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 2
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 2000 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/path/to/ckpt"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "absolute/path/to/whole_data"
    --validation_steps 2000  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
    --max_scene 4
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
