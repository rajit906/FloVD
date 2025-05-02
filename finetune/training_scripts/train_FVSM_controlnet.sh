#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5b-I2V"
    --model_name "cogvideox-flovd"  # ["cogvideox-i2v" or "cogvideox-flovd"]
    --model_type "i2vFlow" # ["t2v", "i2v", "i2vFlow"]
    --training_type "controlnet"
    # --additional_save_blocks "patch_embed" # additional blocks to update when using lora. e.g., "patch_embed,text_proj"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "absolute/path/to/output"
    --report_to "wandb"
    --run_name "FloVD_CogVideoX_controlnet"
)

# Data Configuration
DATA_ARGS=(
    --data_root "absolute/path/to/whole_data"
    --caption_column "prompt.txt" # Do not need
    --video_column "videos.txt" # Do not need
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 2
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    --learning_rate 1e-5
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
    # --resume_from_checkpoint /path/to/ckpt # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "absolute/path/to/whole_data"
    --validation_steps 2000  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt" # Do not need
    --validation_images "images.txt" # Do not need
    --gen_fps 16
    --max_scene 4
)

# Controlnet Configuration
CONTROLNET_ARGS=(
    --controlnet_transformer_num_layers 6
    --controlnet_input_channels 16
    --controlnet_weights 1.0
    --controlnet_guidance_start 0.0
    --controlnet_guidance_end 0.4
    --controlnet_out_proj_dim_factor 64
    --enable_time_sampling false
    --time_sampling_type "truncated_normal"
    --time_sampling_mean 0.95
    --time_sampling_std 0.1
    --notextinflow true
)


# Combine all arguments and launch training
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${CONTROLNET_ARGS[@]}"
