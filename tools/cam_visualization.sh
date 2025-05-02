#!/bin/bash

#---------------------------------------------------------------------------
# RE10K

TARGET=8135673a5a3e3d17

python tools/visualize_trajectory.py \
    --pose_file_path ./assets/re10k_poses/${TARGET}.txt \
    --relative_c2w \
    --base_xval 0.05 \
    --zval 0.15 \
    --x_min -1.0 \
    --x_max  1.0 \
    --y_min -1.0 \
    --y_max 1.0 \
    --z_min -1.0 \
    --z_max 1.0 \


#---------------------------------------------------------------------------
# Manual poses

TARGET=camera_U

python tools/visualize_trajectory.py \
    --pose_file_path ./assets/manual_poses/${TARGET}.txt \
    --relative_c2w \
    --base_xval 0.05 \
    --zval 0.15 \
    --x_min -1.0 \
    --x_max  1.0 \
    --y_min -1.0 \
    --y_max 1.0 \
    --z_min -1.0 \
    --z_max 1.0 \