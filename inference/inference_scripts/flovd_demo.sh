#!/bin/bash

FVSM_PATH="path/to/FVSM/blahblah.pt" # e.g., ./ckpt/FVSM/FloVD_FVSM_Controlnet.pt
OMSM_PATH="path/to/OMSM/directory" # e.g., ./ckpt/OMMSM/
DEPTH_CKPT_PATH="path/to/depth_estimator.pth" # e.g., './ckpt/others/depth_anything_v2_metric_hypersim_vitb.pth'

POSE_TYPE="re10k" # Choose between ['re10k', 'manual']
OUTPUT_PATH="./results/"
CONTROLNET_GUIDANCE_END=0.4
SPEED=1.0

PROMPT="A chef in a white coat and glasses preparing a meal in a professional kitchen. He is seen standing at a counter, surrounded by various ingredients and cookware, including bowls, plates, and utensils. The chef is focused on his task, carefully selecting and arranging the ingredients on the counter. The kitchen is well-equipped with stainless steel appliances, such as a refrigerator, oven, and microwave, and features a ventilation system to ensure a clean and efficient environment. The chef's meticulous preparation process is captured as he carefully selects and organizes the ingredients, indicating a high level of culinary skill and attention to detail."
IMAGE_PATH="./assets/example_image/input1.png"
CAM_POSE_NAME="1593596b99e2dde9.txt"

python inference/flovd_demo.py \
    --prompt "$PROMPT" \
    --image_path $IMAGE_PATH \
    --fvsm_path $FVSM_PATH \
    --omsm_path $OMSM_PATH \
    --output_path $OUTPUT_PATH \
    --controlnet_guidance_end $CONTROLNET_GUIDANCE_END \
    --use_flow_integration \
    --pose_type $POSE_TYPE \
    --speed $SPEED \
    --cam_pose_name $CAM_POSE_NAME \
    --depth_ckpt_path $DEPTH_CKPT_PATH \


PROMPT="A stunning and untouched coastal landscape. It begins with a view of a rugged rock formation emerging from the sea, surrounded by churning waves. As the video progresses, the scene shifts to a solitary rock formation standing in the midst of the sea, with the waves crashing against the shore. The natural beauty of the coastline is highlighted by the absence of any human activity. The video then presents a panoramic view of the coastline, revealing the dynamic interplay between the sea and the land. The scene is characterized by the presence of a solitary rock formation, which stands out against the backdrop of the sea and the overcast sky. The video concludes with a view of the coastline, emphasizing the natural beauty and the absence of any human activity."
IMAGE_PATH="./assets/example_image/input2.png"
CAM_POSE_NAME="6b6d20c6a46b9fe9.txt"

python inference/flovd_demo.py \
    --prompt "$PROMPT" \
    --image_path $IMAGE_PATH \
    --fvsm_path $FVSM_PATH \
    --omsm_path $OMSM_PATH \
    --output_path $OUTPUT_PATH \
    --controlnet_guidance_end $CONTROLNET_GUIDANCE_END \
    --use_flow_integration \
    --pose_type $POSE_TYPE \
    --speed $SPEED \
    --cam_pose_name $CAM_POSE_NAME \
    --depth_ckpt_path $DEPTH_CKPT_PATH \