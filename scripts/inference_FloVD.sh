SVD_PATH=path_to_SVD # checkpoints/svd

MODEL_CONFIG=path_to_FVSM_config
FVSM_CKPT=path_to_FVSM_ckpt

OMSM_MODEL_CONFIG=path_to_OMSM_config
OMSM_UNET_CKPT_PATH=path_to_OMSM_ckpt
SAM_CKPT_PATH=path_to_Grounded_SAM2_ckpt # checkpoints/sam2/sam2.1_hiera_base_plus.pt

N_PROCS=1 # the number of GPUs

# TRAJECTORY_LIST="['assets/manual_poses/test_camera_L.txt','assets/manual_poses/test_camera_R.txt','assets/manual_poses/test_camera_I.txt','assets/manual_poses/test_camera_O.txt','assets/manual_poses/test_camera_U.txt','assets/manual_poses/test_camera_D.txt','assets/manual_poses/test_camera_Stop.txt']"
TRAJECTORY_LIST="['assets/manual_poses/test_camera_L.txt']"
RANDOM_PORT=$((49152 + RANDOM % 16384))
SPEED=0.25

#--------------------------------------------------------------------------------------------------------------------------------

DATA_PATH=path_to_evaluation_dataset
OUTPUT_PATH="./results/"

python -m torch.distributed.launch --nproc_per_node=${N_PROCS} --master_port=${RANDOM_PORT} inference.py \
      --out_root "${OUTPUT_PATH}" \
      --num_frames 14 \
      --ori_model_path $SVD_PATH \
      --pose_adaptor_ckpt $FVSM_CKPT \
      --model_config $MODEL_CONFIG \
      --data_path "${DATA_PATH}" \
      --trajectory_file_list $TRAJECTORY_LIST \
      --OMSM_unet_ckpt_path $OMSM_UNET_CKPT_PATH \
      --OMSM_model_config $OMSM_MODEL_CONFIG \
      --sam_ckpt_path $SAM_CKPT_PATH \
      --speed ${SPEED} \
