import argparse
import json
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import ast
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.utils.import_utils import is_xformers_available
from packaging import version as pver
from einops import rearrange, repeat
from glob import glob
from datetime import timedelta

import sys
sys.path.append('./')

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from flovd.pipelines.pipeline_animation_FVSM import StableVideoDiffusionPipelineFVSM
from flovd.pipelines.pipeline_animation_OMSM import StableVideoDiffusionPipelineOMSM

from flovd.models.autoencoder_flow import AutoencoderKLTemporalDecoder_Flow
from flovd.models.unet_EncPoseCond import UNetSpatioTemporalConditionModelEncPoseCond
from flovd.models.pose_adaptor import FlowEncoder
from flovd.models.flow_generator import FlowGenerator, get_flow_generator_input
from flovd.modules.Segmentation.segmentation_wrapper import Segmentation_wrapper
from flovd.utils.util import save_videos_grid

import torchvision

from einops import rearrange, repeat
from torch.utils.data.dataset import Dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30), **kwargs)

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    # https://github.com/pytorch/pytorch/issues/98763
    # torch.cuda.set_device(local_rank)

    return local_rank


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)



def get_relative_pose(cam_params, zero_first_frame_scale):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    if zero_first_frame_scale:
        cam_to_origin = 0
    else:
        cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses



def get_pipeline_FVSM(ori_model_path, unet_subfolder, down_block_types, up_block_types, flow_encoder_kwargs, flow_generator_kwargs,
                      attention_processor_kwargs, pose_adaptor_ckpt, enable_xformers, device, sam_ckpt_path):
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(ori_model_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(ori_model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(ori_model_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(ori_model_path, subfolder="vae")
    
    unet = UNetSpatioTemporalConditionModelEncPoseCond.from_pretrained(ori_model_path,
                                                                        subfolder=unet_subfolder,
                                                                        down_block_types=down_block_types,
                                                                        up_block_types=up_block_types)
    
    
    flow_generator = FlowGenerator(**flow_generator_kwargs)
    flow_encoder = FlowEncoder(**flow_encoder_kwargs)
    print("Setting the attention processors")
    unet.set_pose_cond_attn_processor(enable_xformers=(enable_xformers and is_xformers_available()), **attention_processor_kwargs)
    print(f"Loading weights of camera encoder and attention processor from {pose_adaptor_ckpt}")
    ckpt_dict = torch.load(pose_adaptor_ckpt, map_location=unet.device)
    flow_encoder_state_dict = ckpt_dict['flow_encoder_state_dict']
    flow_encoder_m, flow_encoder_u = flow_encoder.load_state_dict(flow_encoder_state_dict)
    assert len(flow_encoder_m) == 0 and len(flow_encoder_u) == 0
    attention_processor_state_dict = ckpt_dict['attention_processor_state_dict']
    _, attention_processor_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attention_processor_u) == 0
    print("Loading done")
    vae.to(device)
    image_encoder.to(device)
    unet.to(device)
    flow_generator.to(device)
    
    if sam_ckpt_path is not None:
        segmentation_module = Segmentation_wrapper(sam_ckpt_path)
        segmentation_module.to(device)
    else:
        segmentation_module = None
    
    pipeline = StableVideoDiffusionPipelineFVSM(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        flow_encoder=flow_encoder,
        flow_generator=flow_generator,
        segmentation_module=segmentation_module,
        )
    
    pipeline = pipeline.to(device)
    return pipeline


def get_pipeline_OMSM(ori_model_path, unet_subfolder, OMSM_unet_ckpt_path, flow_generator_kwargs, device):
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(ori_model_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(ori_model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(ori_model_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder_Flow.from_pretrained(ori_model_path, subfolder="vae")
    
    unet_flow = UNetSpatioTemporalConditionModel.from_pretrained(ori_model_path,
                                                                 subfolder=unet_subfolder)
    # load checkpoint of unet_flow from OMSM_unet_ckpt_path
    ckpt = torch.load(OMSM_unet_ckpt_path, map_location='cpu')
    unet_flow_sd = ckpt['unet']
    unet_flow_m, unet_flow_u = unet_flow.load_state_dict(unet_flow_sd, strict=False)
    assert len(unet_flow_m) == 0 and len(unet_flow_u) == 0
        
    flow_generator = FlowGenerator(**flow_generator_kwargs)                                                                    
    
    print("Loading done")
    vae.to(device)
    image_encoder.to(device)
    unet_flow.to(device)
    flow_generator.to(device)
    pipeline = StableVideoDiffusionPipelineOMSM(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet_flow,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        flow_generator=flow_generator)
    pipeline = pipeline.to(device)
    return pipeline



class VideoDataset(Dataset):
    def __init__(
        self,
        input_path,
        trajectory_root_path,
        trajectory_file_list,
        sample_size=[320, 576],
        original_size=[720, 1280],
    ):
        self.input_path = input_path
        scene_list = os.listdir(self.input_path)
        self.scene_path_list = sorted([os.path.join(self.input_path, x) for x in scene_list])
        self.length = len(self.scene_path_list)
        
        assert self.length > 0, 'set appropriate path.'
        
        
        camera_dict = {}
        for trajectory_file in trajectory_file_list:
            trajectory_path = os.path.join(trajectory_root_path, trajectory_file)
            
            print(f'Loading K, R, t matrix from {trajectory_path}')
            with open(trajectory_path, 'r') as f:
                poses = f.readlines()
            poses = [pose.strip().split(' ') for pose in poses[1:]]
            cam_params = [[float(x) for x in pose] for pose in poses]
            cam_params = [Camera(cam_param) for cam_param in cam_params]
        
            sample_wh_ratio = sample_size[1] / sample_size[0]
            pose_wh_ratio = args.original_pose_width / args.original_pose_height
            if pose_wh_ratio > sample_wh_ratio:
                resized_ori_w = sample_size[0] * pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / sample_size[1]
            else:
                resized_ori_h = sample_size[1] / pose_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / sample_size[0]
            intrinsic = np.asarray([[cam_param.fx * sample_size[1],
                                    cam_param.fy * sample_size[0],
                                    cam_param.cx * sample_size[1],
                                    cam_param.cy * sample_size[0]]
                                    for cam_param in cam_params], dtype=np.float32)
            K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]

            c2ws = get_relative_pose(cam_params, zero_first_frame_scale=True)
            c2ws[:, :3, -1] = c2ws[:, :3, -1] * args.speed
            
            c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
            
            pose_name = trajectory_path.split("/")[-1].split(".")[0]

            # extrinsics = c2ws.unsqueeze(1) # b 1 t 4 4
            # intrinsics = K.unsqueeze(1) # b 1 t 4
            
            camera_dict[pose_name] = {
                'intrinsics': K,
                'extrinsics': c2ws,
            }
        
        self.camera_dict = camera_dict

        
    def __len__(self):
        return self.length    
    
    def get_first_image(self, idx):
        first_image_path = sorted(os.listdir(self.scene_path_list[idx]))[0]
        image_path = os.path.join(self.scene_path_list[idx], first_image_path)
        
        image = torch.tensor(np.array(Image.open(image_path).convert('RGB'))).permute(2,0,1) / 255. * 2 - 1.  # [-1,1]

        return image, image_path

    def __getitem__(self, idx):
        condition_image, image_path = self.get_first_image(idx)
        cam_dict = self.camera_dict
        return {'condition_image': condition_image,
                'cam_dict': cam_dict,
                'sample_name': image_path.split('/')[-2]}


def main(args):
    
    local_rank = init_dist(launcher=args.launcher, port=args.port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = args.global_seed + global_rank
    torch.manual_seed(seed)
    
    
    if is_main_process:
        os.makedirs(os.path.join(args.out_root, 'generated_videos'), exist_ok=True)
    
    target_data = VideoDataset(
        input_path=args.data_path,
        trajectory_root_path="./",
        trajectory_file_list=args.trajectory_file_list,
    )
    
    distributed_sampler = DistributedSampler(
        target_data,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=False,
        seed=args.global_seed,
    )
    
    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        target_data,
        batch_size=1,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f'Constructing pipeline')
    model_configs = OmegaConf.load(args.model_config)
    OMSM_unet_ckpt_path = args.OMSM_unet_ckpt_path
    sam_ckpt_path = args.sam_ckpt_path
    
    USE_OMSM=False
    if args.OMSM_model_config is not None:
        USE_OMSM=True
        OMSM_model_configs = OmegaConf.load(args.OMSM_model_config)
        assert OMSM_unet_ckpt_path is not None
        assert sam_ckpt_path is not None
    print(f"Use OMSM: {USE_OMSM}")
    
    
    print(f'Constructing FVSM pipeline')
    pipeline_FVSM = get_pipeline_FVSM(args.ori_model_path, model_configs['unet_subfolder'], model_configs['down_block_types'],
                                 model_configs['up_block_types'], model_configs['flow_encoder_kwargs'],  model_configs['flow_generator_kwargs'], 
                                 model_configs['attention_processor_kwargs'], args.pose_adaptor_ckpt, args.enable_xformers, local_rank, sam_ckpt_path=sam_ckpt_path)
    
    
    if USE_OMSM:
        print(f'Constructing OMSM pipeline')
        pipeline_OMSM = get_pipeline_OMSM(args.ori_model_path, model_configs['unet_subfolder'], args.OMSM_unet_ckpt_path, OMSM_model_configs['flow_generator_kwargs'], local_rank)
    
    print('Done')
    
    
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)
    
    dataloader.sampler.set_epoch(1)
    
    dist.barrier()
    
    data_iter = iter(dataloader)
    for step in tqdm(range(0, len(dataloader))):
        batch = next(data_iter)
        
        cam_dict = batch['cam_dict']
        
        for cam_name, cam_param in cam_dict.items():
            sample_name = batch['sample_name'][0]
            
            # Check the existence
            tmp = os.path.join(args.out_root, 'sample_image', f'{sample_name}_{cam_name}')
            if os.path.isdir(tmp) and len(os.listdir(tmp)) == 14:
                print(f"pass {tmp}")
            
            else:           
                if is_main_process:
                    print(f"generate - {tmp}")
                
                condition_image = batch['condition_image'].to(local_rank)

                intrinsics = cam_param['intrinsics'].to(local_rank) # b
                extrinsics = cam_param['extrinsics'].to(local_rank)
                
                flow_generator_condition_image = repeat(condition_image, 'b c h w -> (b n) c h w', n=args.sample_num)
                flow_generator_input = get_flow_generator_input(condition_image=flow_generator_condition_image,
                                                                pixel_values=None,
                                                                intrinsics=repeat(intrinsics.to(device=local_rank), 'b c h w -> (b n) c h w', n=args.sample_num),
                                                                c2w=repeat(extrinsics.to(device=local_rank), 'b f c h w -> (b n) f c h w', n=args.sample_num),
                                                                runtime_type='inference')
                
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    with torch.no_grad():
                        if USE_OMSM:
                            obj_flow, OMSM_log_dict = pipeline_OMSM(
                                image=condition_image, # condition_image,
                                flow_generator_input=flow_generator_input,
                                height=args.image_height,
                                width=args.image_width,
                                num_frames=args.num_frames,
                                num_inference_steps=args.num_inference_steps,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                do_image_process=True,
                                generator=generator,
                                output_type='pt',
                            )
                        
                            sample, guidance_log_dict = pipeline_FVSM(
                                image=condition_image, # condition_image,
                                flow_generator_input=flow_generator_input,
                                height=args.image_height,
                                width=args.image_width,
                                num_frames=args.num_frames,
                                num_inference_steps=args.num_inference_steps,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                do_image_process=True,
                                generator=generator,
                                output_type='pt',
                                OMSM_log_dict=OMSM_log_dict,
                            )
                        else:
                            sample, guidance_log_dict = pipeline_FVSM(
                                image=condition_image, # condition_image,
                                flow_generator_input=flow_generator_input,
                                height=args.image_height,
                                width=args.image_width,
                                num_frames=args.num_frames,
                                num_inference_steps=args.num_inference_steps,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                do_image_process=True,
                                generator=generator,
                                output_type='pt',
                            )
                    
                
                sample_0 = sample.frames[0].cpu()     # [f, 3, h, w] 0-1
                sample_images = sample_0.detach().clone()
                sample_0 = sample_0.transpose(0, 1)      # [3, f, h, w] 0-1
                
                # Save results
                # 1) video
                video_save_path = os.path.join(args.out_root, 'generated_videos', f'{sample_name}_{cam_name}.mp4')
                save_videos_grid(sample_0[None], video_save_path)
                
                # 2) image
                sample_image_path = os.path.join(args.out_root, 'sample_image', f'{sample_name}_{cam_name}')
                os.makedirs(sample_image_path, exist_ok=True)
                for idx, image in enumerate(sample_images):
                    save_image_path = os.path.join(sample_image_path, f"{idx:02d}.png")
                    torchvision.utils.save_image(image, save_image_path)
                
                # 3) warped frames
                warped_image_path = os.path.join(args.out_root, 'warped_image', f'{sample_name}_{cam_name}')
                os.makedirs(warped_image_path, exist_ok=True)
                warped_images = guidance_log_dict['depth_warped_frames'][0].cpu().detach().clone()
                warped_images = (warped_images + 1.) / 2.
                
                for idx, image in enumerate(warped_images):
                    save_warped_image_path = os.path.join(warped_image_path, f"{idx:02d}.png")
                    torchvision.utils.save_image(image, save_warped_image_path)
                

            dist.barrier()
            




def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f"Argument {s} is not a list")
    return v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str)
    parser.add_argument("--image_height", type=int, default=320)
    parser.add_argument("--image_width", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=14, help="14 for svd and 25 for svd-xt", choices=[14, 25])
    parser.add_argument("--ori_model_path", type=str)
    parser.add_argument("--unet_subfolder", type=str, default='unet')
    parser.add_argument("--enable_xformers", action='store_true')
    parser.add_argument("--pose_adaptor_ckpt", default=None, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--min_guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_guidance_scale", type=float, default=3.0)
    parser.add_argument("--fix_guidance_scale", type=float, default=1.0)
    parser.add_argument("--original_pose_width", type=int, default=1280)
    parser.add_argument("--original_pose_height", type=int, default=720)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--use_adapter_classifier_free_guidance", action='store_true')
    parser.add_argument("--use_resolution_weighting", action='store_true')
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_num", type=int, default=1)
    
    # Object Motion Synthesis Model
    parser.add_argument("--OMSM_unet_ckpt_path", default=None, required=False)
    parser.add_argument("--OMSM_model_config", required=False, default=None)
    parser.add_argument("--data_path", required=True, help='path to testset')
    parser.add_argument("--sam_ckpt_path", required=False, default=None)
    parser.add_argument("--trajectory_file_list", type=arg_as_list, required=True)

    # DDP args
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of the distributed processes.")
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Replica rank on the current node. This field is required '
                             'by `torch.distributed.launch`.')
    parser.add_argument("--global_seed", default=42, type=int,
                        help="seed")
    parser.add_argument("--port", type=int)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    args = parser.parse_args()
    main(args)
