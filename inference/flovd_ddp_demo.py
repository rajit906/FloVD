"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import argparse
import logging
import os
import sys
from typing import Literal, Optional
from pathlib import Path
import json
from datetime import timedelta
import random
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from einops import rearrange, repeat
import math
import numpy as np
from PIL import Image

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import export_to_video, load_image, load_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

sys.path.append(os.path.abspath(os.path.join(sys.path[0], "../")))
from finetune.pipeline.flovd_FVSM_cogvideox_controlnet_pipeline import FloVDCogVideoXControlnetImageToVideoPipeline
from finetune.pipeline.flovd_OMSM_cogvideox_pipeline import FloVDOMSMCogVideoXImageToVideoPipeline
from finetune.schemas import Components, Args
from finetune.modules.cogvideox_controlnet import CogVideoXControlnet
from finetune.modules.cogvideox_custom_model import CustomCogVideoXTransformer3DModel
from transformers import AutoTokenizer, T5EncoderModel

from finetune.modules.camera_sampler import SampleManualCam
from finetune.modules.camera_flow_generator import CameraFlowGenerator
from finetune.modules.utils import get_camera_flow_generator_input, forward_bilinear_splatting, flow_to_color
from finetune.modules.depth_warping.depth_warping import unnormalize_intrinsic

from finetune.datasets.utils import (
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)


from torch.utils.data import Dataset
from torchvision import transforms

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import pdb
sys.path.append(os.path.abspath(os.path.join(sys.path[-1], 'finetune'))) # for camera flow generator


os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}




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


def load_cogvideox_flovd_FVSM_controlnet_pipeline(controlnet_path, backbone_path, device, dtype):
    controlnet_sd = torch.load(controlnet_path, map_location='cpu')['module']
    
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(backbone_path, subfolder="text_encoder")
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(backbone_path, subfolder="transformer")
    vae = AutoencoderKLCogVideoX.from_pretrained(backbone_path, subfolder="vae")
    scheduler = CogVideoXDPMScheduler.from_pretrained(backbone_path, subfolder="scheduler")
    
    additional_kwargs = {
        'num_layers': 6,
        'out_proj_dim_factor': 64,
        'out_proj_dim_zero_init': True,
        'notextinflow': True,
    }
    controlnet = CogVideoXControlnet.from_pretrained(backbone_path, subfolder="transformer", **additional_kwargs)
    controlnet.eval()
    
    missing, unexpected = controlnet.load_state_dict(controlnet_sd)
    
    if len(missing) != 0 or len(unexpected) != 0:
        print(f"Missing keys : {missing}")
        print(f"Unexpected keys : {unexpected}")
        
    pipe = FloVDCogVideoXControlnetImageToVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
    )
    
    # pipe.enable_model_cpu_offload(device=device)
    pipe = pipe.to(device, dtype)
    
    return pipe

def load_cogvideox_flovd_OMSM_lora_pipeline(omsm_path, backbone_path, transformer_lora_config, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(backbone_path, subfolder="text_encoder")
    transformer = CogVideoXTransformer3DModel.from_pretrained(backbone_path, subfolder="transformer")
    vae = AutoencoderKLCogVideoX.from_pretrained(backbone_path, subfolder="vae")
    scheduler = CogVideoXDPMScheduler.from_pretrained(backbone_path, subfolder="scheduler")

    # 1) Load Lora weight
    transformer.add_adapter(transformer_lora_config)

    lora_state_dict = FloVDOMSMCogVideoXImageToVideoPipeline.lora_state_dict(omsm_path)
    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.")
    }
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logger.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )

    # 2) Load Other weight
    load_path = os.path.join(omsm_path, "selected_blocks.safetensors")
    if os.path.exists(load_path):
        tensor_dict = load_file(load_path)
        
        block_state_dicts = {}
        for k, v in tensor_dict.items():
            block_name, param_name = k.split(".", 1)
            if block_name not in block_state_dicts:
                block_state_dicts[block_name] = {}
            block_state_dicts[block_name][param_name] = v
        
        for block_name, state_dict in block_state_dicts.items():
            if hasattr(transformer, block_name):
                getattr(transformer, block_name).load_state_dict(state_dict)
            else:
                raise ValueError(f"Transformer has no attribute '{block_name}'")
    
    
    pipe = FloVDOMSMCogVideoXImageToVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
    )
    
    # pipe.load_lora_weights(omsm_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
    # pipe.fuse_lora(components=["transformer"], lora_scale=1.0)
    
    # pipe.enable_model_cpu_offload(device=device)
    pipe = pipe.to(device, dtype)
    
    return pipe


class I2VFlowDataset_Inference(Dataset):
    def __init__(
        self, 
        max_num_frames: int, 
        height: int, 
        width: int, 
        data_root: str,
        max_num_videos: int = None,
    ) -> None:
        
        self.train_resolution = (int(max_num_frames), int(height), int(width))
        
        data_root = Path(data_root)
        metadata_path = data_root / "metadata_reformat.jsonl"
        assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl in the root path"
        
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append( json.loads(line) )
        
        metadata = random.sample(metadata, max_num_videos)

        self.prompts = [x["prompt"] for x in metadata]
        self.prompt_embeddings = [data_root / "prompt_embeddings_revised" / (x["hash_code"] + '.safetensors') for x in metadata]
        self.videos = [data_root / "video_latent" / "x".join(str(x) for x in self.train_resolution) / (x["hash_code"] + '.safetensors') for x in metadata]
        self.images = [data_root / "first_frames" / (x["hash_code"] + '.png') for x in metadata]
        self.flows = [data_root / "flow_direct_f_latent" / (x["hash_code"] + '.safetensors') for x in metadata]
        self.masks = [data_root / "valid_mask" / (x["hash_code"] + '.bin') for x in metadata]
        
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms
        
        self.length = len(self.videos)

        print(f"Dataset size: {self.length}")
        
    def __len__(self) -> int:
        return self.length
    
    def load_data_pair(self, index):
        prompt_embedding_path = self.prompt_embeddings[index]
        encoded_video_path = self.videos[index]
        encoded_flow_path = self.flows[index]
        
        prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"] 
        encoded_video = load_file(encoded_video_path)["encoded_video"] # CFHW
        encoded_flow = load_file(encoded_flow_path)["encoded_flow_f"] # CFHW
    
        return prompt_embedding, encoded_video, encoded_flow

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                prompt_embedding, encoded_video, encoded_flow = self.load_data_pair(index)
                break
            except Exception as e:
                print(f"Error loading {self.prompt_embeddings[index]}: {str(e)}")
                index = random.randint(0, self.length - 1)
            
        image_path = self.images[index]
        prompt = self.prompts[index]
        
        _, image = self.preprocess(None, image_path)
        image = self.image_transform(image)
        
        
        # shape of encoded_video: [C, F, H, W]
        # shape and scale of image: [C, H, W], [-1,1]
        return {
            "image": image,
            "prompt": prompt,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "encoded_flow": encoded_flow,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }
    
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image
    
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)

def initialize_flow_generator(target):
    depth_estimator_kwargs = {
        "target": target,
        "kwargs": {
            "ckpt_path": '/workspace/workspace/checkpoints/depth_anything/depth_anything_v2_metric_hypersim_vitb.pth',
            "model_config": {
                "max_depth": 20,
                "encoder": 'vitb',
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            }

        }
    }

    return CameraFlowGenerator(depth_estimator_kwargs)

def integrate_flow(camera_flow, object_flow, depth_ctxt, camera_flow_generator, camera_flow_generator_input):
    # camera_flow: (BF)CHW
    # object_flow: (BF)CHW
    # depth_ctxt: B1HW
    
    B, F = camera_flow_generator_input["target"]["intrinsics"].shape[:2]
    H, W = object_flow.shape[-2:]
    
    c2w_ctxt = repeat(camera_flow_generator_input["context"]["extrinsics"], "b t h w -> (b v t) h w", v=F) # No need to apply inverse as it is an eye matrix.
    c2w_trgt = rearrange(torch.inverse(camera_flow_generator_input["target"]["extrinsics"]), "b t h w -> (b t) h w")
    intrinsics_ctxt = unnormalize_intrinsic(repeat(camera_flow_generator_input["context"]["intrinsics"], "b t h w -> (b v t) h w", v=F), size=(H, W))
        
    with torch.cuda.amp.autocast(enabled=False):
        warped_object_flow = camera_flow_generator.depth_warping_module.warper.forward_warp_displacement(
            depth1=repeat(depth_ctxt, "b c h w -> (b f) c h w", f=F),
            flow1=object_flow, 
            transformation1=c2w_ctxt, 
            transformation2=c2w_trgt, 
            intrinsic1=intrinsics_ctxt, 
            intrinsic2=None,
        )
    
    integrated_flow = camera_flow + warped_object_flow
    
    return integrated_flow
    
def save_flow(flow, filename, fps=16):
    # flow: (BF)CHW, arbitrary scale
    flow_RGB = flow_to_color(flow) # BF,C,H,W (B=1)

    frame_list = []
    for frame in flow_RGB:
        frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
        frame_list.append(Image.fromarray(frame))
    
    export_to_video(frame_list, filename, fps=fps)

def save_flow_warped_video(image, flow, filename, fps=16):
    # image: CHW, 0~255 scale
    # flow: (BF)CHW, arbitrary scale
    warped_video = forward_bilinear_splatting(repeat(image, 'c h w -> f c h w', f=flow.size(0)), flow.to(torch.float))
    
    frame_list = []
    for frame in warped_video:
        frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
        frame_list.append(Image.fromarray(frame))
    
    export_to_video(frame_list, filename, fps=fps)

def generate_video(
    # prompt: str,
    launcher: str,
    port: int,
    data_root: str,
    fvsm_path: str,
    omsm_path: str,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 16,
    controlnet_guidance_end: float = 0.4,
    max_num_videos: int = None,
    use_dynamic_cfg: bool = False,
    pose_type: str = "manual",
    speed: float = 0.5,
    use_flow_integration: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    
    # Distributed
    local_rank = init_dist(launcher=launcher, port=port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    if is_main_process:
        os.makedirs(os.path.join(output_path, 'generated_videos'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'generated_flow_videos'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'flow_warped_videos'), exist_ok=True)

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    model_name = "cogvideox-5b-i2v".lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m")
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )

    """
        # Prepare Dataset Class..
    """
    # image = load_image(image=image_or_video_path)
    
    # prompt
    # first image
    # camera parameters
    dataset = I2VFlowDataset_Inference(
        max_num_frames=num_frames,
        height=height,
        width=width,
        data_root=data_root,
        max_num_videos=max_num_videos,
    )
    
    
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=False,
        seed=seed,
    )
    
    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    

    """
        # Prepare Pipeline
    """
    transformer_lora_config = LoraConfig(
        r=128,
        lora_alpha=64,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "norm1.linear", "norm2.linear", "ff.net.2"],
    )
    
    print(f'Constructing pipeline')
    pipe_omsm = load_cogvideox_flovd_OMSM_lora_pipeline(omsm_path, backbone_path="THUDM/CogVideoX-5b-I2V", transformer_lora_config=transformer_lora_config, device=local_rank, dtype=dtype)   
    pipe_fvsm = load_cogvideox_flovd_FVSM_controlnet_pipeline(fvsm_path, backbone_path="THUDM/CogVideoX-5b-I2V", device=local_rank, dtype=dtype)   
    print(f'Done loading pipeline')
    
    assert pose_type in ['re10k', 'manual'], "Choose other pose_type between ['re10k', 'manual']"
    if pose_type == 're10k':
        root_path = "./manual_poses_re10k"
    else:
        root_path = "./manual_poses"
        
    CameraSampler = SampleManualCam(pose_type=pose_type, root_path=root_path)
    camera_flow_generator_target = 'finetune.modules.depth_warping.depth_warping.DepthWarping_wrapper'
    camera_flow_generator = initialize_flow_generator(camera_flow_generator_target).to(local_rank)
    #--------------------------------------------------------------------------------------------------------


    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe_fvsm.scheduler = CogVideoXDPMScheduler.from_config(pipe_fvsm.scheduler.config, timestep_spacing="trailing")
    pipe_omsm.scheduler = CogVideoXDPMScheduler.from_config(pipe_omsm.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    # pipe.to("cuda")

    # pipe_fvsm.enable_model_cpu_offload()
    # pipe_omsm.enable_model_cpu_offload()
    # pipe_fvsm.enable_sequential_cpu_offload()
    # pipe_omsm.enable_sequential_cpu_offload()
    
    pipe_fvsm.vae.enable_slicing()
    pipe_fvsm.vae.enable_tiling()
    pipe_omsm.vae.enable_slicing()
    pipe_omsm.vae.enable_tiling()
    
    dataloader.sampler.set_epoch(1)
    dist.barrier()
    
    output_video_path = os.path.join(output_path, 'generated_videos')
    output_flow_path = os.path.join(output_path, 'generated_flow_videos')
    output_warped_video_path = os.path.join(output_path, 'flow_warped_videos')
    
    data_iter = iter(dataloader)
    for step in tqdm(range(0, len(dataloader))):
        batch = next(data_iter)
        
        prompt = batch["prompt"][0]
        image = batch["image"].to(local_rank)
        prompt_embedding = batch["prompt_embedding"].to(local_rank)
        prompt_short = prompt[:20].strip()
        
        # if step < 10:
        #     step += 1
        #     continue
        
        # Get Camera flow
        camparam, cam_name = CameraSampler.sample() # W2C
        image_torch_255 = ((image.detach().clone()+1)/2. * 255.).squeeze(0)
        camera_flow_generator_input = get_camera_flow_generator_input(image_torch_255, camparam, device=local_rank, speed=speed)
        image_torch = ((image_torch_255.unsqueeze(0) / 255.) * 2. - 1.).to(local_rank)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                
                # camera_flow, log_dict = camera_flow_generator(image_torch, camera_flow_generator_input)
                # camera_flow = camera_flow.to(local_rank, dtype)
                
                # camera_flow_latent = rearrange(encode_flow(camera_flow, pipe_omsm.vae, flow_scale_factor=[60, 36]), 'b c f h w -> b f c h w').to(local_rank, dtype)
                
                flow_latent = pipe_omsm(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    prompt=None,
                    prompt_embeds=prompt_embedding,
                    image=image,
                    generator=torch.Generator().manual_seed(seed),
                    num_inference_steps=num_inference_steps,
                    use_dynamic_cfg=use_dynamic_cfg,
                    output_type='latent'
                ).frames[0]
                object_flow = decode_flow(flow_latent.detach().clone().unsqueeze(0).to(local_rank), pipe_omsm.vae, flow_scale_factor=[60, 36]) # BF,C,H,W
                
                if use_flow_integration:
                    # Integrate camera (from 3D warping) and object (from OMSM) flow maps
                    # Using segmentation model will be implemented later..
                    
                    camera_flow, log_dict = camera_flow_generator(image_torch, camera_flow_generator_input)
                    camera_flow = camera_flow.to(local_rank, dtype)

                    integrated_flow = integrate_flow(camera_flow, object_flow, log_dict['depth_ctxt'], camera_flow_generator, camera_flow_generator_input)
                    integrated_flow_latent = rearrange(encode_flow(integrated_flow, pipe_omsm.vae, flow_scale_factor=[60, 36]), 'b c f h w -> b f c h w').to(local_rank, dtype)
                else:
                    integrated_flow_latent = rearrange(flow_latent, '(b f) c h w -> b f c h w', b=image.size(0))
                
                # 4. Generate the video frames based on the prompt.
                # `num_frames` is the Number of frames to generate.
                video_generate = pipe_fvsm(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    prompt=None,
                    prompt_embeds=prompt_embedding,
                    image=image,
                    flow_latent=integrated_flow_latent,
                    valid_mask=None,
                    generator=torch.Generator().manual_seed(seed),
                    num_inference_steps=num_inference_steps,
                    controlnet_guidance_start = 0.0,
                    controlnet_guidance_end = controlnet_guidance_end,
                    use_dynamic_cfg=use_dynamic_cfg,
                ).frames[0]

        # Save logs
        # 1) Synthesized flow (object_flow)
        save_path = os.path.join(output_flow_path, f"{prompt_short}_DCFG-{use_dynamic_cfg}_ContGuide-{controlnet_guidance_end}_{cam_name}.mp4")
        save_flow(object_flow, filename=save_path, fps=fps)
        
        # 2) Flow-Warped Video
        save_path = os.path.join(output_warped_video_path, f"{prompt_short}_DCFG-{use_dynamic_cfg}_ContGuide-{controlnet_guidance_end}_{cam_name}.mp4")
        save_flow_warped_video(image_torch_255, object_flow, filename=save_path, fps=fps)

        # 3) Flow-Cond. Synthesized Video
        save_path = os.path.join(output_video_path, f"{prompt_short}_DCFG-{use_dynamic_cfg}_ContGuide-{controlnet_guidance_end}_{cam_name}.mp4")
        export_to_video(video_generate, save_path, fps=fps)
        
        dist.barrier()
        
        step += 1


#--------------------------------------------------------------------------------------------------
def encode_video(video: torch.Tensor, vae) -> torch.Tensor:
    # shape of input video: [B, C, F, H, W]
    video = video.to(vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    return latent

def encode_flow(flow, vae, flow_scale_factor):
    # flow: BF,C,H,W
    # flow_scale_factor [sf_x, sf_y]
    assert flow.ndim == 4
    num_frames, _, height, width = flow.shape

    # Normalize optical flow
    # ndim: 4 -> 5
    flow = rearrange(flow, '(b f) c h w -> b f c h w', b=1)
    flow_norm = adaptive_normalize(flow, flow_scale_factor[0], flow_scale_factor[1])

    # ndim: 5 -> 4
    flow_norm = rearrange(flow_norm, 'b f c h w -> (b f) c h w', b=1)

    # Duplicate mean value for third channel
    num_frames, _, H, W = flow_norm.shape
    flow_norm_extended = torch.empty((num_frames, 3, height, width)).to(flow_norm)
    flow_norm_extended[:,:2] = flow_norm
    flow_norm_extended[:,-1:] = flow_norm.mean(dim=1, keepdim=True)
    flow_norm_extended = rearrange(flow_norm_extended, '(b f) c h w -> b c f h w', f=num_frames)

    return encode_video(flow_norm_extended, vae)


def decode_flow(flow_latent, vae, flow_scale_factor):
    flow_latent = flow_latent.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    flow_latent = 1 / vae.config.scaling_factor * flow_latent
    
    flow = vae.decode(flow_latent).sample # BCFHW

    # discard third channel (which is a mean value of f_x and f_y)
    flow = flow[:,:2].detach().clone()

    # Unnormalize optical flow
    flow = rearrange(flow, 'b c f h w -> b f c h w')
    flow = adaptive_unnormalize(flow, flow_scale_factor[0], flow_scale_factor[1])

    flow = rearrange(flow, 'b f c h w -> (b f) c h w')
    return flow # BF,C,H,W

def adaptive_normalize(flow, sf_x, sf_y):
    # x: BFCHW, optical flow
    assert flow.ndim == 5, 'Set the shape of the flow input as (B, F, C, H, W)'
    assert sf_x is not None and sf_y is not None
    b, f, c, h, w = flow.shape
    
    max_clip_x = math.sqrt(w/sf_x) * 1.0
    max_clip_y = math.sqrt(h/sf_y) * 1.0
    
    flow_norm = flow.detach().clone()
    flow_x = flow[:, :, 0].detach().clone()
    flow_y = flow[:, :, 1].detach().clone()
    
    flow_x_norm = torch.sign(flow_x) * torch.sqrt(torch.abs(flow_x)/sf_x + 1e-7)
    flow_y_norm = torch.sign(flow_y) * torch.sqrt(torch.abs(flow_y)/sf_y + 1e-7)

    flow_norm[:, :, 0] = torch.clamp(flow_x_norm, min=-max_clip_x, max=max_clip_x)
    flow_norm[:, :, 1] = torch.clamp(flow_y_norm, min=-max_clip_y, max=max_clip_y)

    return flow_norm


def adaptive_unnormalize(flow, sf_x, sf_y):
    # x: BFCHW, optical flow
    assert flow.ndim == 5, 'Set the shape of the flow input as (B, F, C, H, W)'
    assert sf_x is not None and sf_y is not None
    
    flow_orig = flow.detach().clone()
    flow_x = flow[:, :, 0].detach().clone()
    flow_y = flow[:, :, 1].detach().clone()
    
    flow_orig[:, :, 0] = torch.sign(flow_x) * sf_x * (flow_x**2 - 1e-7)
    flow_orig[:, :, 1] = torch.sign(flow_y) * sf_y * (flow_y**2 - 1e-7)
    
    return flow_orig

#--------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    # parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument("--image_path", type=str, default=None, help="The path of the image to be used as the background of the video",)
    parser.add_argument("--data_root", type=str, required=True, help="The path of the dataset root",)
    parser.add_argument("--fvsm_path", type=str, required=True, help="Path of the pre-trained model use")
    parser.add_argument("--omsm_path", type=str, required=True, help="Path of the pre-trained model use")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.4, help="Controlnet guidance end during sampling")
    parser.add_argument("--max_num_videos", type=int, default=None, help="# of videos for inference")
    parser.add_argument("--use_dynamic_cfg", action='store_true')
    parser.add_argument("--pose_type", type=str, default='manual', help="pose type in the inference time")
    parser.add_argument("--speed", type=float, default=0.5, help="pose type in the inference time")
    parser.add_argument("--use_flow_integration", action='store_true')
    
    
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
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    
    generate_video(
        # prompt=args.prompt,
        launcher=args.launcher,
        port=args.port,
        data_root=args.data_root,
        fvsm_path=args.fvsm_path,
        omsm_path=args.omsm_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_path=args.image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
        controlnet_guidance_end=args.controlnet_guidance_end,
        max_num_videos=args.max_num_videos,
        use_dynamic_cfg=args.use_dynamic_cfg,
        pose_type=args.pose_type,
        speed=args.speed,
        use_flow_integration=args.use_flow_integration,
    )
