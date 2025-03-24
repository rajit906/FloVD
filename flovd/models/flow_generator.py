import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from kornia.morphology import dilation
from torchvision.utils import flow_to_image as flow_to_image_pt

from flovd.utils.util import instantiate_from_config
from flovd.modules.camera_flow_generator import CameraFlowGenerator # Inference
from flovd.modules.optical_flow_estimator import OpticalFlowEstimator # Training
from flovd.modules.raft.core.utils.flow_viz import flow_to_image
from flovd.modules.Segmentation.segmentation_wrapper import Segmentation_wrapper

import os
import pdb

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

def to_zero_to_one(x):
    return (x+1)/2

def flow_vis(flow, clip_flow=None):
    flow_np = flow.permute(0,2,3,1).detach().cpu().numpy()
    flow_ = torch.stack([torch.tensor(flow_to_image(x, clip_flow=clip_flow)) for x in flow_np], dim=0).permute(0,3,1,2)
    # flow_ = flow_ / 255 * 2 - 1
    flow_ = flow_ / 255
    return flow_


class FlowGenerator(nn.Module):
    def __init__(
        self, 
        optical_flow_estimator_kwargs,
        camera_flow_generator_kwargs,
        do_normalize_flow=True,
        scale_factor=None,
        segmentation_kwargs=None,
    ):
        super().__init__()
        print("[FlowGenerator] Start to init FlowGenerator")

        self.optical_flow_estimator = OpticalFlowEstimator(**optical_flow_estimator_kwargs) # Training
        self.camera_flow_generator = CameraFlowGenerator(**camera_flow_generator_kwargs) # Inference
        
        self.do_normalize_flow = do_normalize_flow
        self.scale_factor = scale_factor
        if scale_factor is not None:
            assert len(scale_factor) == 2, "Set scale_factor as length 2 of tuple or list! (sf_x, sf_y)"
    
        if segmentation_kwargs is not None:
            self.segmentation_wrapper = Segmentation_wrapper(sam_ckpt_path)
    
    def normalize_flow(self, flow):
        # flow: b t c h w
        assert flow.ndim==5
        assert self.scale_factor is not None
        
        flow_norm = adaptive_normalize(flow, self.scale_factor[0], self.scale_factor[1])
        
        return flow_norm
    
    def unnormalize_flow(self, flow):
        # flow: b t c h w
        assert flow.ndim==5
        assert self.scale_factor is not None
        
        flow_unnorm = adaptive_unnormalize(flow, self.scale_factor[0], self.scale_factor[1])

        return flow_unnorm

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
    
    @torch.no_grad()
    def forward(self, condition_image, camera_flow_generator_input, runtime_type, pixel_values=None):
        """
        Args
        - Refer to the func 'get_flow_generator_input'
        
        Returns
        - flow_embedding: input of FlowEncoder
        - flow_log_dict
        """
        
        assert runtime_type in ['training', 'inference'], "Choose runtime_type between ['training', 'inference']"
        H,W = condition_image.shape[-2:]
        
        # Get flow map
        input_dict = {
            "condition_image": condition_image,
        }
        
        if runtime_type == 'training':
            flow_generator = self.optical_flow_estimator
            input_dict.update({
                    "pixel_values": pixel_values
                })
        elif runtime_type == 'inference':
            flow_generator = self.camera_flow_generator
            input_dict.update({
                    "camera_flow_generator_input": camera_flow_generator_input
                })
        
        flow_maps, log_dict = flow_generator(**input_dict) # [bf c h w]
        flow_maps_vis = rearrange(flow_vis(flow_maps.detach().clone(), clip_flow=max(flow_maps.shape[-2:])), '(b f) c h w -> b f c h w', b=condition_image.shape[0])
        
        flow_maps = rearrange(flow_maps, "(b f) c h w -> b f c h w", b=condition_image.shape[0]) # [b f c h w]
        
        
        if self.do_normalize_flow:
            flow_maps = self.normalize_flow(flow_maps)
        
        flow_log_dict = {
            'flow_maps_vis': flow_maps_vis,
            'observed_mask': log_dict['observed_mask'] if 'observed_mask' in log_dict else None,
            'flow_warped_frames': rearrange(log_dict['flow_warped_frames'], "(b f) c h w -> b f c h w", b=condition_image.shape[0]) if 'flow_warped_frames' in log_dict else None,
            'depth_warped_frames': rearrange(log_dict['depth_warped_frames'], "(b f) c h w -> b f c h w", b=condition_image.shape[0]) if 'depth_warped_frames' in log_dict else None,
            'depth_ctxt': rearrange(log_dict['depth_ctxt'], "(b f) c h w -> b f c h w", b=condition_image.shape[0]) if 'depth_ctxt' in log_dict else None,
        }
        
        return flow_maps, flow_log_dict


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


def get_K(intrinsics, size):
    def normalize_intrinsic(x, size):
        h, w = size
        x[:,:,0:1] = x[:,:,0:1] / w
        x[:,:,1:2] = x[:,:,1:2] / h
        return x
    
    b, _, t, _ = intrinsics.shape
    K = torch.zeros((b, t, 9), dtype=intrinsics.dtype, device=intrinsics.device)
    fx, fy, cx, cy = intrinsics.squeeze(1).chunk(4, dim=-1)
    
    K[:,:,0:1] = fx
    K[:,:,2:3] = cx
    K[:,:,4:5] = fy
    K[:,:,5:6] = cy
    K[:,:,8:9] = 1.0
    
    K = rearrange(K, "b t (h w) -> b t h w", h=3, w=3)
    K = normalize_intrinsic(K, size)
    
    return K
    
    
def get_camera_flow_generator_input(condition_image, intrinsics, c2w, variable_intrinsic):
    """
    Args
    - condition_image: [b c h w]
    - intrinsics: [b, 1, t, 4] (fx, fy, cx, cy)
    - c2w: [b, 1, t, 4, 4]
    """
    c2w = c2w.squeeze(1)
    b = condition_image.shape[0]
    device = condition_image.device
    K = get_K(intrinsics, size=condition_image.shape[2:]) # [b t 3 3]
    c2w_dummy = repeat(torch.eye(4, dtype=c2w.dtype, device=device), "h w -> b 1 h w", b=c2w.shape[0])
    
    t = 1
    assert t == 1, "We use single image setting in 3D estimation networks! Now, you use more than one image for the context view."
    
    batch = dict()
    batch['context'] = {
            'image': to_zero_to_one(condition_image),
            'intrinsics': K[:,:1], 
            'extrinsics': c2w_dummy,
            'near': torch.ones((b, t), device=device),
            'far': torch.ones((b, t), device=device) * 100,
            'index': torch.arange(t).to(device)
    }

    
    b, t = c2w.shape[:2]

    batch['target'] = {
            'intrinsics': K,
            'extrinsics': c2w,
            'near': torch.ones((b, t), device=device),
            'far': torch.ones((b, t), device=device) * 100,
            'index': repeat(torch.arange(t).to(device), "t -> b t", b=b)
    }

    batch['scene'] = 'random'
    batch['variable_intrinsic'] = variable_intrinsic
    return batch


def get_flow_generator_input(condition_image, pixel_values, intrinsics, c2w, runtime_type, variable_intrinsic=False):
    """
    Args
        condition_image: Tensor, [b, c, h, w]
        pixel_values: Tensor, [bf, c, h, w] or None
        intrinsics: (fx, fy, cx, cy) Tensor, [b, f, 4]
        c2w: Tensor, [b, f, 4, 4]
        variable_intrinsic: bool, For Dolly-Zoom.
    """

    camera_flow_generator_input = get_camera_flow_generator_input(condition_image, intrinsics, c2w, variable_intrinsic)
    
    flow_generator_input = {
        'condition_image': condition_image,
        'pixel_values': pixel_values,
        'camera_flow_generator_input': camera_flow_generator_input,
        'runtime_type': runtime_type,
    }
    
    return flow_generator_input

    

def dimension_normalize(flow, H, W):
    flow_norm = flow.detach().clone()
    # flow_norm[:, :, 0] = 2.0 * flow[:, :, 0].detach().clone() / max(W - 1, 1) - 1.0
    # flow_norm[:, :, 1] = 2.0 * flow[:, :, 1].detach().clone() / max(H - 1, 1) - 1.0
    flow_norm[:, :, 0] = flow[:, :, 0].detach().clone() / max(W - 1, 1)
    flow_norm[:, :, 1] = flow[:, :, 1].detach().clone() / max(H - 1, 1)
    
    return flow_norm