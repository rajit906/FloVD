from transformers import pipeline
from PIL import Image
import requests
import torchvision
import os
from .camera.WarperPytorch import Warper
import numpy as np
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from flovd.modules.depth_warping.depth_anything_v2.dpt import DepthAnythingV2

import pdb

def to_pil_image(x):
    # x: c h w, [-1, 1]
    x_np = ((x+1)/2*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    x_pil = Image.fromarray(x_np)
    
    return x_pil

def to_npy(x):
    return ((x+1)/2*255).permute(1,2,0).detach().cpu().numpy()

def unnormalize_intrinsic(x, size):
    h, w = size
    x_ = x.detach().clone()
    x_[:,0:1] = x[:,0:1].detach().clone() * w
    x_[:,1:2] = x[:,1:2].detach().clone() * h
    return x_

class DepthWarping_wrapper(nn.Module):
    def __init__(self,
                 model_config,
                 ckpt_path,):
        super().__init__()
        
        # self.depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model = DepthAnythingV2(**model_config)
        self.depth_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.depth_model = self.depth_model.eval()
        self.warper = Warper()
    
    def get_input(self, batch):
        # pdb.set_trace()
        
        b, v = batch["target"]["intrinsics"].shape[:2]
        h, w = batch["context"]["image"].shape[-2:]
        
        
        image = (batch["context"]["image"]) * 2 - 1
        image_ctxt = repeat(image, "b c h w -> (b v) c h w", v=v)
        c2w_ctxt = repeat(batch["context"]["extrinsics"], "b t h w -> (b v t) h w", v=v) # No need to apply inverse as it is an eye matrix.
        c2w_trgt = rearrange(torch.inverse(batch["target"]["extrinsics"]), "b t h w -> (b t) h w")
        intrinsics_ctxt = unnormalize_intrinsic(repeat(batch["context"]["intrinsics"], "b t h w -> (b v t) h w", v=v), size=(h,w))
        intrinsics_trgt = unnormalize_intrinsic(rearrange(batch["target"]["intrinsics"], "b t h w -> (b t) h w"), size=(h,w))
        
        # image = image.squeeze(1)
        # depth_ctxt = torch.stack([torch.tensor(self.depth_model.infer_image(to_npy(x))) for x in image], dim=0).to(image.device).unsqueeze(1) # B 1 H W
        depth_ctxt = torch.stack([self.depth_model.infer_image(to_npy(x)) for x in image], dim=0).to(image.device).unsqueeze(1) # B 1 H W

        # depth_ctxt = torch.nn.functional.interpolate(
        #     depth_ctxt,
        #     size=(h,w),
        #     mode="bicubic",
        #     align_corners=False,
        # )
        
        return image_ctxt, c2w_ctxt, c2w_trgt, intrinsics_ctxt, intrinsics_trgt, depth_ctxt, batch['variable_intrinsic']
    
    def forward(self, batch):
        image_ctxt, c2w_ctxt, c2w_trgt, intrinsics_ctxt, intrinsics_trgt, depth_ctxt, variable_intrinsic = self.get_input(batch)
        
        b, v = batch["target"]["intrinsics"].shape[:2]
        # h, w = image_ctxt.shape[-2:]
        

        with torch.cuda.amp.autocast(enabled=False):
            warped_trgt, mask_trgt, warped_depth_trgt, flow_f = self.warper.forward_warp(
                frame1=image_ctxt, 
                mask1=None, 
                depth1=repeat(depth_ctxt, "b c h w -> (b t) c h w", t=v), 
                transformation1=c2w_ctxt, 
                transformation2=c2w_trgt, 
                intrinsic1=intrinsics_ctxt, 
                intrinsic2=intrinsics_trgt if variable_intrinsic else None)
            
            warped_src, mask_src, warped_depth_src, flow_b = self.warper.forward_warp(
                frame1=warped_trgt, 
                mask1=None, 
                depth1=warped_depth_trgt, 
                transformation1=c2w_trgt, 
                transformation2=c2w_ctxt, 
                intrinsic1=intrinsics_trgt, 
                intrinsic2=None)
        
        # if use_backward_flow:
        #     mask = mask_trgt
        # else:
        #     mask = mask_src

        return flow_f, flow_b, warped_trgt, depth_ctxt, warped_depth_trgt 

        
