import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# from flovd.modules.depth_warping_module 
from flovd.utils.util import instantiate_from_config
from flovd.modules.utils import run_filtering, warp_image

import pdb

class CameraFlowGenerator(nn.Module):
    def __init__(
        self,
        depth_estimator_kwargs,
        use_observed_mask=False,
        cycle_th=3.,
    ):
        super().__init__()
        
        self.depth_warping_module = instantiate_from_config(depth_estimator_kwargs)
        self.use_observed_mask = use_observed_mask
        self.cycle_th = cycle_th
    
    def forward(self, condition_image, camera_flow_generator_input):
        # NOTE. camera_flow_generator_input is a dict of network inputs!
        # camera_flow_generator_input: Dict
        # - image
        # - intrinsics
        # - extrinsics
        with torch.no_grad():
            flow_f, flow_b, depth_warped_frames, depth_ctxt, depth_trgt = self.depth_warping_module(camera_flow_generator_input)
        image_ctxt = repeat(condition_image, "b c h w -> (b v) c h w", v=(depth_warped_frames.shape[0]//condition_image.shape[0]))
        
        log_dict = {
            'depth_warped_frames': depth_warped_frames,
            'depth_ctxt': depth_ctxt,
            'depth_trgt': depth_trgt,
        }
        
        if self.use_observed_mask:
            observed_mask = run_filtering(flow_f, flow_b, cycle_th=self.cycle_th)
            log_dict[
                'observed_mask': observed_mask
            ]
        
        return flow_f, log_dict
