import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from flovd.utils.util import instantiate_from_config
from flovd.modules.utils import run_filtering, warp_image

import pdb

class OpticalFlowEstimator(nn.Module):
    def __init__(
        self,
        raft_iter=20,
        cycle_th=3.,
        use_observed_mask=False
    ):
        super().__init__()
        
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.raft_iter = raft_iter
        
        self.use_observed_mask = use_observed_mask
        self.cycle_th = cycle_th
    
    def compute_optical_flow(self, ctxt, trgt):
        with torch.no_grad():
            return self.raft(ctxt, trgt, num_flow_updates=self.raft_iter)[-1]
    
    def forward(self, condition_image, pixel_values):
        v = pixel_values.shape[0] // condition_image.shape[0]
        image_ctxt_raft = repeat(condition_image, "b c h w -> (b v) c h w", v=v)
        
        flow_f = self.compute_optical_flow(image_ctxt_raft, pixel_values)
        flow_warped_frames = warp_image(image_ctxt_raft, flow_f)
        
        log_dict = {
            'flow_warped_frames': flow_warped_frames
        }
        
        if self.use_observed_mask:
            flow_b = self.compute_optical_flow(pixel_values, image_ctxt_raft)
            observed_mask = run_filtering(flow_f, flow_b, cycle_th=self.cycle_th)
            log_dict[
                'observed_mask': observed_mask
            ]
        
        return flow_f, log_dict