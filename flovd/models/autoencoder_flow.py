import torch
import numpy as np
from einops import rearrange, repeat
from diffusers import AutoencoderKLTemporalDecoder
import pdb

class AutoencoderKLTemporalDecoder_Flow(AutoencoderKLTemporalDecoder):
    @torch.no_grad()
    def encode_flow(self, flow):
        # flow: BF,C,H,W
        b, c, h, w = flow.shape

        flow_input = torch.empty((b, 3, h, w)).to(flow)
        flow_input[:,:2] = flow
        # flow_input[:,-1:] = torch.linalg.norm(flow, dim=1, keepdim=True) # Duplicate norm of optical flow to channel axis
        flow_input[:,-1:] = flow.mean(dim=1, keepdim=True)
        
        return self.encode(flow_input)
    
    @torch.no_grad()
    def decode_flow(
        self,
        z: torch.Tensor,
        num_frames: int,
        return_dict: bool=True,
    ):
        b, c, h, w = z.shape
        output = self.decode(z, num_frames, return_dict)
        output.sample = output.sample[:,:2] # Only use two channels for optical flow
        
        return output