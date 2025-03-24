import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cameractrl.utils.util import instantiate_from_config

import pdb



class RAFT_wrapper():
    def __init__(
        self,
        raft_model_args,
        iters=20,
    ):
        
        self.raft_model = instantiate_from_config(raft_model_args)
        print("[RAFT] Start to load ckpt!")
        sd = torch.load(self.raft_model.args.model)
        sd_new = self.get_module_name(sd)
        print("[RAFT] Complete to load ckpt. Now, copy params!")
        self.raft_model.load_state_dict(sd_new)

        self.iters = iters
    
    def get_module_name(self, sd):
        sd_new = {}
        for key, val in sd.items():
            if key.split('.')[0] == 'module':
                sd_new[key[7:]] = val
        
        return sd_new
    

    def warp_image(self, image, flow):
        h, w = flow.shape[2:]
        flow = -flow
        
        # Create a mesh grid
        meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        grid = torch.stack(meshgrid, dim=2).float().to(image.device)  # Shape: (h, w, 2)
        
        # Apply flow to the grid
        flow_map = repeat(grid, "h w c -> b h w c", b=flow.shape[0]) + flow.permute(0, 2, 3, 1)  # Permute to match grid shape (h, w, 2)
        # flow_map = flow_map.unsqueeze(0)  # Add batch dimension

        # Normalize the flow map to [-1, 1] range for grid_sample
        flow_map[..., 0] = 2.0 * flow_map[..., 0] / max(w - 1, 1) - 1.0
        flow_map[..., 1] = 2.0 * flow_map[..., 1] / max(h - 1, 1) - 1.0
        
        # Warp image using grid_sample
        warped_image = F.grid_sample(image, flow_map, mode='bilinear', align_corners=True)
        
        # Create the unobserved mask
        observed_mask = (flow_map[..., 0] >= -1.0) & (flow_map[..., 0] <= 1.0) & (flow_map[..., 1] >= -1.0) & (flow_map[..., 1] <= 1.0)
        
        # Visualization
        # from PIL import Image
        # idx=13
        # Image.fromarray((((warped_image[idx].permute(1,2,0).detach().cpu().numpy()+1)/2)*255).astype(np.uint8)).save(f"warped_image_{idx:02d}.png")
        # Image.fromarray((unobserved_mask[idx].float().detach().cpu().numpy()*255).astype(np.uint8)).save(f"UM_{idx:02d}.png")
        
        return warped_image, observed_mask.float()

    
    def predict(self, ctxt, trgt):
        """
        Args:
            ctxt: input image, (b)chw, [-1,1]
            trgt: video frames, (bt)chw, [-1,1]
        Returns:
            1) optical flow
            2) optical flow based masks..
        """
        self.raft_model.to(ctxt.device)
        
        ctxt_input = repeat(ctxt, "b c h w -> (b t) c h w", t=(trgt.shape[0]//ctxt.shape[0]))
        
        # Forward flow
        flow_low, flow_up = self.raft_model(ctxt_input, trgt, iters=self.iters, test_mode=True)
        
        # Backward flow
        # flow_1_low, flow_1_up = self.raft_model(trgt, ctxt_input, iters=self.iters, test_mode=True)
        
        # Get warped_image and unobserved_mask
        warped_image, observed_mask = self.warp_image(ctxt_input, flow_up)
        
        # Image.fromarray((((trgt[idx].permute(1,2,0).detach().cpu().numpy()+1)/2)*255).astype(np.uint8)).save(f"trgt_image_{idx:02d}.png")
        # Image.fromarray((((warped_image[idx].permute(1,2,0).detach().cpu().numpy()+1)/2)*255).astype(np.uint8)).save(f"warped_image_{idx:02d}.png")
                
        
        return warped_image, observed_mask