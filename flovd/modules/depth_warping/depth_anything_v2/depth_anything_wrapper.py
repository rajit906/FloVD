import os
from cameractrl.modules.depth_anything_v2.dpt import DepthAnythingV2

class MVSplat_wrapper(nn.Module):
    def __init__(
            self,
            model_configs,
            ckpt_path,
        ):
        super().__init__()
        
        depth_anything = DepthAnythingV2(model_configs)