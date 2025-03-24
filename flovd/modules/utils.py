import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def warp_image(image, flow, use_forward_flow=True):
    """
    Args
        image: context image (src view image)
        flow: forward (src -> trgt) or backward optical flow (trgt -> src)
    """
    assert image.ndim==4 and flow.ndim==4
    
    h, w = flow.shape[2:]
    if use_forward_flow:
        flow = -flow

    # Create a mesh grid
    meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    grid = torch.stack(meshgrid, dim=2).float().to(image.device)  # Shape: (h, w, 2)

    # Apply flow to the grid
    flow_map = repeat(grid, "h w c -> b h w c", b=flow.shape[0]) + flow.permute(0, 2, 3, 1)  # Permute to match grid shape (h, w, 2)

    # Normalize the flow map to [-1, 1] range for grid_sample
    flow_map[..., 0] = 2.0 * flow_map[..., 0] / max(w - 1, 1) - 1.0
    flow_map[..., 1] = 2.0 * flow_map[..., 1] / max(h - 1, 1) - 1.0

    # Warp image using grid_sample
    warped_image = F.grid_sample(image, flow_map, mode='bilinear', align_corners=True)

    # Create the unobserved mask
    # observed_mask = (flow_map[..., 0] >= -1.0) & (flow_map[..., 0] <= 1.0) & (flow_map[..., 1] >= -1.0) & (flow_map[..., 1] <= 1.0)

    return warped_image



def gen_grid(h, w, device, normalize=False, homogeneous=False):
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h, device=device)
        lin_x = torch.linspace(-1., 1., steps=w, device=device)
    else:
        lin_y = torch.arange(0, h, device=device)
        lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]


def normalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    if no_shift:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
    else:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.


def run_filtering(flow_f, flow_b, cycle_th=3.):
    """
    Args:
        flow_f: b 2 h w
        flow_b: b 2 h w
    Returns:
        mask_cycle: binary mask (0: Not consistent or 1: consistent), float, [b 1 h w]
    """
    assert flow_f.ndim == 4 and flow_b.ndim == 4
    
    device = flow_f.device
    h, w = flow_f.shape[-2:]
    num_imgs = flow_f.shape[0]
    
    flow_f = flow_f
    flow_b = flow_b
    
    grid = repeat(gen_grid(h, w, device=device).permute(2, 0, 1)[None], "b c h w -> (b v) c h w", v=num_imgs)
    # grid_normed = normalize_coords(grid.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]
    
    
    coord2 = flow_f + grid
    coord2_normed = normalize_coords(coord2.permute(0, 2, 3, 1), h, w)
    flow_21_sampled = F.grid_sample(flow_b, coord2_normed, align_corners=True)
    map_i = flow_f + flow_21_sampled
    fb_discrepancy = torch.norm(map_i.squeeze(), dim=1)
    mask_cycle = fb_discrepancy < cycle_th
    
    return mask_cycle.unsqueeze(1).float()