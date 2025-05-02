import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import pdb

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[:4]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[6:]).reshape(3, 4)
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

def get_camera_flow_generator_input(condition_image, camparams, device, speed=1.0):
    """
    Args
    - condition_image: [c h w], scale~[0,255]
    - camparam: [b, 18] (fx, fy, cx, cy, 0, 0, 3x4 Rt matrix), W2C.
    - intrinsic: [b, 1, t, 4] (fx, fy, cx, cy)
    - c2w: [b, 1, t, 4, 4]
    """

    condition_image = condition_image.unsqueeze(0)/255. # bchw, scale~[0,1]
    sample_size = condition_image.shape[2:]
    
    cam_params = [[float(x) for x in camparam] for camparam in camparams]
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    intrinsic = np.asarray([[cam_param.fx * sample_size[1],
                            cam_param.fy * sample_size[0],
                            cam_param.cx * sample_size[1],
                            cam_param.cy * sample_size[0]]
                            for cam_param in cam_params], dtype=np.float32)
    
    intrinsic = torch.as_tensor(intrinsic).unsqueeze(0).unsqueeze(0) # [1, 1, f, 4]

    c2w = get_relative_pose(cam_params, zero_first_frame_scale=True)
    c2w[:, :3, -1] = c2w[:, :3, -1] * speed
    c2w = torch.as_tensor(c2w)

    c2w = c2w.unsqueeze(0)
    b = condition_image.shape[0]
    t = c2w.shape[1]
    K = get_K(intrinsic, size=condition_image.shape[2:]) # [b t 3 3]
    c2w_dummy = repeat(torch.eye(4, dtype=c2w.dtype, device=device), "h w -> b 1 h w", b=c2w.shape[0])

    t = 1
    assert t == 1, "We use single image setting in 3D estimation networks! Now, you use more than one image for the context view."
    
    batch = dict()
    batch['context'] = {
            'image': condition_image,
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
    batch['variable_intrinsic'] = None
    return batch

def to_zero_to_one(x):
    return (x+1)/2



def instantiate_from_config(config, **additional_kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    additional_kwargs.update(config.get("kwargs", dict()))
    return get_obj_from_str(config["target"])(**additional_kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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

def forward_bilinear_splatting(image, flow, mask=None):
    """
    Forward warping (splatting) with bilinear interpolation for an entire batch at once.

    Args:
      image: (B, 3, H, W)  # 소스 이미지
      flow:  (B, 2, H, W)  # forward flow (dx, dy)
      mask:  (B, 1, H, W)  # 1: valid, 0: invalid

    Returns:
      warped: (B, 3, H, W) # forward warp 결과
    """
    
    device = image.device
    B, C_i, H, W = image.shape
    
    if mask is None:
        mask = torch.ones(B, 1, H, W).to(device, flow.dtype)
    
    assert C_i == 3, f"image의 채널 수는 3이어야 합니다. (현재: {C_i})"
    assert flow.shape == (B, 2, H, W), "flow는 (B,2,H,W) 형태여야 합니다." # (BF)CHW, C=2
    assert mask.shape == (B, 1, H, W), "mask는 (B,1,H,W) 형태여야 합니다." # (BF)CHW, C=1

    # (B,3,H,W) -> (B,H,W,3)
    image_bhwc = image.permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)
    # (B,2,H,W) -> (B,H,W,2)
    flow_bhwt = flow.permute(0, 2, 3, 1).contiguous()    # (B,H,W,2)
    # (B,1,H,W) -> (B,H,W)
    mask_bhw   = mask.view(B, H, W)                     # (B,H,W)

    # 나중에 scatter_add로 누적하기 위해 1D로 펼침
    # 소스 이미지 픽셀 값 (B*H*W, 3)
    image_flat = image_bhwc.view(-1, C_i)
    # 플로우 (B*H*W, 2)
    flow_flat  = flow_bhwt.view(-1, 2)
    # 마스크 (B*H*W,)
    mask_flat  = mask_bhw.view(-1)

    # 각 픽셀이 속한 (batch b, y, x) 좌표를 1D로 만들기
    b_grid = torch.arange(B, device=device).view(B,1,1).expand(-1,H,W)  # (B,H,W)
    y_grid = torch.arange(H, device=device).view(1,H,1).expand(B,-1,W)  
    x_grid = torch.arange(W, device=device).view(1,1,W).expand(B,H,-1)

    b_idx = b_grid.flatten()  # (B*H*W)
    y_idx = y_grid.flatten()
    x_idx = x_grid.flatten()

    # flow 적용 (x+dx, y+dy)
    dx = flow_flat[:, 0]
    dy = flow_flat[:, 1]
    tx = x_idx + dx  # float
    ty = y_idx + dy  # float

    # bilinear 보간을 위해 floor/ceil
    tx0 = tx.floor().long()
    tx1 = tx0 + 1
    ty0 = ty.floor().long()
    ty1 = ty0 + 1

    alpha = tx - tx.floor()  # (B*H*W)
    beta  = ty - ty.floor()

    # 유효 범위 & mask
    valid = ((mask_flat == 1) &
             (tx0 >= 0) & (tx1 < W) &
             (ty0 >= 0) & (ty1 < H))
    valid_idx = valid.nonzero(as_tuple=True)  # (N,)

    # 필요한 부분만 인덱싱
    v_b = b_idx[valid_idx]     # (N,)
    v_x0 = tx0[valid_idx]
    v_x1 = tx1[valid_idx]
    v_y0 = ty0[valid_idx]
    v_y1 = ty1[valid_idx]
    v_alpha = alpha[valid_idx]
    v_beta  = beta[valid_idx]
    v_src   = image_flat[valid_idx]  # (N,3)

    # bilinear 가중치
    w00 = (1 - v_alpha) * (1 - v_beta)
    w01 = v_alpha       * (1 - v_beta)
    w10 = (1 - v_alpha) * v_beta
    w11 = v_alpha       * v_beta

    # 최종 결과 (B,H,W,3)와 가중치맵 (B,H,W)
    warped_bhwc = torch.zeros_like(image_bhwc)  # (B,H,W,3)
    weight_map  = torch.zeros((B, H, W), dtype=image.dtype, device=device)

    # 다시 (B*H*W)로 펼침
    warped_flat = warped_bhwc.view(-1, C_i)  # (B*H*W,3)
    weight_flat = weight_map.view(-1)        # (B*H*W,)

    # (b, y, x)를 (B,H,W) 1D 인덱스로 변환
    # offset_b = b*(H*W), 그 후 y*W + x
    def flatten_index(b, y, x):
        return b*(H*W) + (y * W) + x

    i00 = flatten_index(v_b, v_y0, v_x0)
    i01 = flatten_index(v_b, v_y0, v_x1)
    i10 = flatten_index(v_b, v_y1, v_x0)
    i11 = flatten_index(v_b, v_y1, v_x1)

    # scatter_add로 누적
    warped_flat.index_add_(0, i00, w00.unsqueeze(-1) * v_src)
    warped_flat.index_add_(0, i01, w01.unsqueeze(-1) * v_src)
    warped_flat.index_add_(0, i10, w10.unsqueeze(-1) * v_src)
    warped_flat.index_add_(0, i11, w11.unsqueeze(-1) * v_src)

    weight_flat.index_add_(0, i00, w00)
    weight_flat.index_add_(0, i01, w01)
    weight_flat.index_add_(0, i10, w10)
    weight_flat.index_add_(0, i11, w11)

    # 누적된 값을 weight로 나누어 최종 색상 확정
    w_valid = (weight_flat > 0)
    warped_flat[w_valid] /= weight_flat[w_valid].unsqueeze(-1)

    # (B,H,W,3)로 복원 후, (B,3,H,W)로 permute
    warped_bhwc = warped_flat.view(B, H, W, C_i)
    warped = warped_bhwc.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)
    
    return warped

    
def run_filtering(flow_f, flow_b, cycle_th=3.):
    """
    Args:
        flow_f: b 2 h w
        flow_b: b 2 h w
        cycle_th: distance threshold for inconsistency (e.g., 3.0 pixel)
    Returns:
        valid_mask: binary mask (0: Not consistent or 1: consistent), float, [b 1 h w]
    """
    assert flow_f.ndim == 4 and flow_b.ndim == 4
    
    device = flow_f.device
    h, w = flow_f.shape[-2:]
    num_imgs = flow_f.shape[0]
    
    flow_f = flow_f
    flow_b = flow_b
    
    grid = repeat(gen_grid(h, w, device=device).permute(2, 0, 1)[None], "b c h w -> (b v) c h w", v=num_imgs)
    
    coord2 = flow_f + grid
    coord2_normed = normalize_coords(coord2.permute(0, 2, 3, 1), h, w)
    flow_21_sampled = F.grid_sample(flow_b, coord2_normed, align_corners=True)
    map_i = flow_f + flow_21_sampled
    fb_discrepancy = torch.norm(map_i.squeeze(), dim=1)
    valid_mask = fb_discrepancy < cycle_th
    
    return valid_mask.unsqueeze(1).float()


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
    
#--------------------------------------------------------------------------------------------------------------
# Codes borrowed from https://github.com/ChristophReich1996/Optical-Flow-Visualization-PyTorch

from typing import Optional, Union

import torch
from math import pi as PI


def get_color_wheel(device: torch.device) -> torch.Tensor:
    """
    Generates the color wheel.
    :param device: (torch.device) Device to be used
    :return: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    """
    # Set constants
    RY: int = 15
    YG: int = 6
    GC: int = 4
    CB: int = 11
    BM: int = 13
    MR: int = 6
    # Init color wheel
    color_wheel: torch.Tensor = torch.zeros((RY + YG + GC + CB + BM + MR, 3), dtype=torch.float32)
    # Init counter
    counter: int = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter: int = counter + RY
    # YG
    color_wheel[counter:counter + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    color_wheel[counter:counter + YG, 1] = 255
    counter: int = counter + YG
    # GC
    color_wheel[counter:counter + GC, 1] = 255
    color_wheel[counter:counter + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    counter: int = counter + GC
    # CB
    color_wheel[counter:counter + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    color_wheel[counter:counter + CB, 2] = 255
    counter: int = counter + CB
    # BM
    color_wheel[counter:counter + BM, 2] = 255
    color_wheel[counter:counter + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    counter: int = counter + BM
    # MR
    color_wheel[counter:counter + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    color_wheel[counter:counter + MR, 0] = 255
    # To device
    color_wheel: torch.Tensor = color_wheel.to(device)
    return color_wheel


def _flow_hw_to_color(flow_vertical: torch.Tensor, flow_horizontal: torch.Tensor,
                      color_wheel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Private function applies the flow color wheel to flow components (vertical and horizontal).
    :param flow_vertical: (torch.Tensor) Vertical flow of the shape [height, width]
    :param flow_horizontal: (torch.Tensor) Horizontal flow of the shape [height, width]
    :param color_wheel: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    :param: device: (torch.device) Device to be used
    :return: (torch.Tensor) Visualized flow of the shape [3, height, width]
    """
    # Get shapes
    _, height, width = flow_vertical.shape
    # Init flow image
    flow_image: torch.Tensor = torch.zeros(3, height, width, dtype=torch.float32, device=device)
    # Get number of colors
    number_of_colors: int = color_wheel.shape[0]
    # Compute norm, angle and factors
    flow_norm: torch.Tensor = (flow_vertical ** 2 + flow_horizontal ** 2).sqrt()
    angle: torch.Tensor = torch.atan2(- flow_vertical, - flow_horizontal) / PI
    fk: torch.Tensor = (angle + 1.) / 2. * (number_of_colors - 1.)
    k0: torch.Tensor = torch.floor(fk).long()
    k1: torch.Tensor = k0 + 1
    k1[k1 == number_of_colors] = 0
    f: torch.Tensor = fk - k0
    # Iterate over color components
    for index in range(color_wheel.shape[1]):
        # Get component of all colors
        tmp: torch.Tensor = color_wheel[:, index]
        # Get colors
        color_0: torch.Tensor = tmp[k0] / 255.
        color_1: torch.Tensor = tmp[k1] / 255.
        # Compute color
        color: torch.Tensor = (1. - f) * color_0 + f * color_1
        # Get color index
        color_index: torch.Tensor = flow_norm <= 1
        # Set color saturation
        color[color_index] = 1 - flow_norm[color_index] * (1. - color[color_index])
        color[~color_index] = color[~color_index] * 0.75
        # Set color in image
        flow_image[index] = torch.floor(255 * color)
    return flow_image


def flow_to_color(flow: torch.Tensor, clip_flow: Optional[Union[float, torch.Tensor]] = None, 
                  normalize_over_video: bool = False) -> torch.Tensor:
    """
    Function converts a given optical flow map into the classical color schema.
    :param flow: (torch.Tensor) Optical flow tensor of the shape [batch size (optional), 2, height, width].
    :param clip_flow: (Optional[Union[float, torch.Tensor]]) Max value of flow values for clipping (default None).
    :param normalize_over_video: (bool) If true scale is normalized over the whole video (batch).
    :return: (torch.Tensor) Flow visualization (float tensor) with the shape [batch size (if used), 3, height, width].
    """
    # Check parameter types
    assert torch.is_tensor(flow), "Given flow map must be a torch.Tensor, {} given".format(type(flow))
    assert torch.is_tensor(clip_flow) or isinstance(clip_flow, float) or clip_flow is None, \
        "Given clip_flow parameter must be a float, a torch.Tensor, or None, {} given".format(type(clip_flow))
    # Check shapes
    assert flow.ndimension() in [3, 4], \
        "Given flow must be a 3D or 4D tensor, given tensor shape {}.".format(flow.shape)
    if torch.is_tensor(clip_flow):
        assert clip_flow.ndimension() == 0, \
            "Given clip_flow tensor must be a scalar, given tensor shape {}.".format(clip_flow.shape)
    # Manage batch dimension
    batch_dimension: bool = True
    if flow.ndimension() == 3:
        flow = flow[None]
        batch_dimension: bool = False
    # Save shape
    batch_size, _, height, width = flow.shape
    # Check flow dimension
    assert flow.shape[1] == 2, "Flow dimension must have the shape 2 but tensor with {} given".format(flow.shape[1])
    # Save device
    device: torch.device = flow.device
    # Clip flow if utilized
    if clip_flow is not None:
        flow = flow.clip(max=clip_flow)
    # Get horizontal and vertical flow
    flow_vertical: torch.Tensor = flow[:, 0:1]
    flow_horizontal: torch.Tensor = flow[:, 1:2]
    # Get max norm of flow
    flow_max_norm: torch.Tensor = (flow_vertical ** 2 + flow_horizontal ** 2).sqrt().view(batch_size, -1).max(dim=-1)[0]
    flow_max_norm: torch.Tensor = flow_max_norm.view(batch_size, 1, 1, 1)
    if normalize_over_video:
        flow_max_norm: Tensor = flow_max_norm.max(dim=0, keepdim=True)[0]
    # Normalize flow
    flow_vertical: torch.Tensor = flow_vertical / (flow_max_norm + 1e-05)
    flow_horizontal: torch.Tensor = flow_horizontal / (flow_max_norm + 1e-05)
    # Get color wheel
    color_wheel: torch.Tensor = get_color_wheel(device=device)
    # Init flow image
    flow_image = torch.zeros(batch_size, 3, height, width, device=device)
    # Iterate over batch dimension
    for index in range(batch_size):
        flow_image[index] = _flow_hw_to_color(flow_vertical=flow_vertical[index],
                                              flow_horizontal=flow_horizontal[index], color_wheel=color_wheel,
                                              device=device)
    return flow_image if batch_dimension else flow_image[0]