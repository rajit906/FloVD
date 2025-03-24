import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from einops import rearrange, repeat

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def load_cameras(path):       
    with open(path, 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    cam_params = [Camera(cam_param) for cam_param in cam_params]
    return cam_params

def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    cam_to_origin = 0
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

def get_K(intrinsics, size, do_normalize=False):
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
    if do_normalize:
        K = normalize_intrinsic(K, size)
    
    return K