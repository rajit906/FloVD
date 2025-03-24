import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import glob
from PIL import Image

from torch.utils.data.dataset import Dataset
from packaging import version as pver

from decord import VideoReader
from flovd.data.cam_dummies import SampleManualCam

import pdb

class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


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


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker



class VideoDataset(Dataset):
    def __init__(
            self,
            root_path,
            metadata_path,
            sample_stride=4,
            minimum_sample_stride=1,
            sample_n_frames=16,
            relative_pose=False,
            zero_t_first_frame=False,
            sample_size=[256, 384],
            rescale_fxy=False,
            use_flip=False,
            return_clip_name=False,
            max_scenes=None,
            use_sampled_validation=False,
            frame_num_sampled_validation=8,
            camera_root_path='./assets/manual_poses',
            speed=0.25,
    ):
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_clip_name = return_clip_name
        self.use_sampled_validation = use_sampled_validation
        self.frame_num_sampled_validation = frame_num_sampled_validation
        self.speed = speed
        
        # Read metadata
        print(f"Read metadata from {metadata_path}")
        if os.path.splitext(metadata_path)[-1] == '.txt':
            f = open(metadata_path, 'r')
            lines = f.readlines()
            f.close()
            
            # Gather video paths
            scene_list = [line.split('\t')[0] for line in lines]
            scene_path_list = sorted([os.path.join(self.root_path, path) for path in scene_list])
        elif os.path.splitext(metadata_path)[-1] == '.json':
            with open(metadata_path, "r") as st_json:
                metadata = json.load(st_json)

            # Gather video paths
            scene_list = [k for k, v in metadata.items()]
            scene_path_list = sorted([os.path.join(self.root_path, path) for path in scene_list])
        else:
            raise NotImplementedError("Choose metadata type between 'txt' and 'json' ")
        
        if max_scenes is not None:
            scene_path_list = sorted(random.sample(scene_path_list, k=max_scenes))

        if use_sampled_validation:
            # use last 16 scenes as validation set
            scene_path_list = scene_path_list[-frame_num_sampled_validation:]
        
        
        self.scene_path_list = scene_path_list
        
        self.length = len(self.scene_path_list)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [transforms.Resize(sample_size),
                                RandomHorizontalFlipWithPose(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.use_flip = use_flip
        
        self.PoseSampler = SampleManualCam(camera_root_path)
        
        # for _ in range(10):
        #     self.__getitem__(random.randint(0, self.length - 1))

    def load_video_reader(self, idx):
        video_path = self.scene_path_list[idx]

        # video_path = os.path.join(self.root_path, video_dict['clip_path'])
        # pdb.set_trace()
        video_reader = VideoReader(video_path)
        return video_reader

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
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


    def load_cameras(self, path):       
        with open(path, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def load_images(self, scene_path, indices):
        rgb_path = sorted(glob.glob(os.path.join(scene_path, "*.png")))
        rgb_path = np.array(rgb_path)[indices]
        image_pil = [Image.open(x) for x in rgb_path]
        image_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32)) for x in image_pil]).permute(0, 3, 1, 2).contiguous()
        image_tensor = image_tensor / 255.
        return image_tensor, image_pil

    def get_batch(self, idx):
        # scene_path = self.scene_path_list[idx]
        video_reader = self.load_video_reader(idx)
        
        total_frames = len(video_reader)
        assert total_frames >= (self.sample_n_frames+1)
        
        current_sample_stride = self.sample_stride

        if total_frames < (self.sample_n_frames+1) * current_sample_stride:
            maximum_sample_stride = int(total_frames // (self.sample_n_frames+1))
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= (self.sample_n_frames+1)
        frame_indices_total = np.linspace(start_frame_ind, end_frame_ind - 1, (self.sample_n_frames+1), dtype=int)
        condition_image_ind = frame_indices_total[:1]
        frame_indices = frame_indices_total[1:]
        
        
        
        # image_tensor_all, image_pil_all = self.load_images(scene_path, frame_indices_total)
        image_tensor_all = torch.from_numpy(video_reader.get_batch(frame_indices_total).asnumpy()).permute(0, 3, 1, 2).contiguous() / 255.
        condition_image = image_tensor_all[0:1]
        pixel_values = image_tensor_all[1:]
        
        
        
        poses, cam_name = self.PoseSampler.sample()
        pose_init = poses[0]
        # pose_init[7:] = ['1.000000', '0.000000', '0.000000', '0.000000', '0.000000', '1.000000', '0.000000', '0.000000', '0.000000', '0.000000', '1.000000', '0.000000']
        # pose_init[7:] = [1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
        # poses.insert(0, pose_init)
        cam_params = poses
        
        # poses_dummy = [1000000, 0.474812461, 0.844111024, 0.500000000, 0.500000000, 0.000000000, 0.000000000, 0.683357954, 0.072232656, -0.726501465, 1.272327830, -0.075338788, 0.996758103, 0.028238246, 0.189685652, 0.726185918, 0.035436906, 0.686584473, -1.134427327]
        # poses_dummy = [float(x) for x in poses_dummy]
        # cam_params = [poses_dummy]*len(frame_indices_total)
        # pdb.set_trace()
        
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        
        
        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:                                          # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w_poses[:, :3, -1] = c2w_poses[:, :3, -1] * self.speed
        
        c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device=c2w.device)
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()

        video_caption, clip_name = cam_name, f'{idx:06d}'
        # video_caption, clip_name = 'No_caption', f'{idx:06d}'

        return pixel_values, condition_image, plucker_embedding, video_caption, flip_flag, clip_name, intrinsics, c2w

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            # video, condition_image, plucker_embedding, video_caption, flip_flag, clip_name, intrinsics, c2w = self.get_batch(idx)
            try:
                video, condition_image, plucker_embedding, video_caption, flip_flag, clip_name, intrinsics, c2w = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            for transform in self.pixel_transforms[2:]:
                video = transform(video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
        for transform in self.pixel_transforms:
            condition_image = transform(condition_image)
        if self.return_clip_name:
            sample = dict(pixel_values=video, condition_image=condition_image, plucker_embedding=plucker_embedding, video_caption=video_caption, intrinsics=intrinsics, c2w=c2w, clip_name=clip_name)
        else:
            sample = dict(pixel_values=video, condition_image=condition_image, plucker_embedding=plucker_embedding, video_caption=video_caption, intrinsics=intrinsics, c2w=c2w)

        # pixel_values, condition_image ~ [-1,1]

        return sample