import numpy as np
from glob import glob
import random
import os
import pdb
random.seed(7777)

class SampleManualCam:
    def __init__(
        self,
        pose_type = 'manual',
        root_path = '../assets/manual_poses',
    ):
        self.root_path = root_path
        if pose_type == 'manual':
            self.MANUAL_CAM = ['I', 'D', 'L', 'O', 'R', 'U']
        elif pose_type == 're10k':
            self.RE10K_CAM = os.listdir(root_path)
        # self.pose_path = glob(root_path, "*.txt")
        
        self.pose_type = pose_type
    
    def sample(self, order=None, name=None):
        # Sample camera parameters (W2C)
        
        if self.pose_type == 'manual':
            if name is not None:
                assert name in self.MANUAL_CAM
                cam_name = name
            elif order is not None:
                order = order % len(self.MANUAL_CAM)
                cam_name = self.MANUAL_CAM[order]
            else:
                cam_name = random.choice(self.MANUAL_CAM)
            path = os.path.join(self.root_path, f"camera_{cam_name}.txt")
        elif self.pose_type == 're10k':
            if name is not None:
                assert name in self.RE10K_CAM
                cam_name = name
            elif order is not None:
                order = order % len(self.RE10K_CAM)
                cam_name = self.RE10K_CAM[order]
            else:
                cam_name = random.choice(self.RE10K_CAM)
            path = os.path.join(self.root_path, cam_name)
        with open(path, 'r') as f:
            poses = f.readlines()
        
        poses = [pose.strip().split(' ') for pose in poses]
        poses = [[float(x) for x in pose] for pose in poses]
        
        return poses, cam_name