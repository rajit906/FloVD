import numpy as np
from glob import glob
import random
import os
import pdb
random.seed(7777)

class SampleManualCam:
    def __init__(
        self,
        root_path = './assets/manual_poses',
    ):
        self.root_path = root_path
        self.MANUAL_CAM = ['I', 'D', 'L', 'O', 'R', 'Round-30', 'SPIN_CW-30', 'Stop', 'U']
        # self.pose_path = glob(root_path, "*.txt")
    
    def sample(self):
        cam_name = random.choice(self.MANUAL_CAM)
        path = os.path.join(self.root_path, f"test_camera_{cam_name}.txt")
        with open(path, 'r') as f:
            poses = f.readlines()
        
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        poses = [[float(x) for x in pose] for pose in poses]
        
        return poses, cam_name