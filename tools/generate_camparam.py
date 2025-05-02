import os
import argparse
import numpy as np
from einops import rearrange, repeat
import pdb

# Generate W2C extrinsic parameters

def translation_matrix(direction, length, num_frame):
    assert len(direction)==3, "direction should be [1, 0, 0] or [0, 1, 0] or ..."

    K = np.array([0.474812, 0.844111, 0.500000, 0.500000, abs(0.000000), abs(0.000000)])
    R = np.array([[1.0, abs(0.0), abs(0.0)],
                  [abs(0.0), 1.0, abs(0.0)],
                  [abs(0.0), abs(0.0), 1.0]])
    
    T = (repeat(np.array(direction), 'n -> n f', f=num_frame) * np.linspace(abs(0.), length, num_frame)).transpose(1,0)
    extrinsic = np.concatenate([repeat(R, 'h w -> f h w', f=num_frame), T[:,:,None]], axis=-1)
    camparam = np.concatenate([repeat(K, 'n -> f n', f=num_frame), rearrange(extrinsic, 'f h w -> f (h w)')], axis=-1)

    return camparam

def main(args):
    os.makedirs(args.output_path, exist_ok=True)


    length = 1.5

    # right
    direction = [1., abs(0.), abs(0.)]
    camparam_right = translation_matrix(direction, length, args.num_frame).astype(np.float32)
    np.savetxt(os.path.join(args.output_path, 'camera_R.txt'), camparam_right, fmt='%1.6f')

    # left
    direction = [-1., abs(0.), abs(0.)]
    camparam_left = translation_matrix(direction, length, args.num_frame)
    np.savetxt(os.path.join(args.output_path, 'camera_L.txt'), camparam_left, fmt='%1.6f')

    # up
    direction = [abs(0.), -1.0, abs(0.)]
    camparam_up = translation_matrix(direction, length, args.num_frame)
    np.savetxt(os.path.join(args.output_path, 'camera_U.txt'), camparam_up, fmt='%1.6f')

    # down
    direction = [abs(0.), 1.0, abs(0.)]
    camparam_down = translation_matrix(direction, length, args.num_frame)
    np.savetxt(os.path.join(args.output_path, 'camera_D.txt'), camparam_down, fmt='%1.6f')

    # in
    direction = [abs(0.), abs(0.), 1.0]
    camparam_in = translation_matrix(direction, length, args.num_frame)
    np.savetxt(os.path.join(args.output_path, 'camera_I.txt'), camparam_in, fmt='%1.6f')

    # out
    direction = [abs(0.), abs(0.), -1.0]
    camparam_out = translation_matrix(direction, length, args.num_frame)
    np.savetxt(os.path.join(args.output_path, 'camera_O.txt'), camparam_out, fmt='%1.6f')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_frame", type=int, default=49)

    args = parser.parse_args()
    main(args)
    