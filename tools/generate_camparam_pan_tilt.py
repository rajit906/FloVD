import os
import argparse
import numpy as np
from einops import rearrange, repeat
import pdb


def generate_rotation_extrinsics(direction: str, angle: float, num_frame: int):
    """
    Generate extrinsic camera matrices with rotation only (no translation),
    allowing both positive and negative directions.

    Args:
        direction (str): '+x', '-x', '+y', '-y', '+z', or '-z'
        angle (float): total rotation angle in degrees
        num_frame (int): number of frames to interpolate the rotation

    Returns:
        List[np.ndarray]: List of 3x4 extrinsic matrices (rotation | zero translation)
    """
    assert direction[0] in ('+', '-'), "direction must start with '+' or '-'"
    assert direction[1] in ('x', 'y', 'z'), "direction must be along x, y, or z"

    axis = direction[1]
    sign = 1 if direction[0] == '+' else -1
    angle_rad = np.deg2rad(angle) * sign
    step = angle_rad / (num_frame - 1)

    extrinsics = []
    for i in range(num_frame):
        theta = step * i
        print(theta)
        if axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ])
        elif axis == 'y':
            R = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ])
        elif axis == 'z':
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ])

        Rt = np.hstack([R, np.zeros((3, 1))])  # 3x4 matrix
        extrinsics.append(Rt)

    extrinsics = np.stack(extrinsics)
    
    K = np.array([0.474812, 0.844111, 0.500000, 0.500000, abs(0.000000), abs(0.000000)])
    camparam = np.concatenate([repeat(K, 'n -> f n', f=num_frame), rearrange(extrinsics, 'f h w -> f (h w)')], axis=-1)

    return camparam

def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    angle = 90

    # tilt up
    direction = ('+', 'x')
    camparam_tilt_up = generate_rotation_extrinsics(direction, angle, args.num_frame).astype(np.float32)
    np.savetxt(os.path.join(args.output_path, f'Tilt_Up_{angle:01f}.txt'), camparam_tilt_up, fmt='%1.6f')
    
    # tilt down
    direction = ('-', 'x')
    camparam_tilt_down = generate_rotation_extrinsics(direction, angle, args.num_frame).astype(np.float32)
    np.savetxt(os.path.join(args.output_path, f'Tilt_Down_{angle:01f}.txt'), camparam_tilt_down, fmt='%1.6f')

    # pan right
    direction = ('+', 'y')
    camparam_pan_right = generate_rotation_extrinsics(direction, angle, args.num_frame).astype(np.float32)
    np.savetxt(os.path.join(args.output_path, f'Pan_Right_{angle:01f}.txt'), camparam_pan_right, fmt='%1.6f')
    
    # pan left
    direction = ('-', 'y')
    camparam_pan_left = generate_rotation_extrinsics(direction, angle, args.num_frame).astype(np.float32)
    np.savetxt(os.path.join(args.output_path, f'Pan_Left_{angle:01f}.txt'), camparam_pan_left, fmt='%1.6f')
    
    # Spin clockwise
    direction = ('+', 'z')
    camparam_spin_clockwise = generate_rotation_extrinsics(direction, angle, args.num_frame).astype(np.float32)
    np.savetxt(os.path.join(args.output_path, f'Spin_Clockwise_{angle:01f}.txt'), camparam_spin_clockwise, fmt='%1.6f')
    
    # Spin anticlockwise
    direction = ('-', 'z')
    camparam_spin_anticlockwise = generate_rotation_extrinsics(direction, angle, args.num_frame).astype(np.float32)
    np.savetxt(os.path.join(args.output_path, f'Spin_AntiClockwise_{angle:01f}.txt'), camparam_spin_anticlockwise, fmt='%1.6f')






    # right
    # direction = [1., abs(0.), abs(0.)]
    # camparam_right = translation_matrix(direction, length, args.num_frame).astype(np.float32)
    # np.savetxt(os.path.join(args.output_path, 'camera_R.txt'), camparam_right, fmt='%1.6f')

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_frame", type=int, default=49)

    args = parser.parse_args()
    main(args)
    