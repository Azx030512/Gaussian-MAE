import numpy as np

def sample_camera_on_unit_sphere():
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z])  # 相机位置（cam_pos）

# def look_at(cam_pos, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])):
#     forward = target - cam_pos
#     forward = forward / np.linalg.norm(forward)

#     right = np.cross(up, forward)
#     right = right / np.linalg.norm(right)

#     up_new = np.cross(forward, right)
#     up_new = up_new / np.linalg.norm(up_new)

#     R = np.stack([right, forward, up_new], axis=1)  # x: right, y: forward, z: up

#     t = -R.T @ cam_pos 

#     extrinsic = np.eye(4)
#     extrinsic[:3, :3] = R.T
#     extrinsic[:3, 3] = t

#     return extrinsic



def look_at_gaussian(cam_pos, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])):
    """
    Generate a camera-to-world (c2w) extrinsic matrix under the 3D Gaussian / OpenGL coordinate system (Z backward, Y up).
    Args:
        cam_pos: Camera position in world coordinates, shape (3,)
        target: The point the camera is looking at. Default is the origin.
        up: The up direction vector. Default is the Y-axis.

    Returns:
        c2w: (4, 4) camera-to-world transformation matrix, directly usable for Gaussian rendering.
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_new = np.cross(right, forward)
    up_new = up_new / np.linalg.norm(up_new)
    # Note: OpenGL/Blender uses a right-handed coordinate system with the Z-axis pointing backward,
    # so we negate the forward vector to match that convention.
    R = np.stack([right, up_new, -forward], axis=1)  # camera-to-world rotation matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos

    return c2w


def align_extrinsic_to_gaussian(extrinsic):
    """
    Due to Gaussian utilized transform = [[1,0,0],[0,0,-1],[0,1,0]] while saving as ply files.
    We adopt a reverse transform for the generated open3d cameras.
    """
    inverse_transform = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    T = np.eye(4)
    T[:3, :3] = inverse_transform
    return T @ extrinsic

if __name__ == '__main__':
    cam_pos = sample_camera_on_unit_sphere() * 3.0
    extrinsic = look_at_gaussian(cam_pos)

    print("Camera position:", cam_pos)
    print("Extrinsic matrix:\n", extrinsic)
