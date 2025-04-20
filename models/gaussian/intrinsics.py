import numpy as np

def get_perspective_intrinsics(fov_deg=30.0, aspect=1.0, near=0.1, far=2.0) -> np.ndarray:

    fov_rad = np.deg2rad(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)

    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0

    return proj
if __name__ == '__main__':
    intrinsic = get_perspective_intrinsics(fov_deg=30.0, aspect=1.0)
    print("Projection (intrinsic-like) matrix:\n", intrinsic)
