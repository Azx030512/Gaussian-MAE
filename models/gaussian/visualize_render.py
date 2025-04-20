import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import torch
from tqdm import tqdm
from gaussian_model import Gaussian, represent_config
from gaussian_render import GaussianRenderer
from intrinsics import get_perspective_intrinsics
from extrinsics import sample_camera_on_unit_sphere, look_at_gaussian, align_extrinsic_to_gaussian
from torchvision.utils import save_image
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sphere_pose import create_spheric_poses

def create_gaussian_spheres(gaussian_xyz, radius=0.01, color=[1, 0, 0]):
    spheres = []
    for pt in gaussian_xyz:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres

def create_camera_frame(extrinsic, size=0.1):
    """
    extrinsic: [4, 4] torch.Tensor or numpy.ndarray
    returns: o3d.geometry.TriangleMesh (coordinate frame)
    """
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # 创建坐标系
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.rotate(R, center=(0, 0, 0))
    frame.translate(t)
    return frame

def visualize_gaussians_with_cameras(ply_paths, extrinsics):
    vis_objs = []
    for ply_path in ply_paths:
        gaussian = Gaussian(
            sh_degree=0,
            aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
            mininum_kernel_size=0,
            scaling_bias=0,
            opacity_bias=0,
            scaling_activation='exp'
        )
        gaussian.load_ply(ply_path)

        xyz = gaussian.get_xyz.detach().cpu().numpy()
        spheres = create_gaussian_spheres(xyz, radius=0.01, color=[0.7, 0.1, 0.1])
        vis_objs.extend(spheres)

    for extrinsic in extrinsics:
        cam_frame = create_camera_frame(extrinsic, size=0.1)
        vis_objs.append(cam_frame)

    # 显示
    o3d.visualization.draw_geometries(vis_objs)

def draw_gaussian_with_camera_matplotlib(gaussians: list, cam_poses: torch.Tensor, output_path="gaussian_vis.png"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 gaussian 点云
    for gaussian in gaussians:
        xyz = gaussian.get_xyz.detach().cpu().numpy()
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c='green', label='Gaussian')

    # 绘制相机坐标系（只显示前几个）
    for i in range(min(cam_poses.shape[0], 10)):
        T = cam_poses[i].detach().cpu().numpy()  # [4, 4]
        origin = T[:3, 3]
        x_axis = T[:3, 0] * 0.1
        y_axis = T[:3, 1] * 0.1
        z_axis = T[:3, 2] * 0.1

        ax.quiver(*origin, *x_axis, color='r', linewidth=1)  # X
        ax.quiver(*origin, *y_axis, color='g', linewidth=1)  # Y
        ax.quiver(*origin, *z_axis, color='b', linewidth=1)  # Z

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Gaussian Point Cloud with Camera Poses")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved matplotlib render to {output_path}")

def draw_camera(ax, extrinsic, fov_deg=30, scale=0.2, color='black'):
    # 将 extrinsic 转成 np 数组
    T = extrinsic
    R = T[:3, :3]
    t = T[:3, 3]

    # 坐标轴
    ax.quiver(*t, *R[:, 0] * scale, color='r')  # X
    ax.quiver(*t, *R[:, 1] * scale, color='g')  # Y
    ax.quiver(*t, *R[:, 2] * scale, color='b')  # Z

    # 构建视锥体
    half_fov = np.radians(fov_deg / 2)
    depth = scale * 2
    aspect = 1.0
    h = np.tan(half_fov) * depth
    w = h * aspect

    # 四个视锥角落点（相机坐标系下）
    corners_cam = np.array([
        [ w,  h, depth],
        [-w,  h, depth],
        [-w, -h, depth],
        [ w, -h, depth]
    ]).T  # [3, 4]

    # 转成世界坐标
    corners_world = (R @ corners_cam) + t[:, None]  # [3, 4]

    # 绘制锥体边
    for i in range(4):
        ax.plot([t[0], corners_world[0, i]], [t[1], corners_world[1, i]], [t[2], corners_world[2, i]], color=color)
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([corners_world[0, i], corners_world[0, j]],
                [corners_world[1, i], corners_world[1, j]],
                [corners_world[2, i], corners_world[2, j]], color=color)


def draw_gaussian_and_camera(gaussian, extrinsic, idx=0, output_path_prefix="gaussian_camera"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    xyz = gaussian.get_xyz.detach().cpu().numpy()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c='green', alpha=0.6)
    draw_camera(ax, extrinsic, fov_deg=60, scale=0.2)
    cam_position = extrinsic[:3, 3]
    origin = np.array([0.0, 0.0, 0.0])
    ax.plot(
        [cam_position[0], origin[0]],
        [cam_position[1], origin[1]],
        [cam_position[2], origin[2]],
        color='red',
        linewidth=1,
        linestyle='--',
        label='Camera to Origin'
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(f"Gaussian + Camera {idx}")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    # plt.tight_layout()

    output_path = f"{output_path_prefix}_{idx}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


reps = []
ply_paths = [
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_vis_gaussians.ply",
            "/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1acfbda4ce0ec524bedced414fad522f_original_gaussians.ply",
             ]
for ply_path in ply_paths:
    gaussian = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=represent_config['3d_filter_kernel_size'],
        scaling_bias=represent_config['scaling_bias'],
        opacity_bias=represent_config['opacity_bias'],
        scaling_activation=represent_config['scaling_activation']
    )
    gaussian.load_ply(ply_path)
    reps.append(gaussian)


rendering_options = {"resolution": 400, "near": 0.01, "far": 2.1, "bg_color": [1, 1, 1]}  # 'random'
renderer = GaussianRenderer(rendering_options)
renderer.pipe.kernel_size = represent_config['2d_filter_kernel_size']


# intrinsic = np.eye(4)
# intrinsic[0, 0] = 600
# intrinsic[1, 1] = 600
# intrinsic[0, 2] = 400
# intrinsic[1, 2] = 400

intrinsic = get_perspective_intrinsics(fov_deg=80, aspect=1.0)

intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
intrinsic = intrinsic.cuda()

# extrinsics = []
# for i in range(len(reps)):
#     # cam_pos = sample_camera_on_unit_sphere() * 2.0
#     cam_pos = np.array([0.2, 0.2, 0.2])
#     extrinsic = look_at_gaussian(cam_pos)
#     # extrinsic = align_extrinsic_to_gaussian(extrinsic)
#     extrinsics.append(torch.tensor(extrinsic, dtype=torch.float32).cuda()[None,...])
# extrinsics = torch.concat(extrinsics, dim=0)

cams_pose = create_spheric_poses(1.0, [0, 0, 0])
# extrinsics = np.concatenate([cams_pose, np.array([[[0,0,0,1]]]).repeat(cams_pose.shape[0],axis=0)], axis=1)
# extrinsics = torch.tensor(extrinsics, dtype=torch.float32, device='cuda')



# Step 1: 加齐次项变成 (N, 4, 4)
cams_pose_homo = np.concatenate([
    cams_pose,
    np.array([[[0, 0, 0, 1]]]).repeat(cams_pose.shape[0], axis=0)
], axis=1)

# Step 2: 转换为 OpenGL 风格坐标系
# opencv_to_opengl = np.diag([1, 1, 1, 1])[None, ...]  # shape (1, 4, 4)
# cams_pose_gl = opencv_to_opengl @ cams_pose_homo  # (N, 4, 4)

# coord_transform = np.array([[1, 0, 0,0], [0, 0, -1, 0], [0, 1, 0, 0],[0,0,0,1]])  # flips y and z
# c2w_opengl = cams_pose_homo @ coord_transform.T

# # Step 3: 得到用于渲染的 extrinsics: world-to-camera
# w2c = np.linalg.inv(c2w_opengl)
w2c = np.linalg.inv(cams_pose_homo)
# extrinsics[:,:3,3] = cams_pose_homo[:,:3,3]
extrinsics = torch.tensor(w2c, device='cuda', dtype=torch.float32)



ret = None
representation = reps[0]
for i in tqdm(range(0,30)): # extrinsics.shape[0]
    # extrinsics_w2c = torch.inverse(extrinsics[i])
    # render_pack = renderer.render(representation, extrinsics_w2c, intrinsics[i])

    render_pack = renderer.render(representation, extrinsics[i], intrinsic)
    if ret is None:
        ret = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
    for k, v in render_pack.items():
        ret[k].append(v)
    ret['bg_color'].append(renderer.bg_color)

os.makedirs("./visualize_results", exist_ok=True)
for i, rendered_image in enumerate(ret['color']):
    save_image(rendered_image.clamp(0, 1), f"./visualize_results/rendered-{i}.png")

# draw_gaussian_with_camera_matplotlib(reps, extrinsics, output_path="render_matplotlib.png")

for i in range(len(reps)):
    draw_gaussian_and_camera(reps[i], cams_pose_homo[i], idx=i)

# visualize_gaussians_with_cameras(ply_paths, extrinsics)
