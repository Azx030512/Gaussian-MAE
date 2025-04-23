#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import copy
import json
import torch
import random
from ..utils.system_utils import searchForMaxIteration
from .dataset_readers import sceneLoadTypeCallbacks
from .gaussian_model import GaussianModel
from .deform_model import DeformModel
from ..arguments import ModelParams
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import numpy as np
from mediapy import read_video
import math
from PIL import Image
from typing import NamedTuple
from plyfile import PlyData, PlyElement


import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
    
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class CameraInfo4D(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    flow: np.array
    

def get_c2w_from_up_and_look_at(
    up,
    look_at,
    pos,
    opengl=False,
):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    c2w = np.zeros([4, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    c2w[3, 3] = 1.0

    # opencv to opengl
    if opengl:
        c2w[..., 1:3] *= -1

    return c2w


def get_uniform_poses(num_frames, radius, elevation, opengl=False):
    T = num_frames
    azimuths = np.deg2rad(np.linspace(0, 360, T + 1)[:T])
    elevations = np.full_like(azimuths, np.deg2rad(elevation))
    cam_dists = np.full_like(azimuths, radius)

    campos = np.stack(
        [
            cam_dists * np.cos(elevations) * np.cos(azimuths),
            cam_dists * np.cos(elevations) * np.sin(azimuths),
            cam_dists * np.sin(elevations),
        ],
        axis=-1,
    )

    center = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    poses = []
    for t in range(T):
        poses.append(get_c2w_from_up_and_look_at(up, center, campos[t], opengl=opengl))

    return np.stack(poses, axis=0)

def create_spheric_poses(radius, origin, n_poses=20, radius_level=3):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 0, 1, -t],  # 沿世界坐标系的-y移动
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([  # 绕世界坐标系的z轴旋转
            [np.cos(phi), -np.sin(phi), 0, 0],
            [np.sin(phi), np.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([  # 绕x轴
            [1, 0, 0, 0],
            [0, np.cos(-th), -np.sin(-th), 0],
            [0, np.sin(-th), np.cos(-th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_phi(phi) @ rot_theta(theta) @ trans_t(radius)
        return c2w

    spheric_poses = []
    origin_trans = np.eye(4)
    origin_trans[:3, 3] = np.array(origin)
    for level in range(radius_level):
        for ph in np.linspace(0, 2 * np.pi, n_poses // 2 + 1)[:-1]:  # 绕z轴360度
            for th in np.linspace(0, np.pi / 2, n_poses // 4 + 1)[:-1]:  # 绕x轴90度
                spheric_poses += [
                    (origin_trans @ spheric_pose(th, ph, radius * (level + 1)))]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def constructVideoNVSInfo4D(
    num_frames=21,
    radius=2.0,
    elevation=10.0,
    fov=33.8,
    reso=800,
    images=[],
    masks=[],
    num_pts=100_000,
    train=True,
):
    # test phygaussian ficus data
    radius = 4.11
    # radius = 1.0
    elevation = 8.96
    reso = 800
    focal = 1111.1110311937682
    fov = focal2fov(focal, reso)
    num_frames = 100
    

    poses = get_uniform_poses(num_frames, radius, elevation)
    w2cs = np.linalg.inv(poses)
    train_cam_infos = []

    print("radius: ", radius)
    print("elevation: ", elevation)
    print("fov: ", fov)
    print("num_frames: ", num_frames)
    print("poses.shape: ", poses.shape)
    print("reso: ", reso)
    
    def get_mp4_files(folder_path):
        mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        mp4_files.sort()
        mp4_paths = [os.path.join(folder_path, f) for f in mp4_files]
        return mp4_paths

    folder_path = '/mnt/nas_9/group/leiyuanhang/code/TimeGS/data/flower'  # 替换为包含 .mp4 文件的文件夹路径
    mp4_file_paths = get_mp4_files(folder_path)

    # for path in mp4_file_paths:
    #     print(path)
    
    # total_time = len(mp4_file_paths)
    total_time = 2

    for i in range(total_time):
        fid = i / total_time
        images = []
        frames = read_video(mp4_file_paths[i])
        for frame in frames:
            images.append(Image.fromarray(frame))
            
        for idx, pose in enumerate(w2cs):
            train_cam_infos.append(
                CameraInfo4D(
                    uid=idx,
                    R=np.transpose(pose[:3, :3]),
                    T=pose[:3, 3],
                    # FovY=np.deg2rad(fov),
                    # FovX=np.deg2rad(fov),
                    FovY=fov,
                    FovX=fov,
                    image=images[0],
                    image_path=None,
                    image_name=idx,
                    width=reso,
                    height=reso,
                    fid=fid,
                    flow=None,
                )
            )

    nerf_normalization = getNerfppNorm(train_cam_infos)
   
    # num_pts = 100_000
    # radius = radius
    num_pts = 50_000
    xyz = np.random.randn(num_pts, 3) * radius / 16
    xyz = np.random.randn(num_pts, 3) * radius / 16
    shs = np.ones((num_pts, 3)) * 0.2
   
    ply_path = "./tmp/points3d.ply"
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )
    
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    
    # pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=train_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path="./tmp/points3d.ply",
    )

    return scene_info





def constructPose3D(
    num_frames=21,
    radius=2.0,
    elevation=10.0,
    fov=33.8,
    reso=800,
    images=[],
    masks=[],
    num_pts=100_000,
    train=True,
    adjust_cam=False,
):
    # test phygaussian ficus data
    # radius = 4.11
    # radius = 1.0
    # elevation = 8.96
    reso = 800
    focal = 1111.1110311937682
    fov = focal2fov(focal, reso)
    num_frames = 100
    
    radius = radius
    evaluation = elevation
    
    poses = get_uniform_poses(num_frames, radius, elevation) # poses is c2w
    poses = create_spheric_poses(radius, [0,0,0], n_poses=20, radius_level=3)
    
    # print("poses shape: ", poses.shape)
    # print("poses[0] is: ", poses[0])

    if adjust_cam:
        poses_adjusted = []
        
        z_flip = np.array([
            [1,  0,  0,  0],  # x-axis stays the same
            [0,  1,  0,  0],  # y-axis stays the same
            [0,  0, -1,  0],  # z-axis is flipped
            [0,  0,  0,  1]   # Homogeneous coordinates unchanged
        ])
        
        for c2w in poses: 
            poses_adjusted.append(z_flip @ c2w)
        poses = np.stack(poses_adjusted, axis=0)
        print("adjusted poses shape: ", poses.shape)

    # print("poses[0] is: ", poses[0])


    w2cs = np.linalg.inv(poses)
    train_cam_infos = []

    # print("radius: ", radius)
    # print("elevation: ", elevation)
    # print("fov: ", fov)
    # print("num_frames: ", num_frames)
    # print("poses.shape: ", poses.shape)
    # print("reso: ", reso)
    

    fid = 0
    example_img = Image.open("/data/ckpt/aizixiang/Gaussian-MAE/visualize_results/rendered-0.png")
    
            
    for idx, pose in enumerate(w2cs):
        train_cam_infos.append(
            CameraInfo4D(
                uid=idx,
                R=np.transpose(pose[:3, :3]),
                T=pose[:3, 3],
                # FovY=np.deg2rad(fov),
                # FovX=np.deg2rad(fov),
                FovY=fov,
                FovX=fov,
                image=example_img,
                image_path=None,
                image_name=idx,
                width=reso,
                height=reso,
                fid=fid,
                flow=None,
            )
        )

    # nerf_normalization = getNerfppNorm(train_cam_infos)
   
    # num_pts = 50_000
    # xyz = np.random.randn(num_pts, 3) * radius / 16
    # xyz = np.random.randn(num_pts, 3) * radius / 16
    # shs = np.ones((num_pts, 3)) * 0.2
   
    # ply_path = "./tmp/points3d.ply"
    # pcd = BasicPointCloud(
    #     points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    # )
    
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # pcd = fetchPly(ply_path)

    
    # pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=train_cam_infos,
        test_cameras=train_cam_infos,
        nerf_normalization=None,
        ply_path=None,
    )

    return scene_info



def constructForceScene(
    num_frames=21,
    radius=2.0,
    elevation=10.0,
    fov=33.8,
    reso=800,
    images=[],
    masks=[],
    num_pts=100_000,
    train=True,
    start_time_train=0,
    train_folder_path=None,
    adjust_cam=False,
    render_view=None,
):
    radius = radius
    elevation = elevation
    reso = 800
    focal = 1111.1110311937682
    fov = focal2fov(focal, reso)
    num_frames = 100 # train on only one ref video

    poses_all = get_uniform_poses(num_frames, radius, elevation)
    
    poses = []
    poses.append(poses_all[render_view[0]])
    poses.append(poses_all[render_view[1]])
    poses.append(poses_all[render_view[2]])
    poses.append(poses_all[render_view[3]])
    poses = np.stack(poses, axis=0)
    
    test_poses = []
    test_poses.append(poses_all[render_view[0]])
    test_poses.append(poses_all[render_view[1]])
    test_poses.append(poses_all[render_view[2]])
    test_poses.append(poses_all[render_view[3]])
    test_poses.append(poses_all[render_view[4]])
    test_poses.append(poses_all[render_view[5]])
    test_poses.append(poses_all[render_view[6]])
    test_poses.append(poses_all[render_view[7]])

    w2cs = np.linalg.inv(poses)
    train_cam_infos = []
    
    w2cs_test = np.linalg.inv(test_poses)
    test_cam_infos = []

    print("radius: ", radius)
    print("elevation: ", elevation)
    print("fov: ", fov)
    print("num_frames: ", num_frames)
    print("poses.shape: ", poses.shape)
    print("reso: ", reso)
    
    def get_mp4_files(folder_path):
        mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        mp4_files.sort()
        mp4_paths = [os.path.join(folder_path, f) for f in mp4_files]
        return mp4_paths
    
    folder_path = train_folder_path
    
    train_views_folders = []
    
    for i in range(4):
        train_views_folders.append(os.path.join(folder_path, f"train_view{i}"))
        
    images_all = []
    for folder in train_views_folders:
        image_file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_file_paths = sorted(image_file_paths)
        images = []
        for path in image_file_paths:
            print("path is: ", path)
            img = Image.open(path)
            images.append(img)
        images_all.append(images)
    
    
    # image_file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # image_file_paths = sorted(image_file_paths)

    # images = []
    # for path in image_file_paths:
    #     print("path is: ", path)
    #     img = Image.open(path)
    #     images.append(img)
    
    total_time = 2
    start_time = start_time_train
    
    print("start_time_train is: ", start_time_train)

    for i in range(total_time):
        fid = i + start_time
        for idx, pose in enumerate(w2cs):
            print("idx is: ", idx)
            train_cam_infos.append(
                CameraInfo4D(
                    uid=i,
                    R=np.transpose(pose[:3, :3]),
                    T=pose[:3, 3],
                    # FovY=np.deg2rad(fov),
                    # FovX=np.deg2rad(fov),
                    FovY=fov,
                    FovX=fov,
                    image=images_all[idx][fid],
                    image_path=None,
                    image_name=idx,
                    width=reso,
                    height=reso,
                    fid=fid,
                    flow=None,
                )
            )
            
    # load test cameras
    fid = 0
    example_img = Image.open("/mnt/nas_9/group/leiyuanhang/code2025/windforce/WD-Objects/example/example.png")
        
    for idx, pose in enumerate(w2cs_test):
        test_cam_infos.append(
            CameraInfo4D(
                uid=idx,
                R=np.transpose(pose[:3, :3]),
                T=pose[:3, 3],
                # FovY=np.deg2rad(fov),
                # FovX=np.deg2rad(fov),
                FovY=fov,
                FovX=fov,
                image=example_img,
                image_path=None,
                image_name=idx,
                width=reso,
                height=reso,
                fid=fid,
                flow=None,
            )
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)
   
    num_pts = 50_000
    xyz = np.random.randn(num_pts, 3) * radius / 16
    xyz = np.random.randn(num_pts, 3) * radius / 16
    shs = np.ones((num_pts, 3)) * 0.2
   
    ply_path = "./tmp/points3d.ply"
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )
    
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    
    # pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path="./tmp/points3d.ply",
    )

    return scene_info

def constructForceSceneFlow(
    num_frames=21,
    radius=2.0,
    elevation=10.0,
    fov=33.8,
    reso=800,
    images=[],
    masks=[],
    num_pts=100_000,
    train=True,
):
    # test phygaussian ficus data: in phygs radius is 4.11
    radius = 4.11
    # radius = 1.0
    elevation = 8.96
    reso = 800
    focal = 1111.1110311937682
    fov = focal2fov(focal, reso)
    num_frames = 1 # train on only one ref video
    

    poses = get_uniform_poses(num_frames, radius, elevation)
    w2cs = np.linalg.inv(poses)
    train_cam_infos = []

    print("radius: ", radius)
    print("elevation: ", elevation)
    print("fov: ", fov)
    print("num_frames: ", num_frames)
    print("poses.shape: ", poses.shape)
    print("reso: ", reso)
    
    def get_mp4_files(folder_path):
        mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        mp4_files.sort()
        mp4_paths = [os.path.join(folder_path, f) for f in mp4_files]
        return mp4_paths

    folder_path = '/mnt/nas_9/group/leiyuanhang/code/gic/output_simgt/pampass_flower_gt_fps24_force10/render' 
    flow_folder_path = '/mnt/nas_9/group/leiyuanhang/code/gic/output_simgt/pampass_flower_gt_fps24_force10/flow_gt'
    
    mp4_file_paths = get_mp4_files(folder_path)

    for path in mp4_file_paths:
        print(path)
    
    # total_time = len(mp4_file_paths)
    # reference first 6 frames (t0 is static frame)
    total_time = 6
    
    images = []
    frames = read_video(mp4_file_paths[0])
    for frame in frames:
        images.append(Image.fromarray(frame))
        
    # Load flow files
    flow_files = sorted([os.path.join(flow_folder_path, f) for f in os.listdir(flow_folder_path) if f.endswith('.npy')])
    if len(flow_files) < total_time - 1:
        raise ValueError("Not enough flow files for the specified number of frames.")

    flows = []
    for flow_file in flow_files:
        flow = np.load(flow_file)
        flows.append(flow)

    for i in range(total_time):
        fid = i
        flow = None
        if i < total_time - 1:
            flow = flows[i]
    
        for idx, pose in enumerate(w2cs):
            train_cam_infos.append(
                CameraInfo4D(
                    uid=idx,
                    R=np.transpose(pose[:3, :3]),
                    T=pose[:3, 3],
                    # FovY=np.deg2rad(fov),
                    # FovX=np.deg2rad(fov),
                    FovY=fov,
                    FovX=fov,
                    image=images[i],
                    image_path=None,
                    image_name=idx,
                    width=reso,
                    height=reso,
                    fid=fid,
                    flow=flow,
                )
            )

    nerf_normalization = getNerfppNorm(train_cam_infos)
   
    # num_pts = 100_000
    # radius = radius
    num_pts = 50_000
    xyz = np.random.randn(num_pts, 3) * radius / 16
    xyz = np.random.randn(num_pts, 3) * radius / 16
    shs = np.ones((num_pts, 3)) * 0.2
   
    ply_path = "./tmp/points3d.ply"
    pcd = BasicPointCloud(
        points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
    )
    
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    
    # pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=train_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path="./tmp/points3d.ply",
    )

    return scene_info



class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], pcd=None, load_fix_pcd=False, cam_info=None, render_view=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
       
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.object_path, n_views=args.n_views, random_init=args.random_init, train_split=args.train_split)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            print("args.static_cam is: ", args.static_cam)
            if args.static_cam:
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            else:
                scene_info = sceneLoadTypeCallbacks["Blender_Force"](args.source_path, args.white_background, args.eval, start_time_train=args.start_time, train_folder_path=args.train_folder_path, render_view=render_view) 
            # scene_info = sceneLoadTypeCallbacks["Blender_Force"](args.source_path, args.white_background, args.eval, start_time_train=args.start_time, train_folder_path=args.train_folder_path)
            # scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            # print("Use synthetic data")
            # print("args.mv_num: ", args.mv_num)
            # print("args.radius_set: ", args.radius_set)
            # print("args.elevation_set: ", args.elevation_set)
            # print("args.start_time: ", args.start_time)
            # print("args.train_folder_path: ", args.train_folder_path)
            # print("args.adjust_cam : ", args.adjust_cam)
            # print("args.static_cam: ", args.static_cam)
            
            if args.static_cam:
                scene_info = constructPose3D(radius=args.radius_set, elevation=args.elevation_set, adjust_cam=args.adjust_cam)
            else:
                scene_info = constructForceScene(radius=args.radius_set, elevation=args.elevation_set, start_time_train=args.start_time, train_folder_path=args.train_folder_path, adjust_cam=args.adjust_cam, render_view=render_view) 
            # scene_info = constructForceScene(radius=args.radius_set, elevation=args.elevation_set, start_time_train=args.start_time, train_folder_path=args.train_folder_path, adjust_cam=args.adjust_cam) 
            
            # assert False, "Could not recognize scene type!"
        
        # test loader
        # scene_info = constructVideoNVSInfo4D()
        # scene_info = constructForceScene()
        # scene_info = constructForceSceneFlow()
        
        resolution_scales=[1.0]
        for resolution_scale in resolution_scales:
            # print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            # print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
        
        
       

    def save(self, iteration, fix_pcd=False):
        name = "point_cloud/iteration_{}".format(iteration) if not fix_pcd else "point_cloud_fix_pcd/iteration_{}".format(iteration)
        point_cloud_path = os.path.join(self.model_path, name)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def clipTrainCamerasbyframes(self, f):
        new_cameras = {}
        for scale, cam_list in self.train_cameras.items():
            cam_frames = len(torch.unique(torch.stack([view.fid for view in cam_list])))
            if f < cam_frames:
                sorted_times, _ = torch.sort(torch.unique(torch.stack([view.fid for view in cam_list])))
                max_t = sorted_times[:f][-1]
                new_cameras[scale] = [v for v in cam_list if v.fid <= max_t] #<=
            else:
                new_cameras[scale] = cam_list
        self.train_cameras = new_cameras

    def clipTestCamerasbyframes(self, f):
        new_cameras = {}
        for scale, cam_list in self.test_cameras.items():
            cam_frames = len(torch.unique(torch.stack([view.fid for view in cam_list])))
            if f < cam_frames:
                sorted_times, _ = torch.sort(torch.unique(torch.stack([view.fid for view in cam_list])))
                max_t = sorted_times[:f][-1]
                new_cameras[scale] = [v for v in cam_list if v.fid <= max_t] #<=
            else:
                new_cameras[scale] = cam_list
        self.test_cameras = new_cameras

    
    def overwrite_alphas(self, pipeline, dataset: ModelParams, deform: DeformModel):
        from gaussian_renderer import render
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        xyz_canonical = self.gaussians.get_xyz.detach()
        def overwrite(cam_dict):
            for scale, cam_list in cam_dict.items():
                for view in cam_list:
                    fid = view.fid
                    time_input = fid.unsqueeze(0).expand(1, -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
                    results = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                    alpha = results["alpha"]
                    view.gt_alpha_mask = alpha.to(view.data_device)
            return copy.deepcopy(cam_dict)
        train_cams = overwrite(self.train_cameras)
        test_cams = overwrite(self.test_cameras)
        cameras_extent = copy.deepcopy(self.cameras_extent)
        return train_cams, test_cams, cameras_extent