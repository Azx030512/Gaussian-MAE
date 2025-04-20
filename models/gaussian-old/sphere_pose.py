import open3d as o3d
import numpy as np
import cv2
import json
import os
from collections import defaultdict
import json
from PIL import Image
from tqdm import tqdm

def get_pointcloud(rgb, depth, K, C2W, normal=None):
    mesh_grids = np.meshgrid(np.arange(rgb.shape[1], dtype=np.float32),  # 宽
                             np.arange(rgb.shape[0], dtype=np.float32),  # 高
                             indexing="xy")
    i_coords = mesh_grids[0]
    j_coords = mesh_grids[1]

    norm_cam_coor = np.stack([(i_coords - K[0][2]) / K[0][0],
                              (j_coords - K[1][2]) / K[1][1],
                              np.ones_like(i_coords)], -1)
    cam_coor = norm_cam_coor * depth[..., None]
    world_coor = cam_coor @ C2W[:, :3].T + C2W[:, 3:].T
    mask = depth > 0  # & depth < 10
    world_coor = world_coor[mask].reshape(-1, 3)
    rgb = rgb[mask].reshape(-1, 3)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(world_coor)
    pts.colors = o3d.utility.Vector3dVector(rgb)
    if normal is not None:
        pts.normals = o3d.utility.Vector3dVector(normal[mask].reshape(-1, 3))

    return pts

def count_cam_idx(cam_dict):
    indexs = []
    for cam in cam_dict:
        index = cam_dict[cam]["cam_idx"]
        if index not in indexs:
            indexs.append(index)
    return indexs

def get_camera_frustum(img_size, K, C2W, frustum_length, color, scale_pose_ratio=1.0):
    # pose_scale用于放大位姿
    # [w,h]  [4,4]  [3,4]
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)  # 光心到图像左右两边的角度
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)  # 光心到图像上下两边的角度
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))  # 归一化平面
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))
    # 不就是W,H的一半

    pose = np.eye(4)
    pose[:3] = C2W
    C2W = pose

    # 调整尺度
    '''
    half_w *= 5e-3
    half_h *= 5e-3
    frustum_length *= 5e-3
    '''

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               [-half_w, -half_h, frustum_length],  # 左上角，注意x朝右，y朝上
                               [half_w, -half_h, frustum_length],  # 右上角
                               [half_w, half_h, frustum_length],  # bottom-right image corner
                               [-half_w, half_h, frustum_length],
                               # 坐标轴
                               [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]  # x轴,y轴,z轴
                               ])  # bottom-left image corner
    frustum_points *= scale_pose_ratio

    frustum_lines = np.array([[0, i] for i in range(1, 5)] +
                             [[i, (i + 1)] for i in range(1, 4)] +
                             [[4, 1], [0, 5], [0, 6], [0, 7]])
    # 平铺
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    frustum_colors[-1] = np.array([0, 0, 1])  # Z 蓝色
    frustum_colors[-2] = np.array([0, 1, 0])  # y 绿色
    frustum_colors[-3] = np.array([1, 0, 0])  # x 红色

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    frustum_points = np.dot(
        np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))),  # 齐次坐标
        C2W.T)  # 8，4
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 8, 3))  # 5 vertices per frustum 总共有5个点
    merged_lines = np.zeros((N * 11, 2))  # 8 lines per frustum # 总共有8条线
    merged_colors = np.zeros((N * 11, 3))  # each line gets a color # 每条线一个颜色

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 8:(i + 1) * 8, :] = frustum_points
        merged_lines[i * 11:(i + 1) * 11, :] = frustum_lines + i * 8
        merged_colors[i * 11:(i + 1) * 11, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 8, 3))  # 5 vertices per frustum 总共有5个点
    merged_lines = np.zeros((N * 11, 2))  # 8 lines per frustum # 总共有8条线
    merged_colors = np.zeros((N * 11, 3))  # each line gets a color # 每条线一个颜色

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 8:(i + 1) * 8, :] = frustum_points
        merged_lines[i * 11:(i + 1) * 11, :] = frustum_lines + i * 8
        merged_colors[i * 11:(i + 1) * 11, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def visual_cameras(cam_dicts, data_type="colmap", scale_pose_ratio=1.0, ply=None, bbox=None, mesh=None,
                   visual_depth=False, others=[]):
    # 设置起点在路径中心
    if visual_depth:
        pointcloud = None
    things_to_draw = [ply] + others if ply is not None else others

    if bbox is not None: things_to_draw.append(bbox)
    if mesh is not None: things_to_draw.append(mesh)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))
    # things_to_draw.append(sphere)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
    # things_to_draw.append(coord_frame)

    frustums = []

    cam_idxs = count_cam_idx(cam_dicts)
    colors = {}  # 每个index显示一种颜色
    '''
    color_begin = [1, 0, 0]
    color_end = [0, 1, 1]
    color_delta = (np.array(color_end) - np.array(color_begin)) / len(cam_idxs)
    for i,cam in enumerate(cam_idxs):
        colors[cam] = list(
            np.array(color_begin)+i*color_delta
        )
    '''
    for i, cam in enumerate(cam_idxs):
        colors[cam] = list(
            np.random.random(3)
        )

    for cam_dict in tqdm(cam_dicts):
        cam_info = cam_dicts[cam_dict]
        cam_show_idxs = cam_idxs
        if cam_info["cam_idx"] in cam_show_idxs:  # 筛选index进行展示
            index = cam_info["cam_idx"]

            intrinsics = cam_info["intrinsics"]
            if type(intrinsics) == np.ndarray:
                if intrinsics.shape[0] == 3:
                    K = np.eye(4)
                    K[:3, :3] = intrinsics
                elif intrinsics.shape[0] == 4:
                    K = intrinsics
            elif type(intrinsics) == list:
                if len(intrinsics) == 3:
                    K = np.array([
                        [cam_info["intrinsics"][2], 0, cam_info["intrinsics"][1] / 2, 0],  # 宽
                        [0, cam_info["intrinsics"][2], cam_info["intrinsics"][0] / 2, 0],  # 高
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                else:
                    K = np.array([
                        [cam_info["intrinsics"][0], 0, cam_info["intrinsics"][2], 0],  # 宽
                        [0, cam_info["intrinsics"][1], cam_info["intrinsics"][3], 0],  # 高
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])

            # K = cam_info["intrinsics"]
            C2W = cam_info["c2w"]

            # NeRF坐标系和open3D不一样，需要转换
            if data_type == "nerf":
                C2W[:, 1:3] *= -1

            # C2W[:, 1:3] *= -1
            if "rgb" in cam_info:
                img_size = [cam_info["rgb"].shape[1], cam_info["rgb"].shape[0]]  # 宽、高
            elif type(intrinsics) == list:
                img_size = [cam_info["intrinsics"][1], cam_info["intrinsics"][0]]
            else:
                img_size = [int(cam_info["intrinsics"][0, 2] * 2), int(cam_info["intrinsics"][1, 2] * 2)]
            frustum_length = K[0, 0] / img_size[1]

            color = colors[index] if "color" not in cam_info else cam_info["color"]
            frustums.append(
                get_camera_frustum(img_size, K, C2W, frustum_length, color, scale_pose_ratio=scale_pose_ratio))

            if visual_depth and cam_info["depth"] is not None:
                pts = get_pointcloud(cam_info["rgb"] / 255.0, cam_info["depth"], K, C2W,
                                     normal=None)
                # pts = pts.voxel_down_sample(0.01)
                if pointcloud is None:
                    pointcloud = pts
                else:
                    pointcloud += pts 

                things_to_draw.append(pts)

    cameras = frustums2lineset(frustums)
    # o3d.visualization.draw_geometries([cameras])
    things_to_draw.append(cameras)

    o3d.visualization.draw_geometries(things_to_draw)

    if visual_depth:
        return pointcloud
    return None

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
                    (origin_trans @ spheric_pose(th, ph, radius * (level + 1)))[:3]]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


if __name__ == "__main__":
    # TODO: 1. 增加region的可视化
    ply = o3d.io.read_point_cloud(r"C:\Users\Ai\Desktop\gaussian\1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply")
    mesh = None

    bbox = None

    scale_pose_ratio = 0.1  # 可视化缩小pose

    ######## 求取位姿
    cams_pose = create_spheric_poses(1.0, [0, 0, 0])

    intrinsic = np.eye(4)
    intrinsic[0, 0] = 600
    intrinsic[1, 1] = 600
    intrinsic[0, 2] = 400
    intrinsic[1, 2] = 400

    vis_meta = defaultdict(dict)
    for frame, cams in enumerate(cams_pose):
        c2w = cams

        vis_meta[frame]["c2w"] = c2w[:3]
        vis_meta[frame]["intrinsics"] = intrinsic
        vis_meta[frame]["cam_idx"] = 0

    pointcloud = visual_cameras(vis_meta, data_type="colmap", scale_pose_ratio=scale_pose_ratio, ply=ply, bbox=bbox,
                                mesh=mesh,
                                visual_depth=False)
    if pointcloud is not None:
        o3d.io.write_point_cloud("tmp.ply", pointcloud)
