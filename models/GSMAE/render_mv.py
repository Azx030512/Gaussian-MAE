from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
import time
import torch
import os
import subprocess
from scene import GaussianModel, Scene
from gaussian_renderer import render
from argparse import ArgumentParser, Namespace
import numpy as np
import trimesh as tm
import torchvision
import json
from tqdm import tqdm
import torch.nn.functional as F




# dataset.model_path
def load_pcd_file(path):
    pcd = tm.load_mesh(path)
    np_pcd = np.array(pcd.vertices)
    vol = torch.from_numpy(np_pcd).to('cuda',dtype=torch.float32).contiguous()
    return vol


def gen_bg(scene, color="white"):
    view = scene.getTrainCameras(scale=1.0)[0]
    w, h = view.image_width, view.image_height
    fn = torch.ones if color=="white" else torch.zeros
    bg = fn((3, h, w), device="cuda")
    return bg

def read_bg(phys_args, scene, dataset):
    use_orign_bg = getattr(phys_args, "origin_bg", False)
    if not use_orign_bg:
        return gen_bg(scene)
    cam_index = phys_args.view_id
    real_bg_path = os.path.join(dataset.source_path, "data", f"r_{cam_index}_-1.png")
    img = torchvision.io.read_image("/mnt/nas_9/group/leiyuanhang/code/PhysGaussian/output/0000.png")
    return img.to('cuda') / 255.0


def render_nvs(dataset: ModelParams, pipeline: PipelineParams, scene):
    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[1.0], load_fix_pcd=True)
    gaussians = scene.gaussians
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    viewpoint_stack = scene.getTrainCameras().copy()
    # viewpoint_cam = viewpoint_stack.pop(0)

    with torch.no_grad():
        for i in range(len(viewpoint_stack)):
            print("render cam i: ", i)
            viewpoint_cam = viewpoint_stack[i]
            d_xyz = torch.zeros_like(gaussians.get_xyz)
            results = render(viewpoint_cam, gaussians, pipeline, background, d_xyz, 0.0, 0.0, False)
            rendering = results["render"]
            
            if not os.path.exists(os.path.join(dataset.model_path, 'render_mv')):
                os.makedirs(os.path.join(dataset.model_path, 'render_mv'))
            
            torchvision.utils.save_image(rendering, os.path.join(dataset.model_path, 'render_mv', 'cam_{0:02d}'.format(i) + ".png"))


if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Generate new trajectory")
    parser.add_argument('-vid', '--view_id', type=int, default=3)
    parser.add_argument('-knn', '--use_knn', type=bool, default=False)
    parser.add_argument('-cid', '--config_id', type=int, default=0)
    parser.add_argument('--gs_ply', type=str, default="/data/ckpt/aizixiang/Gaussian-MAE/experiments/pretrain_enc_full_group_xyz_1k/pretrain/gaussian_mae_enc_full_group_xyz_1k/save_ply/1a640c8dffc5d01b8fd30d65663cfd42_ep_0300_full_rebuild_gaussian.ply")
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    dataset = model.extract(gs_args)
    
    ply_path = gs_args.gs_ply

    print("dataset.sh_degree is: ", dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_path)
    
    scene = Scene(dataset, gaussians)

    render_nvs(dataset, pipeline.extract(gs_args), scene)