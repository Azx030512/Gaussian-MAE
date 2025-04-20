import os
import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor
from time import time
from omegaconf import OmegaConf

import cv2
import numpy as np
from sklearn.cluster import KMeans
from time import time



from argparse import ArgumentParser, Namespace
import sys
import os
import math
import ipdb
from utils.rigid_body_utils import get_rigid_transform, quaternion_multiply, matrix_to_quaternion


def interpolate_points_w_R(
    query_points, query_rotation, drive_origin_pts, drive_displacement, top_k_index
):
    """
    Args:
        query_points: [n, 3]
        drive_origin_pts: [m, 3]
        drive_displacement: [m, 3]
        top_k_index: [n, top_k] < m

    Or directly call: apply_discrete_offset_filds_with_R(self, origin_points, offsets, topk=6):
        Args:
            origin_points: (N_r, 3)
            offsets: (N_r, 3)
        in rendering
    """

    # [n, topk, 3]
    top_k_disp = drive_displacement[top_k_index]
    source_points = drive_origin_pts[top_k_index]

    R, t = get_rigid_transform(source_points, source_points + top_k_disp)

    avg_offsets = top_k_disp.mean(dim=1)

    ret_points = query_points + avg_offsets

    new_rotation = quaternion_multiply(matrix_to_quaternion(R), query_rotation)

    return ret_points, new_rotation