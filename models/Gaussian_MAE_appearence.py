import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from .build import MODELS
from utils.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from utils.logger import print_log
import random, math
from knn_cuda import KNN
from models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    Encoder,
    Group,
    SoftEncoder,
)
from pytorch3d.loss import chamfer_distance
from models.Gaussian_MAE import MaskTransformer
from gaussians import GaussianModel, Scene, render
from typing import List
from lpips import LPIPS
import torch.nn.functional as F
from torch.autograd import Variable

attr_index = {
    "xyz":[0,1,2],
    "o":[3],
    "s":[4, 5, 6],
    "r":[7, 8, 9, 10],
    "sh":[11, 12, 13]
}

from argparse import ArgumentParser
from gaussians.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
parser = ArgumentParser(description="Generate new trajectory")
model = ModelParams(parser)#, sentinel=True)
pipeline = PipelineParams(parser)
op = OptimizationParams(parser)
gs_args, phys_args = get_combined_args(parser)
dataset = model.extract(gs_args)
bg_color = [1, 1, 1]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
scene = Scene(dataset, None)
viewpoint_stack = scene.getTrainCameras().copy()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

loss_fn_vgg = None
def lpips(img1, img2, value_range=(0, 1)):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



# pretrain model
@MODELS.register_module()
class Gaussian_MAE_appearence(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log("[Gaussian_MAE] ", logger="Gaussian_MAE")
        self.config = config
        self.soft_knn = getattr(config, "soft_knn", False)
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config, soft_knn=self.soft_knn)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.knn = KNN(k=config.group_size, transpose_mode=True)
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.attribute = config.attribute
        self.group_attribute = config.group_attribute
        self.norm_attribute = config.norm_attribute

        self.pos_feature_dim = []
        if "xyz" in config.group_attribute:
            self.pos_feature_dim.extend([0, 1, 2])

        if "opacity" in config.group_attribute:
            self.pos_feature_dim.append(3)

        if "scale" in config.group_attribute:
            self.pos_feature_dim.extend([4, 5, 6])

        if "rotation" in config.group_attribute:
            self.pos_feature_dim.extend([7, 8, 9, 10])

        if "sh" in config.group_attribute:
            self.pos_feature_dim.extend([11, 12, 13])

        print("pos embedding size", self.pos_feature_dim)
        print("group_attribute", self.group_attribute)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(
            f"[Gaussian_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Gaussian_MAE",
        )
        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            attribute=config.group_attribute,
            soft_knn=self.soft_knn,
        )

        # prediction head for xyz
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        # predication head for density
        if "opacity" in self.attribute:
            self.opacity_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 1 * self.group_size, 1),
                nn.Sigmoid() if "opacity" not in self.norm_attribute else nn.Tanh(),  # otherwise the opacity range is [-1, 1]
            )

        if "scale" in self.attribute and "rotation" in self.attribute:
            self.scale_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 3 * self.group_size, 1),
                nn.ReLU() if "scale" not in self.norm_attribute else nn.Tanh(),
            )

            self.rotation_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 4 * self.group_size, 1), nn.Tanh()
            )

        if "sh" in self.attribute:
            self.sh_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 3 * self.group_size, 1),
            )

        trunc_normal_(self.mask_token, std=0.02)

        # appearence modules
        self.resolution = 400
        self.appearence_loss = getattr(config, "appearence_loss", False)
        # if self.appearence_loss:
        #     self._init_renderer()
        
    
    def _render_batch(self, reps: List[GaussianModel], extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = None
        for i, representation in enumerate(reps):
            render_pack = self.renderer.render(representation, extrinsics[i], intrinsics[i])
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
            for k, v in render_pack.items():
                ret[k].append(v)
            ret['bg_color'].append(self.renderer.bg_color)
        for k, v in ret.items():
            ret[k] = torch.stack(v, dim=0) 
        return ret

    def forward(self, pts, vis=False, save=False, model_ids=None, **kwargs):
        # we do color change here in cuda and batch
        opacity_index = [3]
        scale_index = [4, 5, 6]
        rotation_index = [7, 8, 9, 10]
        sh_index = [11, 12, 13]

        neighborhood, center = self.group_divider(pts)

        center_pos = center[..., self.pos_feature_dim]
        x_vis, mask = self.MAE_encoder(neighborhood, center_pos)
        B, _, C = x_vis.shape  # B VIS C
        feature_dim = neighborhood.shape[-1]

        pos_emd_vis = self.decoder_pos_embed(center_pos[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center_pos[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        # retrieve the recosntruction target: use nearest in the potential neighbors
        if self.soft_knn:
            neighborhood = neighborhood[:, :, : self.group_size, :]

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3) # B M 1024
        gt_points = neighborhood[..., :3][mask].reshape(B * M, -1, 3)
        loss_dict = {}
        loss1 = chamfer_distance(rebuild_points, gt_points, norm=2)[0]
        loss_dict["cd"] = loss1

        if "opacity" in self.attribute:
            rebuild_density = self.opacity_head(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 1) # B M 1024
            gt_density = neighborhood[..., opacity_index][mask].reshape(B * M, -1, 1)
            # L1 loss for density
            loss2 = torch.nn.functional.l1_loss(rebuild_density, gt_density)
            loss_dict["density"] = loss2

        if "scale" in self.attribute and "rotation" in self.attribute:
            rebuild_scale = self.scale_head(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
            
            gt_scale = neighborhood[..., scale_index][mask].reshape(B * M, -1, 3)

            rebuild_rotation = self.rotation_head(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 4)
            # normalize rotation
            rebuild_rotation[..., 0] = 1 - rebuild_rotation[..., 0]
            rebuild_rotation = rebuild_rotation / (torch.norm(rebuild_rotation, p=2, dim=-1, keepdim=True) + 1e-9)
            gt_rotation = neighborhood[..., rotation_index][mask].reshape(B * M, -1, 4)

            loss_scale = torch.nn.functional.l1_loss(rebuild_scale, gt_scale)  # * 0.01
            loss_rotation = torch.nn.functional.l1_loss(rebuild_rotation, gt_rotation)  # * 0.01 # try L1 first
            loss_dict["scale"] = loss_scale  # * 0.01
            loss_dict["rotation"] = loss_rotation  # * 0.01

        if "sh" in self.attribute:
            # print("x_rec", x_rec.shape) # ([128, 38, 384]) #  token M
            rebuild_sh = self.sh_head(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            gt_sh = neighborhood[..., sh_index][mask].reshape(B * M, -1, 3)

            loss3 = torch.nn.functional.l1_loss(rebuild_sh, gt_sh)  # * 0.01
            loss_dict["sh"] = loss3

        if self.appearence_loss:
            rebuild_gaussians = [rebuild_points]
            if "opacity" in self.attribute:
                rebuild_gaussians.append(rebuild_density)
            else:
                rebuild_opacity = neighborhood[..., opacity_index][mask].reshape(B * M, -1, 1)
                rebuild_gaussians.append(rebuild_opacity)
            if "scale" in self.attribute and "rotation" in self.attribute:
                rebuild_gaussians.append(rebuild_scale)
                rebuild_gaussians.append(rebuild_rotation)
            else:
                rebuild_scale = neighborhood[..., scale_index][mask].reshape(B * M, -1, 3)
                rebuild_gaussians.append(rebuild_scale)
                rebuild_rotation = neighborhood[..., rotation_index][mask].reshape(B * M, -1, 4)
                rebuild_gaussians.append(rebuild_rotation)
            if "sh" in self.attribute:
                rebuild_gaussians.append(rebuild_sh)
            
            rebuild_gaussians = torch.cat(rebuild_gaussians, dim=-1)
            vis_gaussians = neighborhood[~mask].reshape(B * (self.num_group - M), -1, feature_dim)[..., :14]
            vis_gaussians[..., :3] = vis_gaussians[..., :3] + center_pos[..., :3][~mask].unsqueeze(1)  # xyz position back to world
            rebuild_gaussians[..., :3] = rebuild_gaussians[..., :3] + center_pos[..., :3][mask].unsqueeze(1)
            vis_gaussians = vis_gaussians.reshape(B, -1, vis_gaussians.shape[-1])
            rebuild_gaussians = rebuild_gaussians.reshape(B, -1, rebuild_gaussians.shape[-1])
            full_gaussians = torch.cat([rebuild_gaussians, vis_gaussians], dim=1)
            original_gaussians = pts.clone().detach()
            rebuild_reps = self.to_representation(full_gaussians)
            original_reps = self.to_representation(original_gaussians)

            d_xyz = torch.zeros([3], device='cuda')
            rebuild_renderings=[]
            original_renderings=[]
            for i in range(len(rebuild_reps)):
                viewpoint_cam = random.choice(viewpoint_stack)
                rebuild_results = render(viewpoint_cam, rebuild_reps[i], pipeline, background, d_xyz, 0.0, 0.0, False)
                rebuild_renderings.append(rebuild_results["render"][None,...])
                with torch.no_grad():
                    original_results = render(viewpoint_cam, original_reps[i], pipeline, background, d_xyz, 0.0, 0.0, False)
                    original_renderings.append(original_results["render"][None,...])
            rebuild_renderings=torch.concat(rebuild_renderings, dim=0)
            original_renderings=torch.concat(original_renderings, dim=0)

            loss4 = l1_loss(rebuild_renderings, original_renderings)
            loss_dict["appearence"] = loss4 * 3
            # t=original_renderings.detach().clone().cpu()
            # import os
            # import torchvision
            # for i in range(t.shape[0]):
            #     torchvision.utils.save_image(t[i], os.path.join('mae-reconstruct-render', 'gt_{0:02d}'.format(i) + ".png"))
            # t=rebuild_renderings.detach().clone().cpu()
            # for i in range(t.shape[0]):
            #     torchvision.utils.save_image(t[i], os.path.join('mae-reconstruct-render', 'rebuild_{0:02d}'.format(i) + ".png"))
            
        if save:
            # debug we choose first in batch
            rebuild_gaussians = [rebuild_points]
            if "opacity" in self.attribute:
                rebuild_gaussians.append(rebuild_density)
            else:
                rebuild_opacity = neighborhood[..., opacity_index][mask].reshape(B * M, -1, 1)
                rebuild_gaussians.append(rebuild_opacity)

            if "scale" in self.attribute and "rotation" in self.attribute:
                rebuild_gaussians.append(rebuild_scale)
                rebuild_gaussians.append(rebuild_rotation)
            else:
                rebuild_scale = neighborhood[..., scale_index][mask].reshape(B * M, -1, 3)
                rebuild_gaussians.append(rebuild_scale)
                rebuild_rotation = neighborhood[..., rotation_index][mask].reshape(B * M, -1, 4)
                rebuild_gaussians.append(rebuild_rotation)

            if "sh" in self.attribute:
                rebuild_gaussians.append(rebuild_sh)

            # get back gaussian feature
            rebuild_gaussians = torch.cat(rebuild_gaussians, dim=-1)
            # print("neighborhood", neighborhood.shape)
            vis_gaussians = neighborhood[~mask].reshape(B * (self.num_group - M), -1, feature_dim)[..., :14]
            vis_gaussians[..., :3] = vis_gaussians[..., :3] + center_pos[..., :3][~mask].unsqueeze(1)  # xyz position back to world
            rebuild_gaussians[..., :3] = rebuild_gaussians[..., :3] + center_pos[..., :3][mask].unsqueeze(1)
            vis_gaussians = vis_gaussians.reshape(B, -1, vis_gaussians.shape[-1])
            rebuild_gaussians = rebuild_gaussians.reshape(B, -1, rebuild_gaussians.shape[-1])
            full_gaussians = torch.cat([rebuild_gaussians, vis_gaussians], dim=1)
            original_gaussians = pts.clone().detach()

            return loss_dict, vis_gaussians, full_gaussians, original_gaussians
        else:
            return loss_dict
    
    def to_representation(self, x: torch.Tensor) -> List[GaussianModel]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        reps = []
        for i in range(x.shape[0]):
            representation = GaussianModel(sh_degree=0,)
            representation.from_xyz(x[i,:,attr_index['xyz']])
            representation.from_opacity(x[i,:,attr_index['o']])
            representation.from_scaling(x[i,:,attr_index['s']])
            representation.from_rotation(x[i,:,attr_index['r']])
            representation.from_features_dc(x[i,:,attr_index['sh']])
            # representation.from_features_rest(torch.zeros([x.shape[1],3*((representation.max_sh_degree+1)**2-1)], device='cuda'))
            reps.append(representation)
        return reps


# finetune model
@MODELS.register_module()
class GaussianTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.soft_knn = getattr(config, "soft_knn", False)

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.attribute = config.attribute

        self.group_attribute = config.group_attribute
        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            attribute=config.group_attribute,
            soft_knn=self.soft_knn,
        )

        self.encoder = (
            SoftEncoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
            if self.soft_knn
            else Encoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_feature_dim = []
        if "xyz" in config.group_attribute:
            self.pos_feature_dim.extend([0, 1, 2])

        if "opacity" in config.group_attribute:
            self.pos_feature_dim.append(3)

        if "scale" in config.group_attribute:
            self.pos_feature_dim.extend([4, 5, 6])

        if "rotation" in self.group_attribute:
            self.pos_feature_dim.extend([7, 8, 9, 10])

        if "sh" in config.group_attribute:
            self.pos_feature_dim.extend([11, 12, 13])

        print("self.pos_feature_dim ", self.pos_feature_dim)
        print("config.group_attribute", config.group_attribute)

        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.type = config.type

        if self.type == "linear":
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, self.cls_dim)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim),
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)  # KNN
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        center_pos = center[..., self.pos_feature_dim]
        pos = self.pos_embed(center_pos)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
