import torch
import torch.nn as nn
import os
from tools import builder
from utils import misc, dist_utils
import time
import wandb
from utils.logger import print_log, get_logger
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from datasets import data_transforms
from utils.gaussian import write_gaussian_feature_to_ply, unnormalize_gaussians
from tqdm import tqdm
import torchvision
from argparse import ArgumentParser
from gaussians.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussians import GaussianModel, Scene, render

train_transforms = data_transforms.PointcloudScaleAndTranslate()


class Acc_Metric:
    def __init__(self, acc=0.0):
        if type(acc).__name__ == "dict":
            self.acc = acc["acc"]
        else:
            self.acc = acc
    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False
    def state_dict(self):
        _dict = dict()
        _dict["acc"] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1.0 / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader) = (builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get("extra_train")else (None, None))
    # build model
    # pass the norm attribute to the model
    config.model.norm_attribute = config.dataset.train.others.norm_attribute
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        print("Using GPU of Rank", args.local_rank)
        base_model = base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.0)
    metrics = Acc_Metric(0.0)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger, strict_load=True)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger, strict_load=True)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    from utils.misc import summary_parameters
    summary_parameters(base_model, logger=logger)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(["Loss"])
        num_iter = 0
        base_model.train()  # set model to training mode
        base_model.zero_grad()
        n_batches = len(train_dataloader)
        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data, scale_c, scale_m) in enumerate(tqdm(train_dataloader,smoothing=0.9)):
        
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train._base_.NAME
            points = data.cuda()

            if config.npoints_fps:
                # using fps gs to select subset of points
                points = misc.fps_gs(points, npoints, attribute=config.model.group_attribute)
            else:
                # using random sampling
                random_idx = np.random.choice(points.size(1), npoints, False)
                points = points[:, random_idx, :].contiguous()

            if epoch != config.max_epoch:
                points = train_transforms.augument(points, attribute=config.model.attribute)
                
            if False: #(epoch%30 == 0 and idx == 0):  # save last epoch ply for visualization
                loss_dict, vis_gaussians, full_rebuild_gaussian, original_gaussians = base_model(points, save=True)
                # save to gaussian ply
                os.makedirs(os.path.join(args.experiment_path, "save_ply"), exist_ok=True)
                original_gaussians, vis_gaussians, full_rebuild_gaussian = unnormalize_gaussians(original_gaussians,vis_gaussians,full_rebuild_gaussian,scale_c,scale_m,config,)
                for i in range(vis_gaussians.shape[0]):  # save whole batch
                    vis_gaussians_ply_path = os.path.join(args.experiment_path,"save_ply",f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_vis_gaussians.ply",)
                    full_rebuild_gaussian_ply_path = os.path.join(args.experiment_path,"save_ply",f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_full_rebuild_gaussian.ply",)
                    original_gaussians_ply_path = os.path.join(args.experiment_path,"save_ply",f"{model_ids[i]}_original_gaussians.ply",)
                    write_gaussian_feature_to_ply(vis_gaussians[i], vis_gaussians_ply_path)
                    write_gaussian_feature_to_ply(full_rebuild_gaussian[i], full_rebuild_gaussian_ply_path)
                    write_gaussian_feature_to_ply(original_gaussians[i], original_gaussians_ply_path)
                    if getattr(config.model, "appearence_loss", False):
                        parser = ArgumentParser(description="Generate new trajectory")
                        model = ModelParams(parser)#, sentinel=True)
                        pipeline = PipelineParams(parser)
                        op = OptimizationParams(parser)
                        gs_args, phys_args = get_combined_args(parser)
                        dataset = model.extract(gs_args)
                        bg_color = [1, 1, 1]
                        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                        vis = GaussianModel(3)
                        vis.load_ply(vis_gaussians_ply_path)
                        full = GaussianModel(3)
                        full.load_ply(full_rebuild_gaussian_ply_path)
                        original = GaussianModel(3)
                        original.load_ply(original_gaussians_ply_path)
                        scene = Scene(dataset, vis)
                        viewpoint_stack = scene.getTrainCameras().copy()
                        d_xyz = torch.zeros([3], device='cuda')

                        viewpoint_cam = viewpoint_stack[0]
                        vis_results = render(viewpoint_cam, vis, pipeline, background, d_xyz, 0.0, 0.0, False)
                        vis_renderings = vis_results["render"].detach().cpu()
                        full_results = render(viewpoint_cam, full, pipeline, background, d_xyz, 0.0, 0.0, False)
                        full_renderings = full_results["render"].detach().cpu()
                        original_results = render(viewpoint_cam, original, pipeline, background, d_xyz, 0.0, 0.0, False)
                        original_renderings = original_results["render"].detach().cpu()

                        torchvision.utils.save_image(vis_renderings, os.path.join(args.experiment_path,"save_ply",f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_vis.png"))
                        torchvision.utils.save_image(full_renderings, os.path.join(args.experiment_path,"save_ply",f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_full_rebuild.png"))
                        torchvision.utils.save_image(original_renderings, os.path.join(args.experiment_path,"save_ply",f"{model_ids[i]}_original.png"))

            else:
                loss_dict = base_model(points)

            # aggregate all loss
            loss = sum([loss_dict[key] for key in loss_dict.keys()])
            
            loss.backward()
            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.detach().item()])
            else:
                losses.update([loss.detach().mean().item()])
                # all loss_dict change to item, follow the pointmae
                loss_dict = {key: loss_dict[key].detach().mean().item() for key in loss_dict.keys()}

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar("Loss/Batch/Loss", loss.detach().item(), n_itr)
                # use loss dict to add scaler
                for key in loss_dict.keys():
                    train_writer.add_scalar(f"Loss/Batch/{key}", loss_dict[key], n_itr)
                train_writer.add_scalar("Loss/Batch/LR", optimizer.param_groups[0]["lr"], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar("Loss/Epoch/Loss_1", losses.avg(0), epoch)
        print_log(
            "[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f"
            % (
                epoch,
                epoch_end_time - epoch_start_time,
                ["%.4f" % l for l in losses.avg()],
                optimizer.param_groups[0]["lr"],
            ),
            logger=logger,
        )
        builder.save_checkpoint(
            base_model,
            optimizer,
            epoch,
            metrics,
            best_metrics,
            "ckpt-last",
            args,
            logger=logger,
        )
        if epoch == 250 or epoch == 275 or epoch == 300:
            builder.save_checkpoint(
                base_model,
                optimizer,
                epoch,
                metrics,
                best_metrics,
                f"ckpt-epoch-{epoch:03d}",
                args,
                logger=logger,
            )

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(
    base_model,
    extra_train_dataloader,
    test_dataloader,
    epoch,
    val_writer,
    args,
    config,
    logger=None,
):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.N_POINTS
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(extra_train_dataloader,smoothing=0.9)):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader,smoothing=0.9)):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(
            train_features.data.cpu().numpy(),
            train_label.data.cpu().numpy(),
            test_features.data.cpu().numpy(),
            test_label.data.cpu().numpy(),
        )

        print_log("[Validation] EPOCH: %d  acc = %.4f" % (epoch, svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar("Metric/ACC", svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass
