import json
import logging
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sgCLIP.contrastive_losses import ClipLoss
from sgCLIP.generative_loss import ReconstractMaskedImageFromSceneGraphLoss
from training.distributed import is_master
from training.precision import get_autocast
from utils import generate_box_mask
from global_var import *


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()
    clip_loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size)

    mim_loss = ReconstractMaskedImageFromSceneGraphLoss(
        triple_dim=1536,
        # image_dim=3072,
        image_dim=768,
        num_img_patches=50,
        num_triple=15,
        sg_only=False
    )
    mim_loss = mim_loss.to(device)

    num_batches_per_epoch = len(dataloader)
    sample_digits = math.ceil(math.log(num_batches_per_epoch + 1, 10))

    total_loss_m = AverageMeter()
    c_loss_m = AverageMeter()
    g_loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, objects, boxes, triples, obj_to_img, triple_to_img = [x.to(device=device, non_blocking=True) for x in batch]
        graphs = [objects, boxes, triples, obj_to_img, triple_to_img]

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            local_gt_image_feature, local_graph_features, norm_global_gt_image_features, norm_global_graph_features, logit_scale = \
                model(images, graphs)

            batch_size, _, H, W = images.shape
            box_mask_for_img = generate_box_mask(boxes_gt=boxes, obj_to_img=obj_to_img, H=H, W=W, threshold=0.2)
            masked_images = images * box_mask_for_img.to(device)
            with torch.no_grad():
                local_masked_image_feature, _, norm_global_masked_image_features, _, _ = model(masked_images.detach(), graphs)

            c_loss = clip_loss(norm_global_gt_image_features, norm_global_graph_features, logit_scale)
            g_loss = mim_loss(local_graph_features, local_masked_image_feature, local_gt_image_feature)
            total_loss = c_loss + g_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()

            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            total_loss_m.update(total_loss.item(), batch_size)
            c_loss_m.update(total_loss.item(), batch_size)
            g_loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Total Loss: {total_loss_m.val:#.5g} ({total_loss_m.avg:#.4g}) "
                f"Contras Loss: {c_loss_m.val:#.5g} ({c_loss_m.avg:#.4g}) "
                f"Gen Loss: {g_loss_m.val:#.5g} ({g_loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size * args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            log_data = {
                "total loss": total_loss_m.val,
                "contras loss": c_loss_m.val,
                "gen loss": g_loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            batch_time_m.reset()
            data_time_m.reset()


def validate_one_epoch(model, dataloader, epoch, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.eval()

    total_acc_g2i = 0.
    total_acc_i2g = 0.
    batch_count = 0
    batch_size = 0
    for i, batch in enumerate(dataloader):
        batch_count += 1

        images, objects, boxes, triples, obj_to_img, triple_to_img = [x.to(device=device, non_blocking=True) for x in
                                                                      batch]
        graphs = [objects, boxes, triples, obj_to_img, triple_to_img]

        with autocast():
            local_gt_image_feature, local_graph_features, norm_global_gt_image_features, norm_global_graph_features, logit_scale = \
                model(images, graphs)

        batch_size = images.shape[0]
        assert batch_size > 1

        acc_g2i, acc_i2g = validate_acc(norm_global_gt_image_features, norm_global_graph_features, device)
        total_acc_g2i += acc_g2i.item()
        total_acc_i2g += acc_i2g.item()

    avg_acc_g2i = total_acc_g2i / batch_count
    avg_acc_i2g = total_acc_i2g / batch_count

    logging.info(
        f"Validate Epoch: {epoch}  "
        f"Validate Batch Size: {batch_size}"
        f"Average accuracy of g2i: {avg_acc_g2i:.3f} @ {batch_size}"
        f"Average accuracy of i2g: {avg_acc_i2g:.3f} @ {batch_size}"
    )

    log_data = {
        "avg_acc_g2i": avg_acc_g2i,
        "avg_acc_i2g": avg_acc_i2g,
    }
    for name, val in log_data.items():
        name = "validate/" + name
        if tb_writer is not None:
            tb_writer.add_scalar(name, val, epoch)

    print(
        "@batch_size %d \n average accuracy of graph-to-image is %f \n average accuracy of image-to-graph is %f \n" % (
        batch_size, avg_acc_g2i, avg_acc_i2g))


def validate_acc(img_emb, graph_emb, device):
    img_emb = img_emb.detach().to(device)
    graph_emb = graph_emb.detach().to(device)
    with torch.no_grad():
        B, D = img_emb.shape
        sim = torch.matmul(img_emb, graph_emb.T)

        pred_graph_to_img = sim - torch.max(sim, dim=0, keepdim=True).values.expand(B, B)
        pred_img_to_graph = sim - torch.max(sim, dim=1, keepdim=True).values.expand(B, B)

        pred_graph_to_img = (pred_graph_to_img >= 0).int().to(device)
        pred_img_to_graph = (pred_img_to_graph >= 0).int().to(device)
        gt_label_mask = mask_correlated_samples(B).to(device)

        pred_graph_to_img = pred_graph_to_img * gt_label_mask
        pred_img_to_graph = pred_img_to_graph * gt_label_mask

        correct_pred_graph_to_img = torch.sum(pred_graph_to_img).to(device)
        correct_pred_img_to_graph = torch.sum(pred_img_to_graph).to(device)

        acc_graph_to_img = correct_pred_graph_to_img * 1.0 / B
        acc_img_to_graph = correct_pred_img_to_graph * 1.0 / B

        return acc_graph_to_img, acc_img_to_graph


def mask_correlated_samples(batch_size):
    N = batch_size
    mask = torch.zeros((N, N), dtype=torch.int)
    mask = mask.fill_diagonal_(1)
    return mask


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


if __name__ == '__main__':
    img_emb = torch.randn([64, 128])
    graph_emb = torch.randn([64, 128])
    img_emb = F.normalize(img_emb, dim=-1)
    graph_emb = F.normalize(graph_emb, dim=-1)
    acc_g2i, acc_i2g = validate_acc(img_emb, graph_emb)
    print(acc_g2i, acc_i2g)