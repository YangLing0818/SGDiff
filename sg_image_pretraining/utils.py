from itertools import repeat
import collections.abc
import torchvision.transforms as T
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from global_var import *

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
def generate_box_mask(boxes_gt, obj_to_img, H, W=None, threshold=0.2):
    if W is None:
        W = H
    bbox_mask = boxes_to_mask(boxes_gt, obj_to_img, H, W, threshold)
    return bbox_mask

class HookTool:
    def __init__(self):
        self.extract_fea_in = None
        self.extract_fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.extract_fea_in = fea_in
        self.extract_fea = fea_out


def get_linear_feas_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks

def boxes_to_mask(boxes, obj_to_img, H, W=None, threshold=0.2):
    O = obj_to_img.size()[0]
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)
    x_rand = torch.rand([O, 1])
    mask_indicator = (x_rand > threshold).float()
    mask_in = mask_indicator.view(O, 1, 1, 1).expand(O, 1, 8, 8)
    mask_in = mask_in.to(device)
    sampled = F.grid_sample(mask_in, grid)
    sampled = (sampled > 0).float().to(device)

    out = assign_mask_to_img(sampled, obj_to_img)
    out = out.to(device)
    mask = 1.0 - out

    return mask.to(device)


def assign_mask_to_img(samples, obj_to_img):
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, samples)

    return out

def _boxes_to_grid(boxes, H, W):
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)

    grid = grid.mul(2).sub(1)

    return grid

def create_tensor_by_assign_samples_to_img(samples, sample_to_img, max_sample_per_img, batch_size):
    dtype, device = samples.dtype, samples.device
    N = batch_size
    D = samples.shape[1]
    assert (sample_to_img.max() + 1) == N

    samples_per_img = []
    for i in range(N):
        s_idxs = (sample_to_img == i).nonzero().view(-1)
        sub_sample = samples[s_idxs]
        len_cur = sub_sample.shape[0]
        if len_cur > max_sample_per_img:
            sub_sample = sub_sample[:max_sample_per_img, :]
        if len_cur < max_sample_per_img:
            zero_vector = torch.zeros([1, D]).to(device)
            padding_vectors = torch.cat([copy.deepcopy(zero_vector) for _ in range(max_sample_per_img - len_cur)], dim=0) # [res, D]
            sub_sample = torch.cat([sub_sample, padding_vectors], dim=0)
        sub_sample = sub_sample.unsqueeze(0)
        samples_per_img.append(sub_sample)
    samples_per_img = torch.cat(samples_per_img, dim=0).to(device)

    return samples_per_img

def idx_to_one_hot(idx, num_classes):
    result = F.one_hot(idx, num_classes)
    result = result.float().to(device)
    return result


def freeze_batch_norm_2d(module, module_match={}, name=''):
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):

    O, D = vecs.size()
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in, grid)  # (O, D, H, W)

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out


def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum'):
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    sampled = F.grid_sample(img_in, grid)

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)
    return out


def _boxes_to_grid(boxes, H, W):
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)

    grid = grid.mul(2).sub(1)

    return grid


def _pool_samples(samples, obj_to_img, pooling='sum'):
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, samples)

    if pooling == 'avg':
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out