""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sgCLIP.module import ModifiedResNet, QuickGELU, VisualTransformer, GraphTripleConv, GraphTripleConvNet, GraphAggregationNetwork, Attention
from training.configs import CLIPVisionCfg, CLIPGraphCfg
from utils import create_tensor_by_assign_samples_to_img, get_linear_feas_by_hook
import clip
from global_var import *

class sgCLIP(nn.Module):
    def __init__(self,
                 graph_vocab: dict,
                 graph_cfg: CLIPGraphCfg,
                 embed_dim: int,
                 max_sample_per_img: int=15,
                 ):
        super().__init__()
        if isinstance(graph_cfg, dict):
            graph_cfg = CLIPGraphCfg(**graph_cfg)

        self.clip_model, preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval().requires_grad_(False).to(device)

        num_objs = len(graph_vocab['object_idx_to_name'])
        num_preds = len(graph_vocab['pred_idx_to_name'])
        self.num_objs = num_objs
        self.num_preds = num_preds
        self.max_sample_per_img = max_sample_per_img
        self.obj_embeddings = nn.Embedding(num_objs + 1, embed_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embed_dim)

        self.graph_conv = GraphTripleConv(embed_dim, output_dim=embed_dim, hidden_dim=graph_cfg.width, pooling='avg', mlp_normalization='none')
        self.graph_net = GraphTripleConvNet(embed_dim, num_layers=graph_cfg.layers, hidden_dim=graph_cfg.width, pooling='avg', mlp_normalization='none')
        self.graph_projection = nn.Linear(embed_dim * 2, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def initialize_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.graph_conv, 'init_parameters'):
            self.graph_conv.init_parameters()
        if hasattr(self.graph_net, 'init_parameters'):
            self.graph_net.init_parameters()
        if hasattr(self.graph_projection, 'init_parameters'):
            self.graph_projection.init_parameters()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=False):
        self.visual.set_grad_checkpointing(enable)

    def encode_image_local_global(self, image):
        with torch.no_grad():
            extract_linear_feas = get_linear_feas_by_hook(self.clip_model.visual)
            global_image_fea = self.clip_model.encode_image(image)
            local_image_fea = extract_linear_feas[-1].extract_fea
        return local_image_fea.detach(), global_image_fea.detach()

    def encode_graph_local_global(self, img, graph):
        batch_size, _, H, W = img.shape

        objs, boxes, triples, obj_to_img, triples_to_img = graph
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.graph_conv, nn.Linear):
            obj_vecs = self.graph_conv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.graph_conv(obj_vecs, pred_vecs, edges)
        if self.graph_net is not None:
            obj_vecs, pred_vecs = self.graph_net(obj_vecs, pred_vecs, edges)

        # Global Branch
        obj_fea = self.pool_samples(obj_vecs, obj_to_img)
        pred_fea = self.pool_samples(pred_vecs, triples_to_img)
        graph_global_fea = self.graph_projection(torch.cat([obj_fea, pred_fea], dim=1))

        # Local Branch
        s_obj_vec, o_obj_vec = obj_vecs[s], obj_vecs[o]
        triple_vec = torch.cat([s_obj_vec, pred_vecs, o_obj_vec], dim=1)
        graph_local_fea = create_tensor_by_assign_samples_to_img(samples=triple_vec, sample_to_img=triples_to_img,
                                                                max_sample_per_img=self.max_sample_per_img,
                                                                batch_size=batch_size)

        return graph_local_fea, graph_global_fea

    def forward(self, image, graph):
        local_image_feature, global_image_features = self.encode_image_local_global(image)
        norm_global_image_features = F.normalize(global_image_features, dim=-1)
        local_graph_features, global_graph_features = self.encode_graph_local_global(image, graph)
        norm_global_graph_features = F.normalize(global_graph_features, dim=-1)

        return local_image_feature, local_graph_features, norm_global_image_features, norm_global_graph_features, self.logit_scale.exp()

    def pool_samples(self, samples, obj_to_img, pooling='avg'):
        dtype, device = samples.dtype, samples.device
        O, D = samples.size()

        N = obj_to_img.data.max().item() + 1

        out = torch.zeros(N, D, dtype=dtype, device=device)
        idx = obj_to_img.view(O, 1).expand(O, D)
        out = out.scatter_add(0, idx, samples)

        if pooling == 'avg':
            ones = torch.ones(O, dtype=dtype, device=device)
            obj_counts = torch.zeros(N, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
            obj_counts = obj_counts.clamp(min=1)
            out = out / obj_counts.view(N, 1)
        elif pooling != 'sum':
            raise ValueError('Invalid pooling "%s"' % pooling)

        return out

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def idx_to_one_hot(idx, num_classes):
    result = F.one_hot(idx, num_classes)
    result = result.float()
    return result
