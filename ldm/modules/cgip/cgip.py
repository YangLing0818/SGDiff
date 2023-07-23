""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import math
from typing import Optional
from ldm.modules.cgip.tools import create_tensor_by_assign_samples_to_img


class CGIPModel(nn.Module):
    def __init__(self,
                 num_objs: int,
                 num_preds: int,
                 width: int,
                 layers: int,
                 embed_dim: int,
                 ckpt_path: str,
                 ignore_keys: list = [],
                 max_sample_per_img: int = 15
                 ):
        super().__init__()

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.


        self.num_objs = num_objs
        self.num_preds = num_preds
        self.max_relationships_per_image = max_sample_per_img
        self.obj_embeddings = nn.Embedding(num_objs + 1, embed_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embed_dim)

        self.graph_conv = GraphTripleConv(embed_dim, output_dim=embed_dim, hidden_dim=width, pooling='avg', mlp_normalization='none')
        self.graph_net = GraphTripleConvNet(embed_dim, num_layers=layers, hidden_dim=width, pooling='avg', mlp_normalization='none')
        self.graph_projection = nn.Linear(embed_dim * 2, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")


    def encode_graph_local_global(self, graph):

        image, objs, boxes, triples, obj_to_img, triples_to_img = graph

        batch_size, _, H, W = image.shape

        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.graph_conv, nn.Linear):
            obj_vecs = self.graph_conv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.graph_conv(obj_vecs, pred_vecs, edges)
        if self.graph_net is not None:
            obj_vecs, pred_vecs = self.graph_net(obj_vecs, pred_vecs, edges)

        obj_fea = self.pool_samples(obj_vecs, obj_to_img)
        pred_fea = self.pool_samples(pred_vecs, triples_to_img)
        graph_global_fea = self.graph_projection(torch.cat([obj_fea, pred_fea], dim=1))

        s_obj_vec, o_obj_vec = obj_vecs[s], obj_vecs[o]
        triple_vec = torch.cat([s_obj_vec, pred_vecs, o_obj_vec], dim=1)
        graph_local_fea = create_tensor_by_assign_samples_to_img(samples=triple_vec, sample_to_img=triples_to_img,
                                                                 max_sample_per_img=self.max_relationships_per_image,
                                                                 batch_size=batch_size)

        return graph_local_fea, graph_global_fea

    def forward(self, graph):
        graph_local_fea, graph_global_fea = self.encode_graph_local_global(graph)

        return graph_local_fea, graph_global_fea

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


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)

class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, output_dim=None, hidden_dim=512,
                 pooling='avg', mlp_normalization='none'):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)

    def forward(self, obj_vecs, pred_vecs, edges):
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]

        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs

    def init_parameters(self):
        self.net1.apply(_init_weights)
        self.net2.apply(_init_weights)

class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg', mlp_normalization='none'):
        super().__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,'pooling': pooling,
            'mlp_normalization': mlp_normalization,
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs

    def init_parameters(self):
        for gc in self.gconvs:
            gc.apply(_init_weights)

def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x