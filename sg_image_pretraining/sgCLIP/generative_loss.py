import torch
import torch.nn as nn
import torch.distributed.nn
from sgCLIP.mm_transformer_module import BasicTransformerBlock
from utils import boxes_to_mask
has_distributed = True

class ReconstractMaskedImageFromSceneGraphLoss(nn.Module):
    def __init__(self, triple_dim, image_dim, num_img_patches=50, num_triple=15, sg_only=False):
        super().__init__()

        self.image_dim = image_dim

        if sg_only:
            self.register_buffer('attn_mask', self.build_attention_mask(tri_length=num_triple, img_length=num_img_patches), persistent=False)
        else:
            self.attn_mask = None

        self.transformer = BasicTransformerBlock(dim=image_dim, n_heads=8, d_head=64, dropout=0., context_dim=triple_dim)

        self.criterion = nn.MSELoss()

    def forward(self, local_graph_fea, local_masked_image_fea, local_gt_image_fea):
        local_masked_image_fea = local_masked_image_fea.permute(1, 0, 2).contiguous()
        local_gt_image_fea = local_gt_image_fea.permute(1, 0, 2).contiguous()

        local_reconstructed_img_fea = self.transformer(local_masked_image_fea, context=local_graph_fea)

        rec_loss = self.criterion(local_reconstructed_img_fea, local_gt_image_fea)
        return rec_loss


class ReconstractMaskedSceneGraphFromImageLoss(nn.Module):
    def __init__(self, triple_dim, image_dim, num_img_patches=50, num_triple=15, sg_only=False):
        super().__init__()

        self.triple_dim = triple_dim

        if sg_only:
            self.register_buffer('attn_mask', self.build_attention_mask(tri_length=num_triple, img_length=num_img_patches), persistent=False)
        else:
            self.attn_mask = None

        self.transformer = BasicTransformerBlock(dim=triple_dim, n_heads=8, d_head=64, dropout=0., context_dim=image_dim)

        self.criterion = nn.MSELoss()

    def forward(self, local_graph_fea, local_masked_graph_fea, local_gt_image_fea):
        local_gt_image_fea = local_gt_image_fea.permute(1, 0, 2).contiguous()

        local_reconstructed_graph_fea = self.transformer(local_masked_graph_fea, context=local_gt_image_fea)

        rec_loss = self.criterion(local_reconstructed_graph_fea, local_graph_fea)
        return rec_loss