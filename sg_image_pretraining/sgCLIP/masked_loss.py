from global_var import *
import torch
import torch.nn as nn
import torch.distributed.nn
from sgCLIP.module import GraphTripleConv, GraphTripleConvNet, Attention, Transformer
from utils import boxes_to_mask
has_distributed = True

class MaskedSceneGraphLoss(nn.Module):
    def __init__(self, triple_dim, max_relationships_per_image=15, threshold=0.3):
        super().__init__()
        self.transformer = Transformer(width=triple_dim, layers=2, heads=4)
        self.triple_mlp = nn.Sequential(nn.Linear(triple_dim, triple_dim))
        self.criterion = nn.MSELoss(reduction='mean')

        self.threshold = threshold

    def forward(self, triple_per_img):
        triple_per_img = self.triple_mlp(triple_per_img)
        rec_loss = self.calculate_reconstruction_loss(triple_per_img)
        return rec_loss

    def calculate_reconstruction_loss(self, triple_fea):
        batch_size = triple_fea.shape[0]
        len_triple_fea = triple_fea.shape[1]

        triple_mask = (torch.rand([batch_size, len_triple_fea, 1]) > self.threshold).float().detach().to(device)    # [0 for mask]
        masked_triple_fea = triple_mask * triple_fea

        rec_fea = self.reconstruct_missing_sg(masked_triple_fea)
        gt_fea = triple_fea.detach()

        valid_mask = (torch.mean(triple_fea, dim=2, keepdim=True)).float()
        loss_mask = (1 - triple_mask) * valid_mask.detach()
        loss_mask.detach()

        rec_loss = self.criterion(rec_fea * loss_mask, gt_fea * loss_mask)
        return rec_loss

    def reconstruct_missing_sg(self, triple_fea):

        triple_fea = triple_fea.permute(1, 0, 2).contiguous()
        rec_fea = self.transformer(triple_fea)
        rec_fea = rec_fea.permute(1, 0, 2).contiguous()
        return rec_fea

class Img2MaskedSceneGraphLoss(nn.Module):
    def __init__(self, triple_dim, image_dim, max_relationships_per_image=30, image_size=32, threshold=0.3):
        super().__init__()
        self.transformer = Transformer(width=image_dim, layers=2, heads=4)
        self.triple_mlp = nn.Sequential(nn.Linear(triple_dim, image_dim))
        self.criterion = nn.MSELoss(reduction='mean')

        self.threshold = threshold

        self.image_size = image_size
        self.image_dim = image_dim

    def forward(self, triple_per_img, image_feature):
        image_feature = image_feature.detach()
        triple_per_img = self.triple_mlp(triple_per_img)
        rec_loss = self.calculate_reconstruction_loss(triple_per_img, image_feature)
        return rec_loss

    def calculate_reconstruction_loss(self, triple_fea, img_fea):
        batch_size = triple_fea.shape[0]
        img_fea = img_fea.permute(0,2,3,1).contiguous()
        img_fea = img_fea.view(batch_size, self.image_size * self.image_size, self.image_dim)
        len_img_fea = img_fea.shape[1]
        assert len_img_fea == self.image_size * self.image_size
        len_triple_fea = triple_fea.shape[1]

        triple_mask = (torch.rand([batch_size, len_triple_fea, 1]) > self.threshold).float().to(device)
        masked_triple_fea = triple_mask * triple_fea

        rec_fea = self.reconstruct_missing_sg(masked_triple_fea, img_fea)
        gt_fea = torch.cat([triple_fea, img_fea], dim=1)

        valid_mask = torch.ones([batch_size, len_triple_fea + len_img_fea, 1]) * 1.0
        valid_mask = valid_mask.to(device)
        valid_mask[:, :len_triple_fea, :] = (torch.mean(triple_fea.detach(), dim=2, keepdim=True)).float()
        loss_mask = torch.ones([batch_size, len_triple_fea + len_img_fea, 1]) * 1.0
        loss_mask = loss_mask.to(device)
        loss_mask[:, :len_triple_fea, :] = triple_mask
        loss_mask = (1 - loss_mask) * valid_mask
        loss_mask.detach()

        rec_loss = self.criterion(rec_fea * loss_mask, gt_fea * loss_mask)
        return rec_loss

    def reconstruct_missing_sg(self, triple_fea, img_fea):

        input_fea = torch.cat([triple_fea, img_fea], dim=1)  # [B, N+HW, C]
        input_fea = input_fea.permute(1, 0, 2).contiguous()
        rec_fea = self.transformer(input_fea)
        rec_fea = rec_fea.permute(1, 0, 2).contiguous()
        return rec_fea

class SceneGraph2MakedImgLoss(nn.Module):
    def __init__(self, triple_dim, image_dim, sg_only, max_relationships_per_image=30, image_size=32, threshold=0.3):
        super().__init__()

        self.transformer = Transformer(width=image_dim, layers=2, heads=4)
        self.triple_mlp = nn.Sequential(nn.Linear(triple_dim, image_dim))
        self.criterion = nn.MSELoss(reduction='mean')

        self.threshold = threshold

        self.image_size = image_size
        self.image_dim = image_dim

        if sg_only:
            self.register_buffer('attn_mask', self.build_attention_mask(tri_length=max_relationships_per_image, img_length=image_size * image_size), persistent=False)
        else:
            self.attn_mask = None

    def forward(self, triple_per_img, image_feature, gt_boxes, obj_to_img):
        triple_per_img = self.triple_mlp(triple_per_img)
        rec_loss = self.calculate_reconstruction_loss(triple_per_img, image_feature, gt_boxes, obj_to_img)
        return rec_loss

    def build_box_mask(self, boxes_gt, obj_to_img, H, W=None, threshold=0.2):
        bbox_mask = boxes_to_mask(boxes_gt, obj_to_img, H, W, threshold)
        return bbox_mask

    def build_attention_mask(self, tri_length, img_length):
        total_length = tri_length + img_length
        mask = torch.empty(total_length, total_length)
        mask.fill_(1)
        mask[tri_length:, tri_length:].fill_(float("-inf"))
        return mask

    def calculate_reconstruction_loss(self, triple_fea, img_fea, boxes_gt, obj_to_img):
        batch_size = triple_fea.shape[0]

        img_fea = img_fea.permute(0, 2, 3, 1).contiguous()
        img_fea = img_fea.view(batch_size, self.image_size * self.image_size, self.image_dim)

        len_img_fea = img_fea.shape[1]
        assert len_img_fea == self.image_size * self.image_size
        len_triple_fea = triple_fea.shape[1]

        image_mask = self.build_box_mask(boxes_gt, obj_to_img, H=self.image_size, W=self.image_size, threshold=self.threshold)
        image_mask = image_mask.view(batch_size, -1, 1)
        masked_img_fea = image_mask * img_fea

        rec_fea = self.reconstruct_missing_patches(triple_fea, masked_img_fea)
        gt_fea = torch.cat([triple_fea, img_fea], dim=1)

        loss_mask = torch.ones([batch_size, len_triple_fea + len_img_fea, 1]) * 1.0
        loss_mask = loss_mask.to(device)
        loss_mask[:, len_triple_fea:, :] = image_mask
        loss_mask = 1 - loss_mask
        loss_mask.detach()

        rec_loss = self.criterion(rec_fea * loss_mask, gt_fea * loss_mask)
        return rec_loss

    def reconstruct_missing_patches(self, triple_fea, img_fea):
        input_fea = torch.cat([triple_fea, img_fea], dim=1) # [B, N+HW, C]
        input_fea = input_fea.permute(1, 0, 2).contiguous()
        rec_fea = self.transformer(input_fea, attn_mask=self.attn_mask)
        rec_fea = rec_fea.permute(1, 0, 2).contiguous()
        return rec_fea