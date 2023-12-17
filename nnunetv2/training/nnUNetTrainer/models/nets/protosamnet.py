import os
import pdb
import torch
import torch.nn as nn
import numpy as np
from torch.cuda import amp
import torch.nn.functional as F
import torch.distributed as dist
from nnunetv2.training.nnUNetTrainer.models.backbones.segment_anything.utils.click_method import get_next_click3D_torch_2

from nnunetv2.training.nnUNetTrainer.models.backbones.backbone_selector import BackboneSelector
from nnunetv2.training.nnUNetTrainer.models.tools.module_helper import ModuleHelper
from nnunetv2.training.nnUNetTrainer.models.modules.projection import ProjectionHead
from nnunetv2.training.nnUNetTrainer.models.modules.hanet_attention import HANet_Conv
from nnunetv2.training.nnUNetTrainer.models.modules.contrast import momentum_update, l2_normalize, ProjectionHead, ProjectionHead_3D,ProjectionHead_3D_upscale
from nnunetv2.training.nnUNetTrainer.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

click_methods = {
    'random': get_next_click3D_torch_2,
}

class ProtoSamNet(nn.Module):
    """
    带有prototype的模型结构
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                configer,
                deep_supervision) -> nn.Module:
        super(ProtoSamNet, self).__init__()
        label_manager = plans_manager.get_label_manager(dataset_json)
        self.configer = configer
        self.gamma = self.configer.get('protoseg', 'gamma')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get('protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get('protoseg', 'pretrain_prototype')
        self.num_classes = label_manager.num_segmentation_heads
        self.configer.add(('data','num_classes'),label_manager.num_segmentation_heads)
        self.configer.add(('data','n_channels'),num_input_channels)
        self.backbone = BackboneSelector(configer).get_backbone(plans_manager = plans_manager, 
                                                                dataset_json = dataset_json, 
                                                                configuration_manager = configuration_manager,
                                                                num_input_channels = num_input_channels, 
                                                                deep_supervision = deep_supervision)

        in_channels = 434
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )
        
        self.cls_head_3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout3d(0.10)
        )

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.proj_head_3d = ProjectionHead_3D(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue
            
            q, indexs = distributed_sinkhorn(init_q)
            
            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim
            # if torch.isinf(f).any() == True:
            #     pdb.set_trace()
            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False):
        """
        ProtoSeg模型的前向传播
        """
        # x = self.backbone(x_)

        x = self.backbone.image_encoder(x_)

        masks, iou_pred, src, upscaled_embedding = self.backbone.mask_decoder(x, # (B, 256, 64, 64)
            image_pe=self.backbone.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            multimask_output=False,
        )
        _, _, d, h, w = masks.size()
        mode_interpolate = "bilinear" if len(x_.size()) == 4 else "trilinear"
        
        feat1 = masks
        feat2 = F.interpolate(x_, size=(d, h, w), mode=mode_interpolate, align_corners=True)
        feat3 = F.interpolate(src, size=(d, h, w), mode=mode_interpolate, align_corners=True)
        feat4 = upscaled_embedding

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)

        if len(x_.size()) == 4 :
            c = self.cls_head(feats)

            c = self.proj_head(c)
            _c = rearrange(c, 'b c h w -> (b h w) c')
        elif len(x_.size()) == 5:
            c = self.cls_head_3d(feats)
            
            c = self.proj_head_3d(c)
            _c = rearrange(c, 'b c d h w -> (b h w d) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
        
        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        if len(x_.size()) == 4 :
            out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=h)
        elif len(x_.size()) == 5:
            out_seg = rearrange(out_seg, "(b h w d) k -> b k d h w", b=feats.shape[0], h=h, d=d)

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)
            # gt_seg = gt_semantic_seg.float().view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            out_seg = F.interpolate(out_seg, size=gt_semantic_seg.shape[-3:], mode='trilinear', align_corners=False)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}
       
        return out_seg