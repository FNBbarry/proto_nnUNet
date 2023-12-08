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
from nnunetv2.training.nnUNetTrainer.models.modules.mask2former.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

from nnunetv2.training.nnUNetTrainer.models.modules.detectron2.utils.memory import retry_if_cuda_oom

from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

click_methods = {
    'random': get_next_click3D_torch_2,
}

class ProtoSamMask2FormerNet(nn.Module):
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
        super(ProtoSamMask2FormerNet, self).__init__()
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

        in_channels = 1920
        n_quires = 100
        # self.cls_head = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
        #     nn.Dropout2d(0.10)
        # )
        
        self.cls_head_3d = nn.Sequential(
            nn.Conv3d(in_channels//16, self.num_classes, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(self.num_classes, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout3d(0.10)
        )

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, n_quires*10),
                                       requires_grad=True)

        # self.proj_head = ProjectionHead(in_channels, in_channels)
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels//2, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//4, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//8, 256, 1),
            nn.ReLU(inplace=True))

        self.mask_features = nn.Conv3d(
            in_channels//16,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.proj_head_3d = ProjectionHead_3D_upscale(in_channels, in_channels // 16)
        
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels = 256,
            mask_classification = True,
            num_classes = self.num_classes,
            hidden_dim = 384,
            num_queries = 100,
            nheads = 8,
            dim_feedforward = 2048,
            dec_layers = 9,
            pre_norm = False,
            mask_dim = 256,
            enforce_input_project = False
        )
        
        self.feat_norm = nn.LayerNorm(n_quires*10)
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

    def semantic_inference(self, mask_cls, mask_pred):
        # mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_cls = F.softmax(mask_cls, dim=-1)
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qdhw->cdhw", mask_cls, mask_pred)
        return semseg

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False):
        """
        ProtoSeg模型的前向传播
        """
        # x = self.backbone(x_)

        x = self.backbone.image_encoder(x_)
    
        _, _, d, h, w = x_.size()
        feats = torch.concat(x,dim=1)
        # feats = x
        
        multi_scale_features = []
        c_tmp = feats
        for i in range(len(self.proj_head_3d.proj)):
            c_tmp = self.proj_head_3d.proj[i](c_tmp)
            if i ==7:
                mask_features = self.mask_features(c_tmp)
            if i in [1,3,5]:
                c_proj_tmp = self.proj[i-1](c_tmp)
                multi_scale_features.append(self.proj[i](c_proj_tmp))
            if i == len(self.proj_head_3d.proj)-1:
                c = c_tmp
        
        # _c = self.cls_head_3d(c)

        outputs = self.transformer_decoder(multi_scale_features[::-1],mask_features,mask=None)
        
        # validation or inference process
        if not self.training:
            mask_cls_results = outputs["pred_logits"][-1]
            mask_pred_results = outputs["pred_masks"]

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                # semantic segmentation inference
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                processed_results.append(r.unsqueeze(0))
            preds_list = torch.concat(processed_results,dim=0)
            return preds_list
        
        # training process
        _c = torch.concat(outputs['pred_logits'],dim=1)
            
        _c = rearrange(_c, 'b c d -> (b d) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
        
        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        
        out_seg = rearrange(out_seg, "(b d) k -> b d k", b=feats.shape[0], d=self.num_classes)

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg_labels = []
            for i in range(feats.shape[0]):
                all_labels = torch.arange(self.num_classes,dtype=torch.float)
                for tmp_i in range(self.num_classes):
                    if tmp_i not in torch.unique(gt_semantic_seg[i,:,:,:,:]):
                        all_labels[tmp_i] = 0
                gt_seg_labels.append(all_labels.unsqueeze(0))
            gt_seg = torch.concat(gt_seg_labels,dim=0)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg.to(out_seg.device).squeeze(0).view(-1), masks)
            return {'seg': outputs, 'logits': contrast_logits, 'target': contrast_target}