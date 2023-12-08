from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.loss_helper import FSAuxRMILoss, FSCELoss
from nnunetv2.training.nnUNetTrainer.models.modules.mask2former.modeling.criterion import SetCriterion
from nnunetv2.training.nnUNetTrainer.models.modules.mask2former.modeling.matcher import HungarianMatcher

class PPC(nn.Module, ABC):
    def __init__(self, configer):
        super(PPC, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self, configer):
        super(PPD, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeCELoss(nn.Module, ABC):
    """
    ProtoSeg对应的Loss
    """
    def __init__(self, configer=None,**kwargs):
        super(PixelPrototypeCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        print('ignore_index: {}'.format(ignore_index))

        self.loss_ppc_weight = self.configer.get('protoseg', 'loss_ppc_weight')
        self.loss_ppd_weight = self.configer.get('protoseg', 'loss_ppd_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        self.seg_criterion_normal = FSCELoss(configer=configer)
        
        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            # self.seg_criterion = FSCELoss(configer=configer)
            # mask2former 专属
            class_weight = 2.0
            mask_weight = 5.0
            dice_weight = 5.0
            train_num_points = 12544
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
            oversample_ratio = 3.0
            importance_sample_ratio = 0.75
            self.seg_criterion = SetCriterion(
                self.configer.get('data','num_classes'),
                matcher=HungarianMatcher(cost_class=class_weight,cost_mask=mask_weight,cost_dice=dice_weight,num_points=train_num_points),
                weight_dict=weight_dict,
                eos_coef=0.1,
                losses=["labels", "masks"],
                num_points=train_num_points,
                oversample_ratio=oversample_ratio,
                importance_sample_ratio=importance_sample_ratio,
                )

        self.ppc_criterion = PPC(configer=configer)
        self.ppd_criterion = PPD(configer=configer)

    def forward(self, preds, target):
        # h, w = target[0].size(3), target[0].size(4)
        # 从这里开始target要变成适用于mask2former代码中关于loss计算的形式

        if isinstance(preds, dict):
            # mask2former 专属
            target = self.prepare_targets(target[0])
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            # loss = self.seg_criterion(seg, target[0].squeeze(1).long())
            # return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd
            # mask2former 专属
            loss = self.seg_criterion(seg, target)
            return loss['loss_mask']+loss['loss_dice'] + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd
            

        # seg = preds
        # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        # loss = self.seg_criterion(preds, target[0].squeeze(1).long())
        # mask2former 专属    
        loss = self.seg_criterion_normal(preds, target[0].squeeze(1).long())
        return loss
    
    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in range(targets.shape[0]):
            gt_classes = torch.unique(targets[targets_per_image])
            d,h,w = targets[targets_per_image].shape[-3:]
            num_labels = len(gt_classes) # 9
            padded_masks = torch.zeros((num_labels, d,h,w))
            for i in range(1, num_labels):  # this loop starts from label 1 to ignore background 0
                padded_masks[i, :, :, :] = targets[targets_per_image] == gt_classes[i]
                
            new_targets.append(
                {
                    "labels": gt_classes.long(),
                    "masks": padded_masks.bool().to(gt_classes.device),
                }
            )
        return new_targets





