##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nnunetv2.training.nnUNetTrainer.models.backbones.resnet.resnet_backbone import ResNetBackbone
from nnunetv2.training.nnUNetTrainer.models.backbones.hrnet.hrnet_backbone import HRNetBackbone
from nnunetv2.training.nnUNetTrainer.models.backbones.pvt.pvt_backbone import PVTBackbone
from nnunetv2.training.nnUNetTrainer.models.backbones.pvt.pcpvt_backbone import PCPVTBackbone
from nnunetv2.training.nnUNetTrainer.models.backbones.pvt.svt_backbone import SVTBackbone
from nnunetv2.training.nnUNetTrainer.models.backbones.stunet.stunet import STUNet
from nnunetv2.training.nnUNetTrainer.models.backbones.mobilenet.mobilenet_v1 import MobileNetV1Backbone
from nnunetv2.training.nnUNetTrainer.models.backbones.mobilenet.mobilenet_v2 import MobileNetV2Backbone
from nnunetv2.training.nnUNetTrainer.models.backbones.mobilenet.mobilenet_v3 import MobileNetV3Backbone
from nnunetv2.training.nnUNetTrainer.models.backbones.unet.unet_model import UNet
from nnunetv2.training.nnUNetTrainer.models.backbones.segment_anything.build_sam3D import sam_model_registry3D
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get('network', 'backbone')

        model = None
        if ('resnet' in backbone or 'resnext' in backbone or 'resnest' in backbone) and 'senet' not in backbone:
            model = ResNetBackbone(self.configer)(**params)

        elif 'hrne' in backbone:
            model = HRNetBackbone(self.configer)(**params)

        elif 'pcpvt' in backbone:
            model = PCPVTBackbone(self.configer)(**params)

        elif 'pvt' in backbone:
            model = PVTBackbone(self.configer)(**params)

        elif 'svt' in backbone:
            model = SVTBackbone(self.configer)(**params)

        elif 'mobilenet_v1' in backbone:
            model = MobileNetV1Backbone(self.configer)(**params)
        elif 'mobilenet_v2' in backbone:
            model = MobileNetV2Backbone(self.configer)(**params)
        elif 'mobilenet_v3' in backbone:
            model = MobileNetV3Backbone(self.configer)(**params)

        elif 'unet' in backbone:
            if backbone == 'nnunet':
                model = get_network_from_plans(params['plans_manager'], 
                                       params['dataset_json'], 
                                       params['configuration_manager'],
                                       params['num_input_channels'], 
                                       params['deep_supervision'])
            elif 'stunet' in backbone:
                kernel_sizes = [[3,3,3]] * 6
                strides=params['configuration_manager'].pool_op_kernel_sizes[1:]
                if len(strides)>5:
                    strides = strides[:5]
                while len(strides)<5:
                    strides.append([1,1,1])
                    
                if 'small' in backbone:
                    dim_multiply = 16
                elif 'base' in backbone:
                    dim_multiply = 32
                elif 'large' in backbone:
                    dim_multiply = 64
                elif 'huge' in backbone:
                    dim_multiply = 96
                
                model = STUNet(params['num_input_channels'], 
                            self.configer.get('data','num_classes'), 
                            depth=[1]*6, 
                            dims= [dim_multiply * x for x in [1, 2, 4, 8, 16, 16]],
                            pool_op_kernel_sizes=strides, 
                            conv_kernel_sizes=kernel_sizes, 
                            enable_deep_supervision=params['deep_supervision'])
            else:
                model = UNet(n_channels=self.configer.get('data','n_channels'), n_classes=self.configer.get('data','num_classes'))
        
        elif 'sam' in backbone:
            model = sam_model_registry3D['vit_b_ori']()

        else:
            print('Backbone {} is invalid.'.format(backbone))
            exit(1)

        return model
