from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn
import torch.nn.functional as F
import copy


class STUNetTrainer_small(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module: 
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes=label_manager.num_segmentation_heads
        kernel_sizes = [[3,3,3]] * 6
        strides=configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])
        return STUNet(num_input_channels, num_classes, depth=[1]*6, dims= [16 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision)


       
