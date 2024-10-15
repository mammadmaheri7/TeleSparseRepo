import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import Conv2dCOB, LinearCOB, BatchNorm2dCOB
from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.pooling import AdaptiveAvgPool2dCOB,AvgPool2dCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.layers.merge import Add


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out
        out = self.relu2(out)

        return out


NUM_CLASSES = 100
class ResNetCifar(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet20_cifar100(pretrained=False, **kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(weight_dict['resnet20_cifar'])["model_state_dict"])
    return model

class Resnet20Cifar100CobFlat(nn.Module):
    def __init__(self, num_classes=100):
        super(Resnet20Cifar100CobFlat, self).__init__()

        # Initial conv layer
        self.conv1 = Conv2dCOB(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2dCOB(16)
        self.relu1 = ReLUCOB(inplace=False)

        # Layer 1 Block 1
        self.layer1_0_conv1 = Conv2dCOB(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_0_bn1 = BatchNorm2dCOB(16)
        self.relu1_1 = ReLUCOB(inplace=False)
        self.layer1_0_conv2 = Conv2dCOB(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_0_bn2 = BatchNorm2dCOB(16)
        self.relu1_1_second = ReLUCOB(inplace=False)
        self.layer1_0_add = Add()

        # Layer 1 Block 2
        self.layer1_1_conv1 = Conv2dCOB(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_1_bn1 = BatchNorm2dCOB(16)
        self.relu1_2 = ReLUCOB(inplace=False)
        self.layer1_1_conv2 = Conv2dCOB(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_1_bn2 = BatchNorm2dCOB(16)
        self.relu1_2_second = ReLUCOB(inplace=False)
        self.layer1_1_add = Add()

        # Layer 1 Block 3
        self.layer1_2_conv1 = Conv2dCOB(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_2_bn1 = BatchNorm2dCOB(16)
        self.relu1_3 = ReLUCOB(inplace=False)
        self.layer1_2_conv2 = Conv2dCOB(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_2_bn2 = BatchNorm2dCOB(16)
        self.relu1_3_second = ReLUCOB(inplace=False)
        self.layer1_2_add = Add()

        # Layer 2 Block 1 (with downsampling)
        self.layer2_0_conv1 = Conv2dCOB(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_0_bn1 = BatchNorm2dCOB(32)
        self.relu2_1 = ReLUCOB(inplace=False)
        self.layer2_0_conv2 = Conv2dCOB(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_0_bn2 = BatchNorm2dCOB(32)
        self.relu2_1_second = ReLUCOB(inplace=False)
        self.layer2_0_downsample_conv = Conv2dCOB(16, 32, kernel_size=1, stride=2, bias=False)
        self.layer2_0_downsample_bn = BatchNorm2dCOB(32)
        self.layer2_0_add = Add()

        # Layer 2 Block 2
        self.layer2_1_conv1 = Conv2dCOB(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_1_bn1 = BatchNorm2dCOB(32)
        self.relu2_2 = ReLUCOB(inplace=False)
        self.layer2_1_conv2 = Conv2dCOB(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_1_bn2 = BatchNorm2dCOB(32)
        self.relu2_2_second = ReLUCOB(inplace=False)
        self.layer2_1_add = Add()

        # Layer 2 Block 3
        self.layer2_2_conv1 = Conv2dCOB(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_2_bn1 = BatchNorm2dCOB(32)
        self.relu2_3 = ReLUCOB(inplace=False)
        self.layer2_2_conv2 = Conv2dCOB(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_2_bn2 = BatchNorm2dCOB(32)
        self.relu2_3_second = ReLUCOB(inplace=False)
        self.layer2_2_add = Add()

        # Layer 3 Block 1 (with downsampling)
        self.layer3_0_conv1 = Conv2dCOB(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_0_bn1 = BatchNorm2dCOB(64)
        self.relu3_1 = ReLUCOB(inplace=False)
        self.layer3_0_conv2 = Conv2dCOB(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_0_bn2 = BatchNorm2dCOB(64)
        self.layer3_0_downsample_conv = Conv2dCOB(32, 64, kernel_size=1, stride=2, bias=False)
        self.layer3_0_downsample_bn = BatchNorm2dCOB(64)
        self.relu3_1_second = ReLUCOB(inplace=False)
        self.layer3_0_add = Add()

        # Layer 3 Block 2
        self.layer3_1_conv1 = Conv2dCOB(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_1_bn1 = BatchNorm2dCOB(64)
        self.relu3_2 = ReLUCOB(inplace=False)
        self.layer3_1_conv2 = Conv2dCOB(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_1_bn2 = BatchNorm2dCOB(64)
        self.relu3_2_second = ReLUCOB(inplace=False)
        self.layer3_1_add = Add()

        # Layer 3 Block 3
        self.layer3_2_conv1 = Conv2dCOB(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_2_bn1 = BatchNorm2dCOB(64)
        self.relu3_3 = ReLUCOB(inplace=False)
        self.layer3_2_conv2 = Conv2dCOB(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_2_bn2 = BatchNorm2dCOB(64)
        self.relu3_3_second = ReLUCOB(inplace=False)
        self.layer3_2_add = Add()

        # Pooling and Fully Connected layers
        self.avgpool = AvgPool2dCOB(kernel_size=8, stride=1)
        self.flatten = FlattenCOB()
        self.fc = LinearCOB(64, num_classes)

    def forward(self, x):
        # Initial layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Layer 1 Block 1
        residual = x
        x = self.layer1_0_conv1(x)
        x = self.layer1_0_bn1(x)
        x = self.relu1_1(x)
        x = self.layer1_0_conv2(x)
        x = self.layer1_0_bn2(x)
        x = self.layer1_0_add(residual, x)
        x = self.relu1_1_second(x)

        # Layer 1 Block 2
        residual = x
        x = self.layer1_1_conv1(x)
        x = self.layer1_1_bn1(x)
        x = self.relu1_2(x)
        x = self.layer1_1_conv2(x)
        x = self.layer1_1_bn2(x)
        x = self.layer1_1_add(residual, x)
        x = self.relu1_2_second(x)

        # Layer 1 Block 3
        residual = x
        x = self.layer1_2_conv1(x)
        x = self.layer1_2_bn1(x)
        x = self.relu1_3(x)
        x = self.layer1_2_conv2(x)
        x = self.layer1_2_bn2(x)
        x = self.layer1_2_add(residual, x)
        x = self.relu1_3_second(x)

        # Layer 2 Block 1 (with downsampling)
        residual = self.layer2_0_downsample_conv(x)
        residual = self.layer2_0_downsample_bn(residual)
        x = self.layer2_0_conv1(x)
        x = self.layer2_0_bn1(x)
        x = self.relu2_1(x)
        x = self.layer2_0_conv2(x)
        x = self.layer2_0_bn2(x)
        x = self.layer2_0_add(residual, x)
        x = self.relu2_1_second(x)

        # Layer 2 Block 2
        residual = x
        x = self.layer2_1_conv1(x)
        x = self.layer2_1_bn1(x)
        x = self.relu2_2(x)
        x = self.layer2_1_conv2(x)
        x = self.layer2_1_bn2(x)
        x = self.layer2_1_add(residual, x)
        x = self.relu2_2_second(x)

        # Layer 2 Block 3
        residual = x
        x = self.layer2_2_conv1(x)
        x = self.layer2_2_bn1(x)
        x = self.relu2_3(x)
        x = self.layer2_2_conv2(x)
        x = self.layer2_2_bn2(x)
        x = self.layer2_2_add(residual, x)
        x = self.relu2_3_second(x)

        # Layer 3 Block 1 (with downsampling)
        residual = self.layer3_0_downsample_conv(x)
        residual = self.layer3_0_downsample_bn(residual)
        x = self.layer3_0_conv1(x)
        x = self.layer3_0_bn1(x)
        x = self.relu3_1(x)
        x = self.layer3_0_conv2(x)
        x = self.layer3_0_bn2(x)
        x = self.layer3_0_add(residual, x)
        x = self.relu3_1_second(x)

        # Layer 3 Block 2
        residual = x
        x = self.layer3_1_conv1(x)
        x = self.layer3_1_bn1(x)
        x = self.relu3_2(x)
        x = self.layer3_1_conv2(x)
        x = self.layer3_1_bn2(x)
        x = self.layer3_1_add(residual, x)
        x = self.relu3_2_second(x)

        # Layer 3 Block 3
        residual = x
        x = self.layer3_2_conv1(x)
        x = self.layer3_2_bn1(x)
        x = self.relu3_3(x)
        x = self.layer3_2_conv2(x)
        x = self.layer3_2_bn2(x)
        x = self.layer3_2_add(residual, x)
        x = self.relu3_3_second(x)

        # Pooling and classification
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def apply_mask_and_zero_out(state_dict):
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if "weight_mask" in key:
            # Extract corresponding weight_orig and weight_mask
            weight_orig_key = key.replace("weight_mask", "weight_orig")
            weight_mask = value
            weight_orig = state_dict[weight_orig_key]

            # Apply the mask: keep values in weight_orig corresponding to mask = 1, zero out others
            new_weights = weight_orig * weight_mask

            # Update the new_state_dict with the proper key (removing _orig)
            new_key = weight_orig_key.replace("weight_orig", "weight")
            new_state_dict[new_key] = new_weights
        else:
            # Keep other parameters (bias, running_mean, running_var, etc.) as they are
            if "weight_orig" not in key and "weight_mask" not in key:
                new_state_dict[key] = value
    return new_state_dict


def adjust_state_dict_keys(pretrained_state_dict):
        """
        This function adjusts the state_dict from the sparse model by adding '.weight', '.bias', etc.
        where needed, so they match the expected structure of `mobilenet_cob`.
        """
        new_state_dict = {}

        for key, value in pretrained_state_dict.items():
            if 'layer' in key:
                layer_num = key.split('.')[0]
                sublayer_num = key.split('.')[1]
                layer_type = key.split('.')[2]
                detail_layer = key.split('.')[3]

                if 'conv' in layer_type:
                    new_key = f"{layer_num}_{sublayer_num}_{layer_type}.{detail_layer}"
                    new_state_dict[new_key] = value
                elif 'bn' in layer_type:
                    new_key = f"{layer_num}_{sublayer_num}_{layer_type}.{detail_layer}"
                    new_state_dict[new_key] = value
                elif 'downsample' in layer_type:
                    bn_or_conv = key.split('.')[3]
                    # set nb_or_conv to 'conv' if it is equal to 0 else 'bn'
                    bn_or_conv = 'conv' if bn_or_conv == '0' else 'bn'
                    detail_layer = key.split('.')[4]
                    new_key = f"{layer_num}_{sublayer_num}_{layer_type}_{bn_or_conv}.{detail_layer}"
                    new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        return new_state_dict

class EfficientNetB0Flat(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0Flat, self).__init__()

        # Stem
        self._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(32)

        # Block 0
        self._blocks_0_depthwise_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self._blocks_0_bn1 = nn.BatchNorm2d(32)
        self._blocks_0_se_reduce = nn.Conv2d(32, 8, kernel_size=1)
        self._blocks_0_se_expand = nn.Conv2d(8, 32, kernel_size=1)
        self._blocks_0_project_conv = nn.Conv2d(32, 16, kernel_size=1, stride=1, bias=False)
        self._blocks_0_bn2 = nn.BatchNorm2d(16)

        # Block 1
        self._blocks_1_expand_conv = nn.Conv2d(16, 96, kernel_size=1, stride=1, bias=False)
        self._blocks_1_bn0 = nn.BatchNorm2d(96)
        self._blocks_1_depthwise_conv = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, groups=96, bias=False)
        self._blocks_1_bn1 = nn.BatchNorm2d(96)
        self._blocks_1_se_reduce = nn.Conv2d(96, 4, kernel_size=1)
        self._blocks_1_se_expand = nn.Conv2d(4, 96, kernel_size=1)
        self._blocks_1_project_conv = nn.Conv2d(96, 24, kernel_size=1, stride=1, bias=False)
        self._blocks_1_bn2 = nn.BatchNorm2d(24)

        # Block 2
        self._blocks_2_expand_conv = nn.Conv2d(24, 144, kernel_size=1, stride=1, bias=False)
        self._blocks_2_bn0 = nn.BatchNorm2d(144)
        self._blocks_2_depthwise_conv = nn.Conv2d(144, 144, kernel_size=3, stride=1, padding=1, groups=144, bias=False)
        self._blocks_2_bn1 = nn.BatchNorm2d(144)
        self._blocks_2_se_reduce = nn.Conv2d(144, 6, kernel_size=1)
        self._blocks_2_se_expand = nn.Conv2d(6, 144, kernel_size=1)
        self._blocks_2_project_conv = nn.Conv2d(144, 24, kernel_size=1, stride=1, bias=False)
        self._blocks_2_bn2 = nn.BatchNorm2d(24)

        # Block 3
        self._blocks_3_expand_conv = nn.Conv2d(24, 144, kernel_size=1, stride=1, bias=False)
        self._blocks_3_bn0 = nn.BatchNorm2d(144)
        self._blocks_3_depthwise_conv = nn.Conv2d(144, 144, kernel_size=5, stride=2, padding=2, groups=144, bias=False)
        self._blocks_3_bn1 = nn.BatchNorm2d(144)
        self._blocks_3_se_reduce = nn.Conv2d(144, 6, kernel_size=1)
        self._blocks_3_se_expand = nn.Conv2d(6, 144, kernel_size=1)
        self._blocks_3_project_conv = nn.Conv2d(144, 40, kernel_size=1, stride=1, bias=False)
        self._blocks_3_bn2 = nn.BatchNorm2d(40)

        # Block 4
        self._blocks_4_expand_conv = nn.Conv2d(40, 240, kernel_size=1, stride=1, bias=False)
        self._blocks_4_bn0 = nn.BatchNorm2d(240)
        self._blocks_4_depthwise_conv = nn.Conv2d(240, 240, kernel_size=5, stride=1, padding=2, groups=240, bias=False)
        self._blocks_4_bn1 = nn.BatchNorm2d(240)
        self._blocks_4_se_reduce = nn.Conv2d(240, 10, kernel_size=1)
        self._blocks_4_se_expand = nn.Conv2d(10, 240, kernel_size=1)
        self._blocks_4_project_conv = nn.Conv2d(240, 40, kernel_size=1, stride=1, bias=False)
        self._blocks_4_bn2 = nn.BatchNorm2d(40)

        # Block 5
        self._blocks_5_expand_conv = nn.Conv2d(40, 240, kernel_size=1, stride=1, bias=False)
        self._blocks_5_bn0 = nn.BatchNorm2d(240)
        self._blocks_5_depthwise_conv = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1, groups=240, bias=False)
        self._blocks_5_bn1 = nn.BatchNorm2d(240)
        self._blocks_5_se_reduce = nn.Conv2d(240, 10, kernel_size=1)
        self._blocks_5_se_expand = nn.Conv2d(10, 240, kernel_size=1)
        self._blocks_5_project_conv = nn.Conv2d(240, 80, kernel_size=1, stride=1, bias=False)
        self._blocks_5_bn2 = nn.BatchNorm2d(80)

        # Block 6
        self._blocks_6_expand_conv = nn.Conv2d(80, 480, kernel_size=1, stride=1, bias=False)
        self._blocks_6_bn0 = nn.BatchNorm2d(480)
        self._blocks_6_depthwise_conv = nn.Conv2d(480, 480, kernel_size=3, stride=1, padding=1, groups=480, bias=False)
        self._blocks_6_bn1 = nn.BatchNorm2d(480)
        self._blocks_6_se_reduce = nn.Conv2d(480, 20, kernel_size=1)
        self._blocks_6_se_expand = nn.Conv2d(20, 480, kernel_size=1)
        self._blocks_6_project_conv = nn.Conv2d(480, 80, kernel_size=1, stride=1, bias=False)
        self._blocks_6_bn2 = nn.BatchNorm2d(80)

        # Block 7
        self._blocks_7_expand_conv = nn.Conv2d(80, 480, kernel_size=1, stride=1, bias=False)
        self._blocks_7_bn0 = nn.BatchNorm2d(480)
        self._blocks_7_depthwise_conv = nn.Conv2d(480, 480, kernel_size=3, stride=1, padding=1, groups=480, bias=False)
        self._blocks_7_bn1 = nn.BatchNorm2d(480)
        self._blocks_7_se_reduce = nn.Conv2d(480, 20, kernel_size=1)
        self._blocks_7_se_expand = nn.Conv2d(20, 480, kernel_size=1)
        self._blocks_7_project_conv = nn.Conv2d(480, 80, kernel_size=1, stride=1, bias=False)
        self._blocks_7_bn2 = nn.BatchNorm2d(80)

        # Block 8
        self._blocks_8_expand_conv = nn.Conv2d(80, 480, kernel_size=1, stride=1, bias=False)
        self._blocks_8_bn0 = nn.BatchNorm2d(480)
        self._blocks_8_depthwise_conv = nn.Conv2d(480, 480, kernel_size=5, stride=1, padding=2, groups=480, bias=False)
        self._blocks_8_bn1 = nn.BatchNorm2d(480)
        self._blocks_8_se_reduce = nn.Conv2d(480, 20, kernel_size=1)
        self._blocks_8_se_expand = nn.Conv2d(20, 480, kernel_size=1)
        self._blocks_8_project_conv = nn.Conv2d(480, 112, kernel_size=1, stride=1, bias=False)
        self._blocks_8_bn2 = nn.BatchNorm2d(112)

        # Block 9
        self._blocks_9_expand_conv = nn.Conv2d(112, 672, kernel_size=1, stride=1, bias=False)
        self._blocks_9_bn0 = nn.BatchNorm2d(672)
        self._blocks_9_depthwise_conv = nn.Conv2d(672, 672, kernel_size=5, stride=1, padding=2, groups=672, bias=False)
        self._blocks_9_bn1 = nn.BatchNorm2d(672)
        self._blocks_9_se_reduce = nn.Conv2d(672, 28, kernel_size=1)
        self._blocks_9_se_expand = nn.Conv2d(28, 672, kernel_size=1)
        self._blocks_9_project_conv = nn.Conv2d(672, 112, kernel_size=1, stride=1, bias=False)
        self._blocks_9_bn2 = nn.BatchNorm2d(112)

        # Block 10
        self._blocks_10_expand_conv = nn.Conv2d(112, 672, kernel_size=1, stride=1, bias=False)
        self._blocks_10_bn0 = nn.BatchNorm2d(672)
        self._blocks_10_depthwise_conv = nn.Conv2d(672, 672, kernel_size=5, stride=1, padding=2, groups=672, bias=False)
        self._blocks_10_bn1 = nn.BatchNorm2d(672)
        self._blocks_10_se_reduce = nn.Conv2d(672, 28, kernel_size=1)
        self._blocks_10_se_expand = nn.Conv2d(28, 672, kernel_size=1)
        self._blocks_10_project_conv = nn.Conv2d(672, 112, kernel_size=1, stride=1, bias=False)
        self._blocks_10_bn2 = nn.BatchNorm2d(112)

        # Block 11
        self._blocks_11_expand_conv = nn.Conv2d(112, 672, kernel_size=1, stride=1, bias=False)
        self._blocks_11_bn0 = nn.BatchNorm2d(672)
        self._blocks_11_depthwise_conv = nn.Conv2d(672, 672, kernel_size=5, stride=1, padding=2, groups=672, bias=False)
        self._blocks_11_bn1 = nn.BatchNorm2d(672)
        self._blocks_11_se_reduce = nn.Conv2d(672, 28, kernel_size=1)
        self._blocks_11_se_expand = nn.Conv2d(28, 672, kernel_size=1)
        self._blocks_11_project_conv = nn.Conv2d(672, 192, kernel_size=1, stride=1, bias=False)
        self._blocks_11_bn2 = nn.BatchNorm2d(192)

        # Block 12
        self._blocks_12_expand_conv = nn.Conv2d(192, 1152, kernel_size=1, stride=1, bias=False)
        self._blocks_12_bn0 = nn.BatchNorm2d(1152)
        self._blocks_12_depthwise_conv = nn.Conv2d(1152, 1152, kernel_size=5, stride=1, padding=2, groups=1152, bias=False)
        self._blocks_12_bn1 = nn.BatchNorm2d(1152)
        self._blocks_12_se_reduce = nn.Conv2d(1152, 48, kernel_size=1)
        self._blocks_12_se_expand = nn.Conv2d(48, 1152, kernel_size=1)
        self._blocks_12_project_conv = nn.Conv2d(1152, 192, kernel_size=1, stride=1, bias=False)
        self._blocks_12_bn2 = nn.BatchNorm2d(192)

        # Block 13
        self._blocks_13_expand_conv = nn.Conv2d(192, 1152, kernel_size=1, stride=1, bias=False)
        self._blocks_13_bn0 = nn.BatchNorm2d(1152)
        self._blocks_13_depthwise_conv = nn.Conv2d(1152, 1152, kernel_size=5, stride=1, padding=2, groups=1152, bias=False)
        self._blocks_13_bn1 = nn.BatchNorm2d(1152)
        self._blocks_13_se_reduce = nn.Conv2d(1152, 48, kernel_size=1)
        self._blocks_13_se_expand = nn.Conv2d(48, 1152, kernel_size=1)
        self._blocks_13_project_conv = nn.Conv2d(1152, 192, kernel_size=1, stride=1, bias=False)
        self._blocks_13_bn2 = nn.BatchNorm2d(192)

        # Block 14
        self._blocks_14_expand_conv = nn.Conv2d(192, 1152, kernel_size=1, stride=1, bias=False)
        self._blocks_14_bn0 = nn.BatchNorm2d(1152)
        self._blocks_14_depthwise_conv = nn.Conv2d(1152, 1152, kernel_size=5, stride=1, padding=2, groups=1152, bias=False)
        self._blocks_14_bn1 = nn.BatchNorm2d(1152)
        self._blocks_14_se_reduce = nn.Conv2d(1152, 48, kernel_size=1)
        self._blocks_14_se_expand = nn.Conv2d(48, 1152, kernel_size=1)
        self._blocks_14_project_conv = nn.Conv2d(1152, 192, kernel_size=1, stride=1, bias=False)
        self._blocks_14_bn2 = nn.BatchNorm2d(192)

        # Block 15
        self._blocks_15_expand_conv = nn.Conv2d(192, 1152, kernel_size=1, stride=1, bias=False)
        self._blocks_15_bn0 = nn.BatchNorm2d(1152)
        self._blocks_15_depthwise_conv = nn.Conv2d(1152, 1152, kernel_size=3, stride=1, padding=1, groups=1152, bias=False)
        self._blocks_15_bn1 = nn.BatchNorm2d(1152)
        self._blocks_15_se_reduce = nn.Conv2d(1152, 48, kernel_size=1)
        self._blocks_15_se_expand = nn.Conv2d(48, 1152, kernel_size=1)
        self._blocks_15_project_conv = nn.Conv2d(1152, 320, kernel_size=1, stride=1, bias=False)
        self._blocks_15_bn2 = nn.BatchNorm2d(320)

        # Head
        self._conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(1280)

        # Fully Connected Layer
        self._fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 0
        residual = x
        x = self._blocks_0_depthwise_conv(x)
        x = self._blocks_0_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 0
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_0_se_reduce(x_se)
        x_se = F.relu(x_se)
        x_se = self._blocks_0_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_0_project_conv(x)
        x = self._blocks_0_bn2(x)
        # x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 1
        residual = x
        x = self._blocks_1_expand_conv(x)
        x = self._blocks_1_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_1_depthwise_conv(x)
        x = self._blocks_1_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 1
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_1_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_1_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_1_project_conv(x)
        x = self._blocks_1_bn2(x)
        # x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 2
        residual = x
        x = self._blocks_2_expand_conv(x)
        x = self._blocks_2_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_2_depthwise_conv(x)
        x = self._blocks_2_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 2
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_2_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_2_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_2_project_conv(x)
        x = self._blocks_2_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 3
        residual = x
        x = self._blocks_3_expand_conv(x)
        x = self._blocks_3_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_3_depthwise_conv(x)
        x = self._blocks_3_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 3
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_3_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_3_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_3_project_conv(x)
        x = self._blocks_3_bn2(x)
        # x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 4
        residual = x
        x = self._blocks_4_expand_conv(x)
        x = self._blocks_4_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_4_depthwise_conv(x)
        x = self._blocks_4_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 4
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_4_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_4_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_4_project_conv(x)
        x = self._blocks_4_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 5
        residual = x
        x = self._blocks_5_expand_conv(x)
        x = self._blocks_5_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_5_depthwise_conv(x)
        x = self._blocks_5_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 5
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_5_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_5_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_5_project_conv(x)
        x = self._blocks_5_bn2(x)
        # x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 6
        residual = x
        x = self._blocks_6_expand_conv(x)
        x = self._blocks_6_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_6_depthwise_conv(x)
        x = self._blocks_6_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 6
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_6_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_6_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_6_project_conv(x)
        x = self._blocks_6_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 7
        residual = x
        x = self._blocks_7_expand_conv(x)
        x = self._blocks_7_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_7_depthwise_conv(x)
        x = self._blocks_7_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 7
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_7_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_7_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_7_project_conv(x)
        x = self._blocks_7_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 8
        residual = x
        x = self._blocks_8_expand_conv(x)
        x = self._blocks_8_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_8_depthwise_conv(x)
        x = self._blocks_8_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 8
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_8_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_8_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_8_project_conv(x)
        x = self._blocks_8_bn2(x)
        # x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 9
        residual = x
        x = self._blocks_9_expand_conv(x)
        x = self._blocks_9_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_9_depthwise_conv(x)
        x = self._blocks_9_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 9
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_9_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_9_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_9_project_conv(x)
        x = self._blocks_9_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 10
        residual = x
        x = self._blocks_10_expand_conv(x)
        x = self._blocks_10_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_10_depthwise_conv(x)
        x = self._blocks_10_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 10
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_10_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_10_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_10_project_conv(x)
        x = self._blocks_10_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 11
        residual = x
        x = self._blocks_11_expand_conv(x)
        x = self._blocks_11_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_11_depthwise_conv(x)
        x = self._blocks_11_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 11
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_11_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_11_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_11_project_conv(x)
        x = self._blocks_11_bn2(x)
        # x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 12
        residual = x
        x = self._blocks_12_expand_conv(x)
        x = self._blocks_12_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_12_depthwise_conv(x)
        x = self._blocks_12_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 12
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_12_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_12_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_12_project_conv(x)
        x = self._blocks_12_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 13
        residual = x
        x = self._blocks_13_expand_conv(x)
        x = self._blocks_13_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_13_depthwise_conv(x)
        x = self._blocks_13_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 13
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_13_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_13_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_13_project_conv(x)
        x = self._blocks_13_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 14
        residual = x
        x = self._blocks_14_expand_conv(x)
        x = self._blocks_14_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_14_depthwise_conv(x)
        x = self._blocks_14_bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # SE for Block 14
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_14_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_14_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_14_project_conv(x)
        x = self._blocks_14_bn2(x)
        x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Block 15
        residual = x
        x = self._blocks_15_expand_conv(x)
        x = self._blocks_15_bn0(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)
        x = self._blocks_15_depthwise_conv(x)
        x = self._blocks_15_bn1(x)
        # x = F.relu(x)
        # swish activation
        x = x * torch.sigmoid(x)

        # SE for Block 15
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self._blocks_15_se_reduce(x_se)
        # x_se = F.relu(x_se)
        x_se = x_se * torch.sigmoid(x_se)
        x_se = self._blocks_15_se_expand(x_se)
        x = x * torch.sigmoid(x_se)

        x = self._blocks_15_project_conv(x)
        x = self._blocks_15_bn2(x)
        # x += residual  # Add the skip connection from the residual
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        # x = F.relu(x)
        x = x * torch.sigmoid(x)

        # Pooling and Fully Connected
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self._fc(x)

        return x

def map_pretrained_weights(pretrained_state_dict):
    """
    Map weights from a pretrained model to a custom model with flat naming conventions.
    
    Args:
    - pretrained_state_dict: state_dict from the pretrained model with original layer names.
    - model_state_dict: state_dict of the flat model with custom layer names.
    
    Returns:
    - A state_dict that can be loaded into the flat model.
    """
    new_state_dict = {}

    # Map the pretrained model's layer names to the flat model's layer names
    new_state_dict = {}

    for old_key in pretrained_state_dict.keys():
        # Replace '.' with '_'

        if 'blocks' in old_key:
            # _blocks.0._depthwise_conv.weight -> _blocks_0_depthwise_conv.weight
            splits = old_key.split('.')
            if len(splits) == 4:
                # check if splits[2] starts with '_'
                if splits[2][0] == '_':
                    new_key = f"{splits[0]}_{splits[1]}{splits[2]}.{splits[3]}"
                else:
                    new_key = f"{splits[0]}_{splits[1]}_{splits[2]}.{splits[3]}"

        else:
            new_key = old_key
        
        # Add the updated key to the new dictionary
        new_state_dict[new_key] = pretrained_state_dict[old_key]
    
    return new_state_dict

if __name__ == '__main__':
    from torchsummary import summary
    from tests.model_test import test_teleport
    import copy
    import sys

    # sample input
    # dummy imagenet input_data
    input_data = torch.randn(1, 3, 224, 224)

    from efficientnet_pytorch import EfficientNet
    # import Swish
    from efficientnet_pytorch.utils import Swish, MemoryEfficientSwish
    from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.eval()
    model = model.cpu()

    # hook all the activations functions of the model to compute range of input (max - min)
    def activation_hook(name, activation_stats, activations_output=None, layer_idx=None):
        def hook(module, input, output):
            input_tensor = input[0]
            activation_stats[name] = {'min': input_tensor.min().item(), 'max': input_tensor.max().item()}
            if activations_output is not None:
                # key of the dict is the layer number (extracted from the name)
                activations_output[layer_idx] = output
        return hook
    
    activation = {}
    # for name, layer in model.named_modules():
    activation_stats = {}
    activations_quant = {}
    hook_handles = []
    layer_idx = 0
    for i, layer in enumerate(model.children()):
        if isinstance(layer, (nn.ReLU, ReLUCOB,nn.Sigmoid,Swish, MemoryEfficientSwish)):
            handle = layer.register_forward_hook(activation_hook(f'relu_{i}', activation_stats=activation_stats, activations_output=activations_quant,layer_idx=layer_idx))
            hook_handles.append(handle)
    
    pred_model = model(input_data)
    print("====================================================")
    # print the all activations ranges
    for key, value in activation_stats.items():
        print(key, value)

    for name, param in model.named_parameters():
        print(name, "\t", param.size())
    print("====================================================")

    # model_cob = swap_model_modules_for_COB_modules(model)
    model_cob = EfficientNetB0Flat(num_classes=1000)
    for name, param in model_cob.named_parameters():
        print(name, "\t", param.size())
    print("====================================================")

    # load the model weights on model_cob  
    new_state_dict = map_pretrained_weights(model.state_dict())
    model_cob.load_state_dict(new_state_dict, strict=True)
    model_cob.eval()
    print(model_cob)
    pred_cob = model_cob(input_data)

    # print diff between the two models
    diff = torch.mean(torch.abs(pred_model - pred_cob)).item()
    print("DIFF: ", diff)
    print("Arg max mode_pred: ", torch.argmax(pred_model), "Arg max pred_cob: ", torch.argmax(pred_cob))


