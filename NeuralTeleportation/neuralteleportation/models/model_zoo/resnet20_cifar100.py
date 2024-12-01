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

if __name__ == '__main__':
    from torchsummary import summary
    from tests.model_test import test_teleport
    import copy
    import sys

    # sample input
    input_data = torch.randn(1, 3, 32, 32)

    # resent20_cifar100_cob = Resnet20Cifar100Cob(BasicBlockCob, [3, 3, 3], num_classes=100)
    resnet20_cifar100_cob = Resnet20Cifar100CobFlat(num_classes=100)
    for name, param in resnet20_cifar100_cob.named_parameters():
        print(name, "\t", param.size())
    print("====================================================")
    resnet20_cifar100_cob.eval()

    # load pretrained model
    model_path = "/Users/mm6322/Downloads/unpruned.pth.tar"
    pretrained_model = torch.load(model_path, map_location='cpu')
    pretrained_model = apply_mask_and_zero_out(pretrained_model)
    new_state_dict = adjust_state_dict_keys(pretrained_model)
    resnet20_cifar100_cob.load_state_dict(new_state_dict, strict=True)
    
    # predict with the pretrained model
    pred_cob = resnet20_cifar100_cob(input_data)
    # teleport the model
    resnet20_cob_teleported = NeuralTeleportationModel(network=resnet20_cifar100_cob, input_shape=(1,3, 32, 32))
    resnet20_cob_teleported.eval()
    pred_resnet20_not_teleported = resnet20_cifar100_cob(input_data)
    # teleport the model
    resnet20_cob_teleported = resnet20_cob_teleported.random_teleport(cob_range=0.8, sampling_type='intra_landscape')
    pred_resnet20_teleported = resnet20_cob_teleported(input_data)

    diff_with_before_teleportation = torch.mean(torch.abs(pred_resnet20_teleported - pred_resnet20_not_teleported)).item()
    diff_with_pretrained = torch.mean(torch.abs(pred_resnet20_teleported - pred_cob)).item()
    print("DIFF WITH BEFORE TELEPORTATION: ", diff_with_before_teleportation)
    print("DIFF WITH PRETRAINED: ", diff_with_pretrained)
    print("====================================================")
    assert diff_with_before_teleportation < 1e-3
    assert diff_with_pretrained < 1e-3

