import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import Conv2dCOB, LinearCOB, BatchNorm2dCOB
from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.pooling import AdaptiveAvgPool2dCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

class BlockCOB(nn.Module):
    '''Depthwise conv + Pointwise conv with COB layers'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockCOB, self).__init__()
        self.conv1 = Conv2dCOB(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = BatchNorm2dCOB(in_planes)
        self.relu1 = ReLUCOB(inplace=True)
        self.conv2 = Conv2dCOB(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = BatchNorm2dCOB(out_planes)
        self.relu2 = ReLUCOB(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # out = self.relu1(self.bn1(self.conv1(x)))
        # out = self.relu2(self.bn2(self.conv2(out)))
        return out

class MobileNetV1COB(nn.Module):
    # Configuration with stride information for certain layers
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetV1COB, self).__init__()
        self.conv1 = Conv2dCOB(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2dCOB(32)
        self.relu1 = ReLUCOB(inplace=True)
        # self.layers = self._make_layers(in_planes=32)
        layers = self._make_layers(in_planes=32)
        # self.layer0 = layers[0]

        # layer0
        self.layer0_conv1 = layers[0].conv1
        self.layer0_bn1 = layers[0].bn1
        self.layer0_relu1 = layers[0].relu1
        self.layer0_conv2 = layers[0].conv2
        self.layer0_bn2 = layers[0].bn2
        self.layer0_relu2 = layers[0].relu2
        # self.layer1 = layers[1]
        # self.layer2 = layers[2]
        # self.layer3 = layers[3]
        # self.layer4 = layers[4]
        # self.layer5 = layers[5]
        # self.layer6 = layers[6]
        # self.layer7 = layers[7]
        # self.layer8 = layers[8]
        # self.layer9 = layers[9]
        # self.layer10 = layers[10]
        # self.layer11 = layers[11]
        # self.layer12 = layers[12]

        # layer1
        self.layer1_conv1 = layers[1].conv1
        self.layer1_bn1 = layers[1].bn1
        self.layer1_relu1 = layers[1].relu1
        self.layer1_conv2 = layers[1].conv2
        self.layer1_bn2 = layers[1].bn2
        self.layer1_relu2 = layers[1].relu2

        # layer2
        self.layer2_conv1 = layers[2].conv1
        self.layer2_bn1 = layers[2].bn1
        self.layer2_relu1 = layers[2].relu1
        self.layer2_conv2 = layers[2].conv2
        self.layer2_bn2 = layers[2].bn2
        self.layer2_relu2 = layers[2].relu2

        # layer3
        self.layer3_conv1 = layers[3].conv1
        self.layer3_bn1 = layers[3].bn1
        self.layer3_relu1 = layers[3].relu1
        self.layer3_conv2 = layers[3].conv2
        self.layer3_bn2 = layers[3].bn2
        self.layer3_relu2 = layers[3].relu2

        # layer4
        self.layer4_conv1 = layers[4].conv1
        self.layer4_bn1 = layers[4].bn1
        self.layer4_relu1 = layers[4].relu1
        self.layer4_conv2 = layers[4].conv2
        self.layer4_bn2 = layers[4].bn2
        self.layer4_relu2 = layers[4].relu2

        # layer5
        self.layer5_conv1 = layers[5].conv1
        self.layer5_bn1 = layers[5].bn1
        self.layer5_relu1 = layers[5].relu1
        self.layer5_conv2 = layers[5].conv2
        self.layer5_bn2 = layers[5].bn2
        self.layer5_relu2 = layers[5].relu2

        # layer6
        self.layer6_conv1 = layers[6].conv1
        self.layer6_bn1 = layers[6].bn1
        self.layer6_relu1 = layers[6].relu1
        self.layer6_conv2 = layers[6].conv2
        self.layer6_bn2 = layers[6].bn2
        self.layer6_relu2 = layers[6].relu2

        # layer7
        self.layer7_conv1 = layers[7].conv1
        self.layer7_bn1 = layers[7].bn1
        self.layer7_relu1 = layers[7].relu1
        self.layer7_conv2 = layers[7].conv2
        self.layer7_bn2 = layers[7].bn2
        self.layer7_relu2 = layers[7].relu2

        # layer8
        self.layer8_conv1 = layers[8].conv1
        self.layer8_bn1 = layers[8].bn1
        self.layer8_relu1 = layers[8].relu1
        self.layer8_conv2 = layers[8].conv2
        self.layer8_bn2 = layers[8].bn2
        self.layer8_relu2 = layers[8].relu2

        # layer9
        self.layer9_conv1 = layers[9].conv1
        self.layer9_bn1 = layers[9].bn1
        self.layer9_relu1 = layers[9].relu1
        self.layer9_conv2 = layers[9].conv2
        self.layer9_bn2 = layers[9].bn2
        self.layer9_relu2 = layers[9].relu2

        # layer10
        self.layer10_conv1 = layers[10].conv1
        self.layer10_bn1 = layers[10].bn1
        self.layer10_relu1 = layers[10].relu1
        self.layer10_conv2 = layers[10].conv2
        self.layer10_bn2 = layers[10].bn2
        self.layer10_relu2 = layers[10].relu2

        # layer11
        self.layer11_conv1 = layers[11].conv1
        self.layer11_bn1 = layers[11].bn1
        self.layer11_relu1 = layers[11].relu1
        self.layer11_conv2 = layers[11].conv2
        self.layer11_bn2 = layers[11].bn2
        self.layer11_relu2 = layers[11].relu2

        # layer12
        self.layer12_conv1 = layers[12].conv1
        self.layer12_bn1 = layers[12].bn1
        self.layer12_relu1 = layers[12].relu1
        self.layer12_conv2 = layers[12].conv2
        self.layer12_bn2 = layers[12].bn2
        self.layer12_relu2 = layers[12].relu2

        # self.avgpool = AdaptiveAvgPool2dCOB((1, 1))
        # self.flatten = FlattenCOB()
        # self.fc = LinearCOB(1024, num_classes)
        # self.fc = nn.Linear(1024, num_classes)

        self.avgpool = AdaptiveAvgPool2dCOB((1, 1))
        self.flatten = FlattenCOB()
        self.fc = LinearCOB(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(BlockCOB(in_planes, out_planes, stride))
            in_planes = out_planes
        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        # out = self.layers(out)
        # for layer in self.layers:
        #     out = layer(out)
        # out = self.layer0(out)

        # layer0
        out = self.layer0_conv1(out)
        out = self.layer0_bn1(out)
        out = self.layer0_relu1(out)
        out = self.layer0_conv2(out)
        out = self.layer0_bn2(out)
        out = self.layer0_relu2(out)

        # layer1
        out = self.layer1_conv1(out)
        out = self.layer1_bn1(out)
        out = self.layer1_relu1(out)
        out = self.layer1_conv2(out)
        out = self.layer1_bn2(out)
        out = self.layer1_relu2(out)

        # layer2
        out = self.layer2_conv1(out)
        out = self.layer2_bn1(out)
        out = self.layer2_relu1(out)
        out = self.layer2_conv2(out)
        out = self.layer2_bn2(out)
        out = self.layer2_relu2(out)

        # layer3
        out = self.layer3_conv1(out)
        out = self.layer3_bn1(out)
        out = self.layer3_relu1(out)
        out = self.layer3_conv2(out)
        out = self.layer3_bn2(out)
        out = self.layer3_relu2(out)

        # layer4
        out = self.layer4_conv1(out)
        out = self.layer4_bn1(out)
        out = self.layer4_relu1(out)
        out = self.layer4_conv2(out)
        out = self.layer4_bn2(out)
        out = self.layer4_relu2(out)

        # layer5
        out = self.layer5_conv1(out)
        out = self.layer5_bn1(out)
        out = self.layer5_relu1(out)
        out = self.layer5_conv2(out)
        out = self.layer5_bn2(out)
        out = self.layer5_relu2(out)

        # layer6
        out = self.layer6_conv1(out)
        out = self.layer6_bn1(out)
        out = self.layer6_relu1(out)
        out = self.layer6_conv2(out)
        out = self.layer6_bn2(out)
        out = self.layer6_relu2(out)

        # layer7
        out = self.layer7_conv1(out)
        out = self.layer7_bn1(out)
        out = self.layer7_relu1(out)
        out = self.layer7_conv2(out)

        # layer8
        out = self.layer8_conv1(out)
        out = self.layer8_bn1(out)
        out = self.layer8_relu1(out)
        out = self.layer8_conv2(out)
        out = self.layer8_bn2(out)
        out = self.layer8_relu2(out)

        # layer9
        # layer9
        out = self.layer9_conv1(out)
        out = self.layer9_bn1(out)
        out = self.layer9_relu1(out)
        out = self.layer9_conv2(out)
        out = self.layer9_bn2(out)
        out = self.layer9_relu2(out)

        # layer10
        out = self.layer10_conv1(out)
        out = self.layer10_bn1(out)
        out = self.layer10_relu1(out)
        out = self.layer10_conv2(out)
        out = self.layer10_bn2(out)
        out = self.layer10_relu2(out)

        # layer11
        out = self.layer11_conv1(out)
        out = self.layer11_bn1(out)
        out = self.layer11_relu1(out)
        out = self.layer11_conv2(out)
        out = self.layer11_bn2(out)
        out = self.layer11_relu2(out)

        # layer12
        out = self.layer12_conv1(out)
        out = self.layer12_bn1(out)
        out = self.layer12_relu1(out)
        out = self.layer12_conv2(out)
        out = self.layer12_bn2(out)
        out = self.layer12_relu2(out)


        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)
        # out = self.layer8(out)
        # out = self.layer9(out)
        # out = self.layer10(out)
        # out = self.layer11(out)
        # out = self.layer12(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    from torchsummary import summary
    from tests.model_test import test_teleport
    import copy

    mobilenet_cob = MobileNetV1COB(num_classes=10)
    original_model = copy.deepcopy(mobilenet_cob)
    random_input = torch.randn(1, 3, 32, 32)
    output = mobilenet_cob(random_input)
    # summary(mobilenet_cob, (3, 32, 32), device='cpu')
    test_teleport(mobilenet_cob, (1,3, 32, 32), verbose=True)
    # print(mobilenet_cob)
    # print(type(mobilenet_cob))
    # model = NeuralTeleportationModel(mobilenet_cob, input_shape=(1,3, 32, 32))

    print("====================================================")
    input_data = torch.randn(1, 3, 32, 32)
    pred_org = original_model(input_data)
    model = NeuralTeleportationModel(network=mobilenet_cob, input_shape=(1,3, 32, 32))
    model = model.random_teleport(cob_range=0.8, sampling_type='intra_landscape')
    pred_cob = model(input_data)
    print("DIFF: ", torch.mean(torch.abs(pred_org - pred_cob)).item())


