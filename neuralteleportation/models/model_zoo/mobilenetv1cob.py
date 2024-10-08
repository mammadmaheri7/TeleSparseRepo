import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import Conv2dCOB, LinearCOB, BatchNorm2dCOB
from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.pooling import AdaptiveAvgPool2dCOB,AvgPool2dCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
import sys
sys.modules['models'] = sys.modules[__name__]
sys.modules['models.mobilenetv1'] = sys.modules[__name__]

class BlockCOB(nn.Module):
    '''Depthwise conv + Pointwise conv with COB layers'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockCOB, self).__init__()
        self.conv1 = Conv2dCOB(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = BatchNorm2dCOB(in_planes)
        self.relu1 = ReLUCOB(inplace=False)
        self.conv2 = Conv2dCOB(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = BatchNorm2dCOB(out_planes)
        self.relu2 = ReLUCOB(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class MobileNetV1COB(nn.Module):
    # Configuration with stride information for certain layers
    default_cfg = [32,       64,     (128, 2), 128,    (256, 2),  256,     (512, 2),  512,     512,       512,      512,      512,     (1024, 2), 1024]
                  
    def __init__(self, num_classes=10, cfg=None):
        super(MobileNetV1COB, self).__init__()
        # define the configuration
        self.cfg = self.default_cfg if cfg is None else cfg
        first_channel = self.cfg[0] if isinstance(self.cfg[0], int) else self.cfg[0][0]
        # define the layers
        self.conv1 = Conv2dCOB(3, first_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2dCOB(first_channel)
        self.relu1 = ReLUCOB(inplace=False)

        layers = self._make_layers(in_planes=first_channel, cfg=self.cfg[1:])

        # layer0
        self.layer0_conv1 = layers[0].conv1
        self.layer0_bn1 = layers[0].bn1
        self.layer0_relu1 = layers[0].relu1
        self.layer0_conv2 = layers[0].conv2
        self.layer0_bn2 = layers[0].bn2
        self.layer0_relu2 = layers[0].relu2

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

        # last layer (classifier)
        self.avgpool = AvgPool2dCOB(kernel_size=2)
        self.flatten = FlattenCOB()
        # self.fc = LinearCOB(1024, num_classes)
        last_dim = self.cfg[-1] if isinstance(self.cfg[-1], int) else self.cfg[-1][0]
        self.linear = LinearCOB(last_dim, num_classes)

    def _make_layers(self, in_planes, cfg=None):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(BlockCOB(in_planes, out_planes, stride))
            in_planes = out_planes
        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

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
        out = self.layer7_bn2(out)
        out = self.layer7_relu2(out)


        # layer8
        out = self.layer8_conv1(out)
        out = self.layer8_bn1(out)
        out = self.layer8_relu1(out)
        out = self.layer8_conv2(out)
        out = self.layer8_bn2(out)
        out = self.layer8_relu2(out)

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

        # classifier
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out
    
def adjust_state_dict_keys(pretrained_state_dict):
        """
        This function adjusts the state_dict from the sparse model by adding '.weight', '.bias', etc.
        where needed, so they match the expected structure of `mobilenet_cob`.
        """
        new_state_dict = {}

        for key, value in pretrained_state_dict.items():
            if 'layer' in key:
                layer_num = key.split('.')[1]
                layer_type = key.split('.')[2]
                detail_layer = key.split('.')[3]

                if 'conv' in layer_type:
                    new_key = f"layer{layer_num}_{layer_type}.{detail_layer}"
                    new_state_dict[new_key] = value
                elif 'bn' in layer_type:
                    new_key = f"layer{layer_num}_{layer_type}.{detail_layer}"
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
    
    # load model weigths from file (trained by torch-pruning project)
    path = 'mobilenetv1_sparse_best.pth'
    pretrained_sparse_model = torch.load(path, map_location='cpu')['model']
    
    # path = 'mobilenetv1_sparse_best_state_dict.pth'
    # pretrained_sparse_model_state_dict = torch.load(path, map_location='cpu')

    # load the state_dict of the pretrained model
    # pretrained_sparse_model.load_state_dict(pretrained_sparse_model_state_dict,strict=True)

    print(pretrained_sparse_model)
    print("====================================================")

    # retrieve the cfg of the pretrained model
    retrieved_cfg = []
    for module in pretrained_sparse_model.children(): 
        if isinstance(module, nn.Conv2d):
            retrieved_cfg.append((module.out_channels, module.stride[0]))
        elif isinstance(module, nn.Sequential):
            for sub_module in module.children():
                if isinstance(sub_module, nn.Conv2d):
                    retrieved_cfg.append((sub_module.out_channels, sub_module.stride[0]))
                elif isinstance(sub_module, Block):
                    retrieved_cfg.append((sub_module.conv2.out_channels, sub_module.conv1.stride[0]))
    print(retrieved_cfg)
    print("====================================================")

    mobilenet_cob = MobileNetV1COB(num_classes=10, cfg=retrieved_cfg)
    print(mobilenet_cob)
    print("====================================================")

    sparse_state_dict = pretrained_sparse_model.state_dict()

    print(" === sparse_state_dict === ")
    new_sparse_state_dict = adjust_state_dict_keys(sparse_state_dict)
    mobilenet_cob.load_state_dict(new_sparse_state_dict, strict=True)

    mobilenet_cob.eval()
    pretrained_sparse_model.eval()

    # set eps to 1e-5
    mobilenet_cob.bn1.eps=1e-5
    # do the set for all bn in mobilenet_cob layers
    for l_index in range(13):
        cob_bn = getattr(mobilenet_cob, f"layer{l_index}_bn1")
        cob_bn.eps = 1e-5
        setattr(cob_bn, "eps", 1e-5)
        cob_bn = getattr(mobilenet_cob, f"layer{l_index}_bn2")
        cob_bn.eps = 1e-5

    # find difference of the pretrained model and the model with COB layers
    input = torch.randn(1, 3, 32, 32)

    pretrained_conv1 = pretrained_sparse_model.conv1
    cob_conv1 = mobilenet_cob.conv1
    out_pretrained = pretrained_conv1(input)
    out_cob = cob_conv1(input)
    print("DIFF_CONV1: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    pretrained_bn1 = pretrained_sparse_model.bn1
    cob_bn1 = mobilenet_cob.bn1
    # make the eps of cob_bn1 to be the same as the pretrained_bn1
    # cob_bn1.eps = pretrained_bn1.eps
    out_pretrained = pretrained_bn1(out_pretrained)
    out_cob = cob_bn1(out_cob)
    # do relu
    out_pretrained = F.relu(out_pretrained)
    out_cob = mobilenet_cob.relu1(out_cob)
    print("MIN BN1: ", torch.min(out_pretrained).item(), torch.min(out_cob).item())
    print("DIFF_BN1: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    pretrained_layer0_conv1 = pretrained_sparse_model.layers[0].conv1
    cob_layer0_conv1 = mobilenet_cob.layer0_conv1
    out_pretrained = pretrained_layer0_conv1(out_pretrained)
    out_cob = cob_layer0_conv1(out_cob)
    # do relu
    out_pretrained = F.relu(out_pretrained)
    out_cob = mobilenet_cob.relu1(out_cob)
    print("DIFF_LAYER0_CONV1: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    pretrained_layer0_bn1 = pretrained_sparse_model.layers[0].bn1
    cob_layer0_bn1 = mobilenet_cob.layer0_bn1
    out_pretrained = pretrained_layer0_bn1(out_pretrained)
    out_cob = cob_layer0_bn1(out_cob)
    # do relu
    out_pretrained = F.relu(out_pretrained)
    out_cob = mobilenet_cob.layer0_relu1(out_cob)
    print("DIFF_LAYER0_BN1: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    pretrained_layer0_conv2 = pretrained_sparse_model.layers[0].conv2
    cob_layer0_conv2 = mobilenet_cob.layer0_conv2
    out_pretrained = pretrained_layer0_conv2(out_pretrained)
    out_cob = cob_layer0_conv2(out_cob)
    print("DIFF_LAYER0_CONV2: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    pretrained_layer0_bn2 = pretrained_sparse_model.layers[0].bn2
    cob_layer0_bn2 = mobilenet_cob.layer0_bn2
    out_pretrained = pretrained_layer0_bn2(out_pretrained)
    out_cob = cob_layer0_bn2(out_cob)
    # do relu
    out_pretrained = F.relu(out_pretrained)
    out_cob = mobilenet_cob.layer0_relu2(out_cob)
    print("DIFF_LAYER0_BN2: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    for l_index in range(1, 13):
        pretrained_layer_conv1 = pretrained_sparse_model.layers[l_index].conv1
        cob_layer_conv1 = getattr(mobilenet_cob, f"layer{l_index}_conv1")
        out_pretrained = pretrained_layer_conv1(out_pretrained)
        out_cob = cob_layer_conv1(out_cob)
        print(f"DIFF_LAYER{l_index}_CONV1: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
        pretrained_layer_bn1 = pretrained_sparse_model.layers[l_index].bn1
        cob_layer_bn1 = getattr(mobilenet_cob, f"layer{l_index}_bn1")
        out_pretrained = pretrained_layer_bn1(out_pretrained)
        out_cob = cob_layer_bn1(out_cob)
        # do relu
        out_pretrained = F.relu(out_pretrained)
        out_cob = getattr(mobilenet_cob, f"layer{l_index}_relu1")(out_cob)
        print(f"DIFF_LAYER{l_index}_BN1: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
        pretrained_layer_conv2 = pretrained_sparse_model.layers[l_index].conv2
        cob_layer_conv2 = getattr(mobilenet_cob, f"layer{l_index}_conv2")
        out_pretrained = pretrained_layer_conv2(out_pretrained)
        out_cob = cob_layer_conv2(out_cob)
        print(f"DIFF_LAYER{l_index}_CONV2: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
        pretrained_layer_bn2 = pretrained_sparse_model.layers[l_index].bn2
        cob_layer_bn2 = getattr(mobilenet_cob, f"layer{l_index}_bn2")
        out_pretrained = pretrained_layer_bn2(out_pretrained)
        out_cob = cob_layer_bn2(out_cob)
        # do relu
        out_pretrained = F.relu(out_pretrained)
        out_cob = getattr(mobilenet_cob, f"layer{l_index}_relu2")(out_cob)
        print(f"DIFF_LAYER{l_index}_BN2: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    
    cob_avgpool = mobilenet_cob.avgpool
    # out_pretrained = pretrained_avgpool(out_pretrained)
    out_pretrained = F.avg_pool2d(out_pretrained, 2)
    out_cob = cob_avgpool(out_cob)
    print("DIFF_AVGPOOL: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())
    # flatten
    out_pretrained = out_pretrained.view(out_pretrained.size(0), -1)
    out_cob = mobilenet_cob.flatten(out_cob)
    pretrained_linear = pretrained_sparse_model.linear
    cob_linear = mobilenet_cob.linear
    out_pretrained = pretrained_linear(out_pretrained)
    out_cob = cob_linear(out_cob)
    print("DIFF_LINEAR: ", torch.mean(torch.abs(out_pretrained - out_cob)).item())

    print(" 1 ====================================================")
    print(out_cob)

    pred_cob = mobilenet_cob(input)
    pred_pretrained = pretrained_sparse_model(input)
    print("DIFF IMPORTANT: ", torch.mean(torch.abs(pred_cob - pred_pretrained)).item())

    print(" 2 ====================================================")
    print(pred_cob)
    print(" 3 ====================================================")
    print(pred_pretrained)
    print("====================================================")
    print("====================================================")
    input_data = torch.randn(1, 3, 32, 32)
    # mobilenet_cob.train()
    # pretrained_sparse_model.train()
    pred_pretrained = pretrained_sparse_model(input_data)
    pred_org = mobilenet_cob(input_data)
    model = NeuralTeleportationModel(network=mobilenet_cob, input_shape=(1,3, 32, 32))
    model.eval()
    model = model.random_teleport(cob_range=0.8, sampling_type='intra_landscape')
    pred_cob = model(input_data)
    print("DIFF WITH BEFORE TELEPORTATION: ", torch.mean(torch.abs(pred_cob - pred_org)).item())
    print("DIFF WITH PRETRAINED: ", torch.mean(torch.abs(pred_cob - pred_pretrained)).item())
    


