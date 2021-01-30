import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools
from networks.util import conv3x3, Bottleneck
from networks.asp_oc_block import ASP_OC_Module
from inplace_abn.bn import InPlaceABNSync, InPlaceABN
from torchsummary import summary

ABN_module = InPlaceABN
BatchNorm2d = functools.partial(ABN_module, activation='none')
affine_par = True


class ResNet(nn.Module):
    """
    Basic ResNet101.
    """
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=1, dilation=2, padding=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        self.features = []
        x = (x - 0.45) / 0.225
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        self.features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.features.append(x)
        x = self.layer2(x)
        self.features.append(x)
        x = self.layer3(x)
        self.features.append(x)
        # x_dsn = self.dsn(x)
        x = self.layer4(x)
        self.features.append(x)

        return self.features


class ResNet_context(nn.Module):
    """
    ResNet101 + self-attention
    """
    def __init__(self, num_classes, disable_self_attn, pretrained):
        self.num_ch_enc = np.array([128, 256, 512, 1024, 2048])
        self.disable_self_attn = disable_self_attn
        super(ResNet_context, self).__init__()

        self.basedir = os.path.dirname(os.path.abspath(__file__))
        self.resnet_model = ResNet(Bottleneck, [3, 4, 23, 3])
        if pretrained:
            pretrained_weights = torch.load(os.path.join(self.basedir, '../splits/resnet101-imagenet.pth'))
            model_dict = self.resnet_model.state_dict()
            self.resnet_model.load_state_dict({k: v for k, v in pretrained_weights.items() if k in model_dict})

        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ABN_module(512),
            ASP_OC_Module(512, 256, disable_self_attn=self.disable_self_attn)
        )
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ABN_module(512),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        all_features = self.resnet_model(x)
        all_features[-1] = self.context(all_features[-1])
        all_features[-1] = self.cls(all_features[-1])

        return all_features


def get_resnet101_asp_oc_dsn(num_classes=128, disable_self_attn=False, pretrained=True):
    model = ResNet_context(num_classes, disable_self_attn, pretrained)
    return model


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    pretrained_weights = torch.load(os.path.join(base, '../splits/resnet101-imagenet.pth'))
    print(pretrained_weights.keys())


