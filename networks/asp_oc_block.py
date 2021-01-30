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
from inplace_abn.bn import InPlaceABNSync, InPlaceABN

ABN_module = InPlaceABN
BatchNorm2d = functools.partial(ABN_module, activation='none')

from networks.base_oc_block import BaseOC_Context_Module


class ASP_OC_Module(nn.Module):
    """
    Network to perform Atrous Spatial Pyramid Pooling (ASPP).
    """
    def __init__(self, features, out_features=256, dilations=(12, 24, 36), disable_self_attn=False):
        super(ASP_OC_Module, self).__init__()
        self.disable_self_attn = disable_self_attn
        self.context_oc = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
            ABN_module(out_features),
            BaseOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                  key_channels=out_features // 2, value_channels=out_features,
                                  dropout=0, sizes=([2])))
        self.context = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
            ABN_module(out_features))

        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   ABN_module(out_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            ABN_module(out_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            ABN_module(out_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            ABN_module(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features * 2, kernel_size=1, padding=0, dilation=1, bias=False),
            ABN_module(out_features * 2),
            nn.Dropout2d(0.1)
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        """
        Concatenate parallel convolution layers with different dilation rates
        to perform ASPP.
        """
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        if self.disable_self_attn:
            feat1 = self.context(x)
        else:
            feat1 = self.context_oc(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output
