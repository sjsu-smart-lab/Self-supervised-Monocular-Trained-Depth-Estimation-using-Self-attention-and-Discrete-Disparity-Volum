from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import warnings


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True, use_norm=False):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.pad(x)

        if self.use_norm:
            out = self.norm(out)

        out = self.conv(out)

        return out


class SoftAttnDepth(nn.Module):
    def __init__(self, alpha=0.01, beta=1.0, dim=1, discretization='UD'):
        super(SoftAttnDepth, self).__init__()
        self.dim = dim

        # The wokring ones uses: 0.01 and 1.0
        self.alpha = alpha
        self.beta = beta
        self.discretization = discretization

    def get_depth_sid(self, depth_labels):
        alpha_ = torch.FloatTensor([self.alpha])
        beta_ = torch.FloatTensor([self.beta])
        t = []
        for K in range(depth_labels):
            K_ = torch.FloatTensor([K])
            t.append(torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * K_ / depth_labels))
        t = torch.FloatTensor(t)
        return t

    def forward(self, input_t, eps=1e-6):
        batch_size, depth, height, width = input_t.shape
        if self.discretization == 'SID':
            grid = self.get_depth_sid(depth).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            grid = torch.linspace(
                self.alpha, self.beta, depth,
                requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        grid = grid.repeat(batch_size, 1, height, width).float()

        z = F.softmax(input_t, dim=self.dim)
        z = z * (grid.to(z.device))
        z = torch.sum(z, dim=1, keepdim=True)

        return z


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class UpBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 use_skip=True,
                 skip_planes=None,
                 is_output_scale=False,
                 output_scale_planes=128):
        super(UpBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.use_skip = use_skip
        self.skip_planes = skip_planes
        self.is_output_scale = is_output_scale
        self.output_scale_planes = output_scale_planes

        self.conv1 = ConvBlock(in_planes, out_planes)

        self.block_two_feature = (out_planes +
                                  skip_planes) if use_skip else out_planes

        self.conv2 = ConvBlock(self.block_two_feature, out_planes)

        self.is_output_scale = is_output_scale
        if self.is_output_scale:
            self.output_layer = Conv3x3(out_planes,
                                        output_scale_planes,
                                        use_norm=True)

    def forward(self, input_t, skip_feature=None):

        # import ipdb; ipdb.set_trace()
        x = self.conv1(input_t)
        x = upsample(x)

        if self.use_skip and skip_feature is not None:
            x = torch.cat([x, skip_feature], 1)

        x = self.conv2(x)
        output_scale = None
        if self.is_output_scale:
            output_scale = self.output_layer(x)

        return x, output_scale


class MSDepthDecoder(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 scales=range(4),
                 num_output_channels=128,
                 use_skips=True,
                 alpha=1e-3,
                 beta=1.0,
                 volume_output=False,
		         discretization='UD'):
        super(MSDepthDecoder, self).__init__()
        """
        To Replicate the paper, use num_output_channels=128
        """
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_enc[-1] = num_output_channels
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # Decoder

        # (24, 80) -> (48, 160)
        self.deconv1 = UpBlock(in_planes=128,
                               out_planes=64,
                               use_skip=True,
                               skip_planes=256,
                               is_output_scale=True,
                               output_scale_planes=self.num_output_channels)

        # (48, 160) -> (96, 320)
        self.deconv2 = UpBlock(in_planes=64,
                               out_planes=64,
                               use_skip=True,
                               skip_planes=128,
                               is_output_scale=True,
                               output_scale_planes=self.num_output_channels)

        # (96, 320) -> (192, 640)
        self.deconv3 = UpBlock(in_planes=64,
                               out_planes=32,
                               use_skip=False,
                               skip_planes=128,
                               is_output_scale=True,
                               output_scale_planes=self.num_output_channels)

        self.sigmoid = nn.Sigmoid()
        self.depth_layer = SoftAttnDepth(alpha=alpha, beta=beta, discretization=discretization)

        self.volume_output = volume_output

    def forward(self, features):
        self.outputs = {}
        # decoder
        x = features["output"]
        self.outputs[("disp", 3)] = self.depth_layer(x)

        x, z = self.deconv1(x, features["layer1"])
        self.outputs[("disp", 2)] = self.depth_layer(z)

        x, z = self.deconv2(x, features["conv3"])
        self.outputs[("disp", 1)] = self.depth_layer(z)

        x, z = self.deconv3(x, None)

        if self.volume_output:
            self.outputs[("volume", 0)] = z

        self.outputs[("disp", 0)] = self.depth_layer(z)

        return self.outputs
