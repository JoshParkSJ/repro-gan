"""
Implementation of convolutional blocks for generator.
"""
import torch.nn as nn
import torch.nn.functional as F

class GBlock(nn.Module):
    r"""
    Convolutional block for generator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        upsample (bool): If True, upsamples the input feature map.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 activation=True,
                 batchNorm=True,
                 upsample=False,
                 reflectionPad=False,
                 ):
        super().__init__()
        self.padding = padding
        self.activation = activation
        self.batchNorm = batchNorm
        self.upsample = upsample
        self.reflectionPad = reflectionPad

        if (reflectionPad):
            self.convLayer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.convLayer = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
            self.paddingLayer = nn.ReflectionPad1d((padding ,padding))

        self.batchNormLayer = nn.BatchNorm1d(out_channels)
        self.upsampleLayer = nn.Upsample(scale_factor=2)
        self.activationLayer = nn.LeakyReLU(0.2)

        nn.init.normal_(self.convLayer.weight.data, 0.0, 0.02)

    def forward(self, x):
        y = self.convLayer(x)

        if (self.upsample):
            y = self.upsampleLayer(y)

        if (self.reflectionPad):
            y = self.paddingLayer(y)

        if (self.batchNorm):
            y = self.batchNormLayer(y)

        if (self.activation):
            y = self.activationLayer(y)

        return y