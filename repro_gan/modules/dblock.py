import torch.nn as nn
import torch.nn.functional as F

class DBlock(nn.Module):
    """
    Block for discriminator.

    Args:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        kernel_size (int): The size of the convolutional kernel.
        stride (int, optional): The stride of the convolution. Default is 1.
        padding (int, optional): The padding added to the input. Default is 0.
        downsample (bool, optional): If True, downsamples the input feature map. Default is False.
        use_reflection_pad (bool, optional): If True, uses reflection padding. Default is False.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 downsample=False,
                 use_reflection_pad=False):
        super().__init__()
        self.use_reflection_pad = use_reflection_pad
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = downsample

        # reflection padding
        self.conv_no_pad = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.reflection_padding = nn.ReflectionPad1d((padding,padding))
        nn.init.normal_(self.conv_no_pad.weight.data, 0.0, 0.02)

        # zero padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)

    def forward(self, x):
        h = x
        if self.use_reflection_pad:
            h = self.reflection_padding(h)
            h = self.conv_no_pad(h)
        else:
            h = self.conv(x)
        h = self.activation(h)
        if self.downsample:
            h = F.avg_pool1d(h, 2)
        return h
