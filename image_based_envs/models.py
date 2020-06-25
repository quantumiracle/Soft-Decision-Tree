# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__=['ConvPool',
        ]

class ConvPool(nn.Module):
    """ 1 convolution + 1 max pooling """
    def __init__(self, input_nc, input_width, input_height, 
                 output_nc=6, kernel_size=5, downsample=True, **kwargs):
        super(ConvPool, self).__init__()
        self.downsample = downsample

        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)
            self.downsample = False

        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size)
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the transformer to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        if self.downsample:
            return F.max_pool2d(out, 2)
        else:
            return out