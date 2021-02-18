"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from mixconv import MDConv, GroupConv2D
from . import mixconv
#import pdb;pdb.set_trace()

__all__ = ['mobilenetv2']

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

'''
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * sigmoid(x)

    @staticmethod
    def backward(ctx, dL_dy):
        #grad_input = grad_output.clone()
        x, = ctx.saved_tensors
        dy_dx = grad_swish(x)
        dL_dx = dL_dy * dy_dx
        return dL_dx

def sigmoid(x):
  return 1.0 / (1.0 + torch.exp(-x))

def swish(x):
    return x * sigmoid(x)

def grad_swish(x):
    return swish(x) + (sigmoid(x) * (1.0 - swish(x)))
'''

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual_mixconv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_mixconv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                #nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                mixconv.MDConv(hidden_dim, n_chunks=3, stride=stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        #self.mdconv = nn.Sequential(
        #    MDConv(in_channels * expand_ratio, n_chunks=3, stride=stride),
        #    nn.BatchNorm2d(in_channels * expand_ratio),
        #    nn.ReLU()
        #)


        self.cfgs_lowfeature = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
        ]

        self.cfgs_highfeature = [
            # t, c, n, s
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.cfgs_unet_low = [
            # t, c, n, s
            [6,  120, 2, 2],
            [6,  120, 2, 2],
            [6,  120, 2, 1],
        ]

        self.cfgs_unet_high = [
            # t, c, n, s
            [6,  120, 2, 1],
            [6,  120, 2, 1],
            [6,  120, 2, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        #import pdb;pdb.set_trace()
        layers_lowfeature = [conv_3x3_bn(3, input_channel, 2)]
        layers_highfeature = []

        layers_unet_low = []
        layers_unet_high = []
        # building inverted residual blocks
        block = InvertedResidual
        block_unet = InvertedResidual_mixconv

        '''
        for t, c, n, s in self.cfgs_lowfeature:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers_lowfeature.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel

        for t, c, n, s in self.cfgs_highfeature:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers_highfeature.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        '''

        for t, c, n, s in self.cfgs_lowfeature:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers_lowfeature.append(block_unet(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel

        for t, c, n, s in self.cfgs_highfeature:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers_highfeature.append(block_unet(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel

        input_channel = 32 ##上のcfgs_highfeatureのinputを引き継いでいるためinit
        for t, c, n, s in self.cfgs_unet_low:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                #print(s)
                layers_unet_low.append(block_unet(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel

        input_channel = 320 ##上のcfgs_highfeatureのinputを引き継いでいるためinit
        for t, c, n, s in self.cfgs_unet_high:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers_unet_high.append(block_unet(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel

        #for t, c, n, s in self.cfgs:
        #    output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
        #    for i in range(n):
        #        layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
        #        input_channel = output_channel
        #self.features = nn.Sequential(*layers)
        self.lowfeatures = nn.Sequential(*layers_lowfeature)
        self.highfeatures = nn.Sequential(*layers_highfeature)
        self.unet_low_features = nn.Sequential(*layers_unet_low)
        self.unet_high_features = nn.Sequential(*layers_unet_high)
        #import pdb;pdb.set_trace()
        # building last several layers
        #output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        #self.conv = conv_1x1_bn(input_channel, output_channel)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(output_channel, num_classes)

        self.deconv1 = nn.ConvTranspose2d(240, 120, 5, stride=1)
        self.deconv2 = nn.ConvTranspose2d(120, 60, 5, stride=1)
        self.deconv3 = nn.ConvTranspose2d(60, 1, 5, stride=1) #output: 64x84

        self._initialize_weights()

    def forward(self, x):
        x = self.lowfeatures(x)
        x_low = self.unet_low_features(x)

        x = self.highfeatures(x)
        x_high = self.unet_high_features(x)
        x_cat = torch.cat([x_low, x_high], dim=1)
        #import pdb;pdb.set_trace()
        #x = self.conv(x)
        #swish = Swish.apply
        #x = swish(self.deconv1(x))
        #x = swish(self.deconv2(x))
        #x = F.sigmoid(self.deconv3(x))
        #hard_sigmoid = MyHardSigmoid.apply
        #x = hard_sigmoid(self.deconv3(x))
        #import pdb;pdb.set_trace()
        x_cat = F.relu(self.deconv1(x_cat))
        x_cat = F.relu(self.deconv2(x_cat))
        x_cat = torch.sigmoid(self.deconv3(x_cat))
        #import pdb;pdb.set_trace()
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x_cat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)
