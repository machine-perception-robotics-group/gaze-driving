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
from . import vgg


class ConvModelCMD(nn.Module):
    def __init__(self):
        super(ConvModelCMD, self).__init__()
        self.encoder = vgg.vgg16_bn(pretrained=True)
        self.deconv1 = nn.ConvTranspose2d(513, 240, 5, stride=1)
        self.deconv2 = nn.ConvTranspose2d(240, 120, 5, stride=1)
        self.deconv3 = nn.ConvTranspose2d(120, 60, 5, stride=1)
        self.deconv4 = nn.ConvTranspose2d(60, 1, 5, stride=1) #output: 64x84
    def forward(self, img, cmd):
        # x, self.inter = self.encoder(img) #ミス発覚，mimickは使うつもりないのでinterは削除
        x = self.encoder(img)

        b, c, h, w = x.size()
        cmd = cmd.view(b,1,1,1)
        cmd1 = F.interpolate(cmd, (h,w), mode='nearest')
        x = torch.cat([x, cmd1], dim=1)
        x = F.relu(self.deconv1(x))

        # b, c, h, w = x.size()
        # cmd2 = F.interpolate(cmd, (h,w), mode='nearest')
        # x = torch.cat([x, cmd2], dim=1)
        x = F.relu(self.deconv2(x))

        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        # x = self.deconv4(x)
        return x