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


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.encoder = vgg.vgg16_bn(pretrained=False) # 学習はしないためpretrained=False
        self.deconv1 = nn.ConvTranspose2d(512, 240, 5, stride=1)
        self.deconv2 = nn.ConvTranspose2d(240, 120, 5, stride=1)
        self.deconv3 = nn.ConvTranspose2d(120, 60, 5, stride=1)
        self.deconv4 = nn.ConvTranspose2d(60, 1, 5, stride=1) #output: 64x84
    def forward(self, img, cmd): # cmd is not used
        x = self.encoder(img)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x