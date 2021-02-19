"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2

特徴マップに結合するコマンド情報をDeconv.を用いることでより高次な特徴にする（チャンネル数も増やす）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import vgg


class ConvModelCMD_v2(nn.Module):
    def __init__(self):
        super(ConvModelCMD_v2, self).__init__() # input: 3x 400x176
        self.encoder = vgg.vgg16_bn(pretrained=True) # output: 12x5
        self.deconv1 = nn.ConvTranspose2d(528, 240, 5, stride=1)
        self.deconv2 = nn.ConvTranspose2d(240, 120, 5, stride=1)
        self.deconv3 = nn.ConvTranspose2d(120, 60, 5, stride=1)
        self.deconv4 = nn.ConvTranspose2d(60, 1, 5, stride=1) #output: 64x84 #28x21

        self.cmd_deconvs = nn.Sequential(
                nn.ConvTranspose2d(1, 4, 3, stride=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(4, 8, (3,5), stride=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 16, (1,6), stride=1),
                nn.ReLU(True)
            )
        # self.cmd_deconv1 = nn.ConvTranspose2d(1, 4, 3, stride=1)
        # self.cmd_deconv2 = nn.ConvTranspose2d(4, 8, (5,3), stride=1)
        # self.cmd_deconv3 = nn.ConvTranspose2d(4, 8, (6,1), stride=1)
    def forward(self, img, cmd):
        x = self.encoder(img)

        b, c, h, w = x.size()
        cmd = cmd.view(b,1,1,1)
        cmd = self.cmd_deconvs(cmd)
        # print(x.size(), cmd.size())

        # cmd1 = F.interpolate(cmd, (h,w), mode='nearest')
        x = torch.cat([x, cmd], dim=1)
        x = F.relu(self.deconv1(x))

        # b, c, h, w = x.size()
        # cmd2 = F.interpolate(cmd, (h,w), mode='nearest')
        # x = torch.cat([x, cmd2], dim=1)
        x = F.relu(self.deconv2(x))

        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        # x = self.deconv4(x)
        return x

    ## ブランチないけどつける
    def forward_branch(self, x, branch_number):
        return self.forward(x, branch_number)

    # @property
    # def insize(self):
    #     return self._insize
    # @property
    # def outsize(self):
    #     return self._outsize
    
    