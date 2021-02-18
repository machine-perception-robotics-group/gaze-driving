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

class Branching(nn.Module):
    def __init__(self, branched_modules=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """
        # TODO: Make an auto naming function for this.

        super(Branching, self).__init__()

        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)

    # TODO: iteration control should go inside the logger, somehow
    def forward(self, x):
        # get only the speeds from measurement labels
        # TODO: we could easily place this speed outside

        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(x))

        return branches_outputs



class CoConvModel(nn.Module):
    def __init__(self):
        super(CoConvModel, self).__init__() # input: 3x 400x176
        self.encoder = vgg.vgg16_bn(pretrained=True) # output: 12x5

        branch_decoder = []
        for i in range(4): # ハードコーディング
            branch_decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(512, 240, 5, stride=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(240, 120, 5, stride=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(120, 60, 5, stride=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(60, 1, 5, stride=1), #output: 64x84 #28x21
                        nn.Sigmoid()
                    )
                )
        self.branch_decoder = Branching(branch_decoder)

        # self.cmd_deconv1 = nn.ConvTranspose2d(1, 4, 3, stride=1)
        # self.cmd_deconv2 = nn.ConvTranspose2d(4, 8, (5,3), stride=1)
        # self.cmd_deconv3 = nn.ConvTranspose2d(4, 8, (6,1), stride=1)
    def forward(self, img, cmd):
        x = self.encoder(img)
        x = self.branch_decoder(x)
        return x

    def forward_branch(self, x, branch_number):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            branch_number: the branch number to be returned

        Returns:
            the forward operation on the selected branch
        """
        # Convert to integer just in case.
        
        output_vec = torch.stack(self.forward(x, branch_number))

        return self.extract_branch(output_vec, branch_number)


    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]


# TODO: there should be a more natural way to do that
def command_number_to_index(command_vector):

    return command_vector-2


