# -*- coding: utf-8 -*-
'''
 coiltraineのDatasets/CoILValを評価に使用するためのプログラム
'''
import argparse
import time
import numpy as np
import os
import cv2
import sys
import glob
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import tensorboardX as tbx

#from models.net_inceptionresnet import MobileNetV3
from models.net_mobilenetv2_multiscalev2 import MobileNetV2
### 追加：簡単な自作モデル
from models.net_mobilenetv2_multiscalev2_cmd import MobileNetV2CMD
from models.conv_model import ConvModel
from models.conv_model_cmd import ConvModelCMD
from models.conv_model_cmd_v2 import ConvModelCMD_v2
#from dataloader import MyDataset

import scipy.io as sio
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import re
from pathlib import Path

from dataloader_eval import CarlaGazeDataset

import matplotlib.pyplot as plt

# ###可視化の都合上、警告を無視（デバッグの際は外そう）
# import warnings
# warnings.simplefilter('ignore', category=RuntimeWarning) 
# #####################

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--folder', default='samples', type=str,
                        help='data folder')
    parser.add_argument('--save-folder', default='results', type=str,
                        help='data folder')
    parser.add_argument('--gpu-id', default='3', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--my_eval_flag', action='store_true',
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    return args

def softmax(targets): #softmaxにより確率分布を取得
    targets_max = np.max(targets)
    expp = np.exp(targets-targets_max)
    total = np.sum(expp)
    return expp/total

def similarity(outputs, targets): #定量的評価指標1: SIM
    sim = 0.
    outputs = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))
    outputs = outputs / np.sum(outputs)
    targets = (targets - np.min(targets)) / (np.max(targets) - np.min(targets))
    targets = targets / np.sum(targets)
    sim = np.sum(np.minimum(outputs, targets))
    return sim

def corrcoef(outputs, targets): #定量的評価指標2: CC
    S_x = np.std(outputs)
    mean = np.average(outputs)
    outputs = (outputs - mean)
    S_y = np.std(targets)
    mean = np.average(targets)
    targets = (targets - mean)
    N = len(outputs)*len(outputs[0])
    S_xy = np.sum((outputs * targets) / N)
    corr = S_xy / (S_x * S_y)
    #import pdb;pdb.set_trace()
    return corr

##NSSはfixation mapが必要であり，saliency mapではだめ
def nss_x(outputs, targets): #定量的評価指標3: NSS
    std = np.std(outputs)
    mean = np.average(outputs)
    o_bar = (outputs - mean) / std
    nss = np.sum(o_bar * targets)
    #import pdb;pdb.set_trace()
    return nss / np.sum(targets)

def kldiv(outputs, targets): #定量的評価指標4: KL-Divergence
    kl = 0.
    eps = 2.220446049250313e-16
    #outputs = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))
    outputs = outputs / np.sum(outputs)
    #targets = (targets - np.min(targets)) / (np.max(targets) - np.min(targets))
    targets = targets / np.sum(targets)
    kl = np.sum(targets * np.log(eps + (targets / (eps + outputs))))
    return kl

def KLLoss(student_output, teacher_output, T=1.0, log=None):
    student_logits = student_output
    teacher_logits = teacher_output.detach()
    student_softmax = F.softmax(student_logits/T, dim=1)
    teacher_softmax = F.softmax(teacher_logits/T, dim=1)
    loss_per_sample = kl_divergence(student_softmax, teacher_softmax, log=log)        
    soft_loss = loss_per_sample.mean()
    loss = soft_loss * (T**2)
    return loss
def kl_divergence(student, teacher, log=None):
    kl = teacher * torch.log((teacher / (student+1e-10)) + 1e-10)
    #import pdb;pdb.set_trace()
    kl = kl.sum(dim=1)
    #kl = kl.sum()
    #import pdb;pdb.set_trace()
    loss = kl
    return loss

def max_min_norm(gazemap):
    mx = np.max(gazemap)
    mn = np.min(gazemap)
    gazemap = (gazemap-mn) / (mx-mn)
    return gazemap


def main():
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if os.path.exists(args.save_folder):
        raise RuntimeError("directory exists")
    else:
        os.mkdir(args.save_folder)

    ## 変更
    # trainset = MyDataset()
    # train_image, train_sal, val_image, val_sal, val_salmat = trainset.trans()
    # iteration = int((len(train_image) / args.batch))
    # print("iteration=" + str(iteration))

    # modelの定義
    #import torchvision.models as models
    #mobilenet = models.mobilenet_v2(pretrained=True)
    ## MNの定義
    net_mn = MobileNetV2CMD().to(device)
    ## VGGの定義
    net_vgg = ConvModelCMD_v2().to(device)
    # net_vgg = torch.nn.DataParallel(net_vgg).to(device)
    ## net = ConvModel()
    # params_finetune = list(net.parameters())
    # print(params_finetune[0][0][0])

    cudnn.benchmark = True

    print("Loading model's parameters")
    ##### モデル呼び出し (途中から学習する場合)
    # net_mn.load_state_dict(torch.load('./checkpoints_new_mn_cmd/model_50.pth'))
    net_vgg.load_state_dict(torch.load('./checkpoints_dsv3_max_vgg_cmd_v2/model_100.pth'))

    ### 最初のVGGテスト
    # net_vgg.load_state_dict(torch.load('./past_models_checkpoints/checkpoints_vgg_b16_400_176/model_best_train_mobilenetv2_multiscalev2.pth'))

    img_size = (400, 176) # 後で変更
    gaze_size_mn = (25, 18)
    gaze_size_vgg = (28, 21) # 後で変更

    ### Dataloader
    dataset = CarlaGazeDataset(args.folder, img_size[0], img_size[1], gaze_size_vgg[0], gaze_size_vgg[1])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Start Evaluation")

    dnum = len(dataloader)

    net_mn.eval()
    net_vgg.eval()
    
    count = 0
    for i, (image, input_image, cmd) in enumerate(dataloader):
        # print(cmd)
        
        count += 1
        sys.stdout.write("\r%d / %d" % (count, dnum))
        sys.stdout.flush()

        ##########データをGPUへ転送############
        input_image = torch.Tensor(input_image).to(device=device)
        # inputs_image = torch.autograd.Variable(inputs_images) #多分いる？
        cmd = torch.tensor(cmd, dtype=torch.float32).to(device=device)
        #####################################
        
        # forward + backward + optimize######
        ## MN
        # outputs_mn = net_mn(input_image, cmd)
        ##outputs_mn = net_mn(input_image)
        ## VGG
        outputs_vgg = net_vgg(input_image, cmd)
        #####################################

        ### 元画像データからbatch軸を消す
        image = image[0].detach().numpy()

        ## 決め打ち正規化
        # outputs_mn = torch.clamp(outputs_mn / 0.4, 0., 1.)
        # outputs_vgg = torch.clamp(outputs_vgg / 0.4, 0., 1.)

        ####Saliencyの可視化############################
        # inputs_ii = inputs_images.cpu().detach().numpy()*255
        # inputs_ii = inputs_ii.transpose(0,2,3,1)
        # inputs_ii = inputs_ii.astype(np.uint8)####cv2のRGB画像を用意

        ### MN
        # out_sals_mn = outputs_mn.cpu().detach().numpy()
        # # out_sals_mn = max_min_norm(out_sals_mn)
        # out_sals_mn = cv2.resize(np.squeeze(out_sals_mn), (image.shape[1], image.shape[0])) * 255
        # #cv2.imwrite("./stock_mn.png", out_sals_mn)
        # sal_img_mn = out_sals_mn.astype(np.uint8)
        # sal_img_mn = cv2.cvtColor(sal_img_mn, cv2.COLOR_GRAY2BGR)
        # #sal_img_mn = cv2.imread("./stock_mn.png")
        # result_mn = cv2.applyColorMap(sal_img_mn, cv2.COLORMAP_JET)
        # result_mn = cv2.addWeighted(image, 0.7, result_mn, 0.3, 0)

        ### VGG
        out_sals_vgg = outputs_vgg.cpu().detach().numpy()
        # out_sals_vgg = max_min_norm(out_sals_vgg)
        out_sals_vgg = cv2.resize(np.squeeze(out_sals_vgg), (image.shape[1], image.shape[0])) * 255
        #cv2.imwrite("./stock_vgg.png", out_sals_vgg)
        sal_img_vgg = out_sals_vgg.astype(np.uint8)
        sal_img_vgg = cv2.cvtColor(sal_img_vgg, cv2.COLOR_GRAY2BGR)
        #sal_img_vgg = cv2.imread("./stock_vgg.png")
        result_vgg = cv2.applyColorMap(sal_img_vgg, cv2.COLORMAP_JET)
        result_vgg = cv2.addWeighted(image, 0.7, result_vgg, 0.3, 0)


        # name = "mn_" + str(img_path.name)
        # cv2.imwrite(str(save_path.joinpath("results"+class_label[i]) / name), result_mn)

        #name = "vgg_" + str(img_path.name)
        name = "vgg_%08d.png" % (count-1)
        cv2.imwrite(os.path.join(args.save_folder, name), result_vgg)


    print('Finished Testing')


if __name__ == '__main__':
    start_time = time.time()
    main()
    #print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))