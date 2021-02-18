# -*- coding: utf-8 -*-
'''
PyTorch版SalNet 
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

#from models.net_inceptionresnet import MobileNetV3
from models.net_mobilenetv2_multiscalev2 import MobileNetV2
### 追加：簡単な自作モデル
from models.net_mobilenetv2_multiscalev2_cmd import MobileNetV2CMD
from models.conv_model import ConvModel
from models.conv_model_cmd import ConvModelCMD
#from dataloader import MyDataset

import scipy.io as sio
from scipy.stats import entropy
# from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


import importlib

import re
from pathlib import Path

from dataloader import CarlaGazeDataset

import matplotlib.pyplot as plt


from dataloader import CarlaGazeVideoDataset
from dataloader import CarlaGazeDataset

from loss import Loss
from models.gaze_models import get_model

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
    parser.add_argument('--gpu-id', default='3', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--test-epoch', type=str, default='50')


    parser.add_argument('--batch', '-b', type=int, default=1,
                        help='batch size(default: 1)')

    parser.add_argument('-vd', '--valid-dataset', type=str, default="./dataset",
                    help='root path of validation dataset')
    parser.add_argument('-m', '--model', type=str, default=None,
                    choices=['co-conv', 'conv-cmd', 'conv-cmd-v2'],
                    help='Model name')
    parser.add_argument('-dt', '--dataset-type', type=str, default='video',
                choices=['v1', 'video'],
                help='Dataset type. v1: 1-frame gaze, video: gaze for video')
    parser.add_argument('-ws', '--work-space', type=str, default='mkei',
                help='Workspace')

    args = parser.parse_args()
    return args

def softmax(targets): #softmaxにより確率分布を取得
    # targets_max = np.max(targets)
    # expp = np.exp(targets-targets_max)
    expp = np.exp(targets)
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
    # #outputs = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))
    # outputs = outputs / np.sum(outputs)
    # #targets = (targets - np.min(targets)) / (np.max(targets) - np.min(targets))
    # targets = targets / np.sum(targets)
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

    exp_base = './experiments'
    exp_folder = os.path.join(exp_base, args.exp_name)
    if not os.path.exists(exp_folder):
        raise RuntimeError("Model directory does not exist")
    save_path = os.path.join(exp_folder, 'checkpoints')
    output_folder = os.path.join(exp_folder, 'outputs')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    EXP_NAME = args.exp_name
    print("exp_name: %s" % EXP_NAME)

    cudnn.benchmark = True


    ## 変更
    net = get_model(args.model)


    print("Loading model's parameters")
    ##### モデル呼び出し
    model_path = os.path.join(save_path, '%s_%s.pth' % (EXP_NAME, args.test_epoch))
    net.load_state_dict(torch.load(model_path))
    net.to(device)


    test_criterion = nn.BCELoss().to(device=device) # 脳死
    
    img_size = (400, 176) # 後で変更
    # gaze_size_mn = (25, 18)
    gaze_size = (28, 21) # 後で変更
    if args.dataset_type == 'v1':
        test_dataset = CarlaGazeDataset(args.valid_dataset, img_size[0], img_size[1], gaze_size[0], gaze_size[1])
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    elif args.dataset_type == 'video':
        test_dataset = CarlaGazeVideoDataset(args.valid_dataset, img_size[0], img_size[1], 
                                                gaze_size[0], gaze_size[1], args.work_space)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    else:
        raise ValueError("Invalid dataset type : %s" % args.dataset_type)


    print("Start Evaluation")

    net.eval()
    #############
    sim = 0.
    corr = 0.
    nss = 0.
    auc = 0.
    kl = 0.
    iteration = len(test_loader)
    test_loss = 0
    sum_iteration_val = 0
    #for i in range(int(iteration/2)):
    with torch.no_grad():
        for i, (inputs_images, salmaps_true, cmd) in enumerate(test_loader):

            ##########データをGPUへ転送############
            inputs_images = torch.Tensor(inputs_images).to(device=device)
            salmaps_true = torch.Tensor(salmaps_true).to(device=device)
            inputs_images = torch.autograd.Variable(inputs_images) #多分いる？
            salmaps_true = torch.autograd.Variable(salmaps_true) #多分いる？
            cmd = torch.tensor(cmd, dtype=torch.float32).to(device=device)
            #####################################
            
            # forward + backward + optimize######
            outputs = net.forward_branch(inputs_images, cmd)
            #####################################

            ####Saliencyの可視化############################
            inputs_ii = inputs_images.cpu().detach().numpy()*255
            inputs_ii = inputs_ii.transpose(0,2,3,1)
            inputs_ii = inputs_ii.astype(np.uint8)####cv2のRGB画像を用意
            out_sals = outputs.cpu().detach().numpy()
            salmaps_true_np = salmaps_true.cpu().detach().numpy()
            
            ###############################################

            #####定量的評価######
            # for sal in range(args.batch):
            for sal in range(out_sals.shape[0]):
                out_sal = cv2.resize(out_sals[sal][0], (200, 88))
                #out_sal = (out_sal-np.min(out_sal)) / (np.max(out_sal)-np.min(out_sal))#0~1へ正規化
                out_sal = out_sal / np.max(out_sal)
                out_sal_prob = softmax(out_sal)
                #out_sal = softmax(out_sal)
                ##
                salmap_true = salmaps_true_np[sal][0]#softmaxかけないバージョン
                salmap_true = cv2.resize(salmap_true, (200, 88))
                salmap_true_prob = softmax(salmap_true)

                sim += similarity(out_sal, salmap_true)
                corr += corrcoef(out_sal, salmap_true)
                # nss += nss_x(out_sal, fixation)
                # auc += roc_auc_score(fixation.reshape(-1), out_sal.reshape(-1))
                kl += kldiv(out_sal_prob, salmap_true_prob)

                ### 可視化
                image = cv2.resize(inputs_ii[sal], (200, 88))

                out_sals_img = out_sal * 255
                out_sals_img = out_sals_img.astype(np.uint8)
                out_sals_img = cv2.cvtColor(out_sals_img, cv2.COLOR_GRAY2BGR)
                result_img = cv2.applyColorMap(out_sals_img, cv2.COLORMAP_JET)
                result_img = cv2.addWeighted(image, 0.7, result_img, 0.3, 0)

                name = "output_%08d.png" % i
                cv2.imwrite(os.path.join(output_folder, name), result_img)

            print('iteration=' + str(i))
            print('similarity=' + str(sim/(args.batch*(i+1))))
            print('corrcoef=' + str(corr/(args.batch*(i+1))))
            # print('nss=' + str(nss/(args.batch*(i+1))))
            # print('auc=' + str(auc/(args.batch*(i+1))))
            print('kl=' + str(kl/(args.batch*(i+1))))
            ####################

            #salmaps_true = salmaps_true.long()
            test_loss += test_criterion(outputs, salmaps_true).item()

            ###各変数の初期化，加算########
            inputs_images = inputs_images.cpu()
            salmaps_true = salmaps_true.cpu()
            inputs_images = torch.Tensor(np.zeros((0, 3, 480, 640)))
            salmaps_true = torch.Tensor(np.zeros((0, 1, 480, 640)))
            sum_iteration_val += 1
            #######################

    # 変更
    print('====== EVAL RESULT ======')
    print('similarity=' + str(sim/(args.batch*(iteration))))
    print('corrcoef=' + str(corr/(args.batch*(iteration))))
    print('kl=' + str(kl/(args.batch*(iteration))))


    with open(os.path.join(exp_folder, 'eval_result_log.txt'), 'w') as f:
        f.write('Test model : %s\n' % model_path)
        f.write('Similarity : %s\n' % str(sim/(args.batch*(iteration))))
        f.write('Corrcoef   : %s\n' % str(corr/(args.batch*(iteration))))
        f.write('KL-Div     : %s\n' % str(kl/(args.batch*(iteration))))
    

    print('Finished Testing')


if __name__ == '__main__':
    start_time = time.time()
    main()
    #print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))