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
from dataloader import MyDataset

import scipy.io as sio
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import re
from pathlib import Path
###可視化の都合上、警告を無視（デバッグの際は外そう）
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning) 
#####################

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', '-l', type=float, default=0.03,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--batch', '-b', type=int, default=40,
                        help='batch size(default: 40)')
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

def main():
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    trainset = MyDataset()
    train_image, train_sal, val_image, val_sal, val_salmat = trainset.trans()
    iteration = int((len(train_image) / args.batch))
    print("iteration=" + str(iteration))

    # modelの定義
    #import torchvision.models as models
    #mobilenet = models.mobilenet_v2(pretrained=True)
    net = MobileNetV2()
    #net.load_state_dict(torch.load('./mobilenetv2-c5e733a8.pth'), strict=False) #pretrainモデル読み込み
    params_finetune = list(net.parameters())
    print(params_finetune[0][0][0])
    #import pdb;pdb.set_trace()

    #params_pretrain = list(mobilenet.parameters())
    #params_finetune = list(net.parameters())

    net = torch.nn.DataParallel(net).to(device=device)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    # define loss function and optimier
    criterion_mse = nn.MSELoss().to(device=device)
    criterion = nn.BCELoss().to(device=device)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr, momentum=0.99, nesterov=True, weight_decay=1e-4)
    
    #pretrainの重みをぶち込む
    #for p_count in range(156):
    #    optimizer.param_groups[0]['params'][p_count] = nn.Parameter(params_pretrain[p_count])
    #import pdb;pdb.set_trace()
    #print(optimizer.param_groups[0]['params'][0][0][0])
    #print(params_net[0][0][0])
    #import pdb;pdb.set_trace()

    writer = tbx.SummaryWriter(log_dir="./logs/exp01")

    #モデル呼び出し (途中から学習する場合)
    #net.load_state_dict(torch.load('./checkpoints/model_best.pth'))

    # train
    dir_path1_train = './salicon_train_saliency_time/'
    sal_train_paths = [str(p) for p in sorted(Path(dir_path1_train).glob("*.png"))]
    sal_train_paths = sorted(sal_train_paths, key = numericalSort)
    #import pdb;pdb.set_trace()

    inputs_images = torch.Tensor(np.zeros((0, 3, 480, 640)))
    salmaps_true = torch.Tensor(np.zeros((0, 1, 27, 32)))
    sum_iteration = 0
    sum_iteration_val = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        net.train()
        #import pdb;pdb.set_trace()
        for i in range(iteration):
            # zero the parameter gradients
            break
            optimizer.zero_grad()
            #######データの前処理########
            #import pdb;pdb.set_trace()
            for j in range(args.batch): #バッチサイズごとに画像を読み出す処理、前処理も行なっている
                input_image = cv2.imread(train_image[i*args.batch+j]) / 255
                salmap = cv2.imread(train_sal[i*args.batch+j], cv2.IMREAD_GRAYSCALE)
                #salmap = cv2.imread(sal_train_paths[i*args.batch+j], cv2.IMREAD_GRAYSCALE) ####データセット変えてるからね!!!!!!!!!!!!!!!!!!!!!!!!!!#######
                salmap = cv2.resize(salmap, (32, 27)) / 255 #resizeの引数は(width, height)と逆になっていることに注意されたし
                #print(sal_train_paths[i*args.batch+j])
                ###デバッグ用コード
                '''
                input_test = cv2.imread(train_image[i*args.batch+j])
                sal_test = cv2.imread(sal_train_paths[i*args.batch+j], cv2.IMREAD_GRAYSCALE)
                jet_map = cv2.applyColorMap(sal_test, cv2.COLORMAP_JET)
                jet_map = cv2.addWeighted(input_test, 0.5, jet_map, 0.5, 0)
                cv2.imwrite('./test_jetmap.png', jet_map)
                import pdb;pdb.set_trace()
                '''
                
                input_image = np.reshape(input_image, (1, len(input_image), len(input_image[0]), len(input_image[0][0])))
                input_image = np.transpose(input_image, (0, 3, 1, 2)) #reshapeじゃなくてtransposeにしないとぐちゃぐちゃになる
                inputs_images = np.append(inputs_images, input_image, axis=0)#(batch, channel, height, width)

                salmap = np.reshape(salmap, (1, 1, len(salmap), len(salmap[0])))
                salmaps_true = np.append(salmaps_true, salmap, axis=0) #(batch, channel, height, width)
            ###########################

            ##########データをGPUへ転送############
            inputs_images = torch.Tensor(inputs_images).to(device=device)
            salmaps_true = torch.Tensor(salmaps_true).to(device=device)
            #inputs_images = torch.autograd.Variable(inputs_images) #多分いる？
            #salmaps_true = torch.autograd.Variable(salmaps_true) #多分いる？
            #import pdb;pdb.set_trace()
            #####################################
     
            # forward + backward + optimize
            outputs = net(inputs_images)
            print('max=' + str(torch.max(outputs)))#もしsaliency mapの最大値が0になるケースが増えたら、局所解に落ちた可能性が高い
            print('min=' + str(torch.min(outputs)))#
            
            #print(optimizer.param_groups[0]['params'][0][0][0])
            #import pdb;pdb.set_trace()

            #salmaps_true = salmaps_true.long()
            loss = criterion(outputs, salmaps_true)
            #loss_mse = criterion_mse(outputs, salmaps_true)
            #loss_kl = KLLoss(outputs, salmaps_true)
            #loss = loss_bce + loss_kl + loss_mse
            #loss = criterion(outputs.view(len(outputs), -1), salmaps_true.view(len(salmaps_true), -1))
            loss.backward()
            optimizer.step()
            #print(optimizer.param_groups[0]['params'][0][0][0])
            #import pdb;pdb.set_trace()
            
            inputs_images = inputs_images.cpu()
            salmaps_true = salmaps_true.cpu()
            inputs_images = torch.Tensor(np.zeros((0, 3, 480, 640)))
            salmaps_true = torch.Tensor(np.zeros((0, 1, 27, 32)))
            sum_iteration += 1

            # print statistics
            running_loss += loss.item()
            #print("running_loss=" + str(running_loss))

            #スケジューラ, learning rateの変更
            if epoch == 15:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
                #print(optimizer.param_groups[0]['lr'] * 0.1)
            elif epoch == 30:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            elif epoch == 50:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            #######
            
            if i == (iteration-1):
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch+1, i+1, running_loss/iteration))
                running_loss = 0.0
            else:
                print('iteration=' + str(i))
                print('loss=' + str(loss.item()))
            ###tensorboard###
            writer.add_scalars('train/loss',
                {
                 'sal_bce': loss.item()
                },sum_iteration)

            ###tensorboard###
            writer.add_scalars('train/accuracy',
                {
                 'sal_bce': loss.item()
                },sum_iteration)
            ##################
            #sys.stdout.write('\r max={}'.format(torch.max(outputs)))
            #sys.stdout.flush()

        
        #torch.save(net.state_dict(), './checkpoints/model_best_mobilenetv2_multiscalev2.pth')
        #net.load_state_dict(torch.load('./checkpoints/model_best_anosal_time.pth'))#学習済みモデルの読み込み
        net.load_state_dict(torch.load('./checkpoints/model_best_mobilenetv2_multiscalev2.pth'))#学習済みモデルの読み込み
        if args.my_eval_flag:
            dir_path1_myval = './saliency_my_photo/'
            val_image = [str(p) for p in sorted(Path(dir_path1_myval).glob("*.png"))]
            val_image = sorted(val_image, key = numericalSort)
            import pdb;pdb.set_trace()

        #import pdb;pdb.set_trace()
        #####evaltion######
        print("evalution mode")
        net.eval()
        sim = 0.
        corr = 0.
        nss = 0.
        auc = 0.
        kl = 0.
        salmaps_true = torch.Tensor(np.zeros((0, 1, 480, 640)))
        for i in range(int(iteration/2)):
            #######データの前処理########
            #break
            for j in range(args.batch): #バッチサイズごとに画像を読み出す処理、前処理も行なっている
                input_image = cv2.imread(val_image[i*args.batch+j]) / 255
                #input_image = cv2.resize(input_image, (640, 480)) / 255
                salmap = cv2.imread(val_sal[i*args.batch+j], cv2.IMREAD_GRAYSCALE)
                #salmap = cv2.resize(salmap, (89, 69)) / 255 #resizeの引数は(width, height)と逆になっていることに注意されたし
                salmap = salmap / 255 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
                input_image = np.reshape(input_image, (1, len(input_image), len(input_image[0]), len(input_image[0][0])))
                input_image = np.transpose(input_image, (0, 3, 1, 2)) #reshapeじゃなくてtransposeしないとぐちゃぐちゃになる
                inputs_images = np.append(inputs_images, input_image, axis=0)#(batch, channel, height, width)

                salmap = np.reshape(salmap, (1, 1, len(salmap), len(salmap[0])))
                salmaps_true = np.append(salmaps_true, salmap, axis=0) #(batch, channel, height, width)
            ###########################

            ##########データをGPUへ転送############
            inputs_images = torch.Tensor(inputs_images).to(device=device)
            salmaps_true = torch.Tensor(salmaps_true).to(device=device)
            inputs_images = torch.autograd.Variable(inputs_images) #多分いる？
            salmaps_true = torch.autograd.Variable(salmaps_true) #多分いる？
            #####################################
            
            # forward + backward + optimize######
            outputs = net(inputs_images)
            #####################################

            ####Saliencyの可視化############################
            inputs_ii = inputs_images.cpu().detach().numpy()*255
            inputs_ii = inputs_ii.transpose(0,2,3,1)
            inputs_ii = inputs_ii.astype(np.uint8)####cv2のRGB画像を用意
            out_sals = outputs.cpu().detach().numpy()
            salmaps_true_np = salmaps_true.cpu().detach().numpy()
            
            '''
            for sal in range(args.batch):
                #saliency mapの真値と予測値をsoftmaxにより確率化
                out_sal = softmax(out_sals[sal][0])#softmax
                out_sal = (out_sal-np.min(out_sal))/(np.max(out_sal)-np.min(out_sal))*255 #正規化
                out_sal = cv2.resize(out_sal, (640, 480))
                out_sal = out_sal.astype(np.uint8)
                salmap_true_np = softmax(salmaps_true_np[sal][0])#softmax
                salmap_true_np = (salmap_true_np-np.min(salmap_true_np))/(np.max(salmap_true_np)-np.min(salmap_true_np))*255 #正規化
                salmap_true_np = cv2.resize(salmap_true_np, (640, 480))
                salmap_true_np = salmap_true_np.astype(np.uint8)
                #import pdb;pdb.set_trace()
                jet_map1 = cv2.applyColorMap(out_sal, cv2.COLORMAP_JET)
                jet_map1 = cv2.addWeighted(inputs_ii[sal], 0.5, jet_map1, 0.5, 0)

                jet_map2 = cv2.applyColorMap(salmap_true_np, cv2.COLORMAP_JET)
                jet_map2 = cv2.addWeighted(inputs_ii[sal], 0.5, jet_map2, 0.5, 0)
                jet_concat = np.concatenate([jet_map2, jet_map1], axis=0)
                #cv2.imwrite('./outputs_val/out_sal_val' + str(i*args.batch+sal) +'.png', jet_map1)
                cv2.imwrite('./saliency_my_photo/out_sal_cat' + str(i*args.batch+sal) +'.png', jet_concat)
            
            #import pdb;pdb.set_trace()
            '''
            
            
            ###############################################

            #####定量的評価######
            for sal in range(args.batch):
                out_sal = cv2.resize(out_sals[sal][0], (640, 480))
                #out_sal = (out_sal-np.min(out_sal)) / (np.max(out_sal)-np.min(out_sal))#0~1へ正規化
                #out_sal = softmax(out_sal)
                salmap_true = salmaps_true_np[sal][0]#softmaxかけないバージョン
                #salmap_true = softmax(salmaps_true_np[sal][0])
                mat = sio.loadmat(val_salmat[i*args.batch+sal])
                fixation = np.zeros(mat['resolution'][0])
                for k in range(len(mat['gaze'])):
                    for l in range(len(mat['gaze'][k][0][2])):
                        fixation[mat['gaze'][k][0][2][l][1]-1, mat['gaze'][k][0][2][l][0]-1] = 1 #(x, y)で格納されているため1, 0とする
                        #xxx = int(mat['gaze'][k][0][0][l][1]-1)
                        #yyy = int(mat['gaze'][k][0][0][l][0]-1)
                        #fixation[xxx, yyy] = 255
                #fixation = cv2.resize(fixation, (89, 69), interpolation = cv2.INTER_AREA)
                fixation[fixation > 0] = 1
                fixation = fixation.astype(np.uint8)
                #cv2.imwrite('./fixation_test_crowd.png', fixation)
                #import pdb;pdb.set_trace()

                sim += similarity(out_sal, salmap_true)
                corr += corrcoef(out_sal, salmap_true)
                nss += nss_x(out_sal, fixation)
                auc += roc_auc_score(fixation.reshape(-1), out_sal.reshape(-1))
                kl += kldiv(out_sal, salmap_true)
            print('iteration=' + str(i))
            print('similarity=' + str(sim/(args.batch*(i+1))))
            print('corrcoef=' + str(corr/(args.batch*(i+1))))
            print('nss=' + str(nss/(args.batch*(i+1))))
            print('auc=' + str(auc/(args.batch*(i+1))))
            print('kl=' + str(kl/(args.batch*(i+1))))
            #import pdb;pdb.set_trace()
            ####################

            #salmaps_true = salmaps_true.long()
            #loss = criterion(outputs, salmaps_true)

            ###各変数の初期化，加算########
            inputs_images = inputs_images.cpu()
            salmaps_true = salmaps_true.cpu()
            inputs_images = torch.Tensor(np.zeros((0, 3, 480, 640)))
            salmaps_true = torch.Tensor(np.zeros((0, 1, 480, 640)))
            sum_iteration_val += 1
            #######################

            #####torchvisionによるplot########
            '''
            if i == (iteration-1):
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch+1, i+1, running_loss/iteration))
                running_loss = 0.0
            else:
                print('iteration=' + str(i))
                print('loss=' + str(loss.item()))

            writer.add_scalars('val/loss',
                {
                 'sal_bce': loss.item()
                },sum_iteration_val)
            '''
            #############################
        
        break
        print('similarity=' + str(sim/(args.batch*(iteration/2))))
        print('corrcoef=' + str(corr/(args.batch*(iteration/2))))
        #import pdb;pdb.set_trace()


    
    print('Finished Training')

    # test
    correct = 0
    total = 0


if __name__ == '__main__':
    start_time = time.time()
    main()
    #print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))