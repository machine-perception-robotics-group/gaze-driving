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
from PIL import Image

import matplotlib.pylab as plt

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils
from torch.utils.data import DataLoader

import time
import traceback
import scipy.ndimage as ndi
from scipy import interpolate

import re
from pathlib import Path
from os import path
import csv
###可視化の都合上、警告を無視（デバッグの際は外そう）
# import warnings
# warnings.simplefilter('ignore', category=RuntimeWarning) 
#####################

CVT_DICT = {
    "reach_goal": 0,
    "lane_follow": 2,
    "turn_left": 3,
    "turn_right": 4,
    "go_straight": 5
}

class CarlaGazeVideoDataset(data.Dataset):
    """
    Dataset with only rgb image
    """
    def __init__(self, root="./dataset", img_col=200, img_row=88, gaze_col=20, gaze_row=10, workspace='mkei'):
        self.data_root = root
        self.img_col = img_col
        self.img_row = img_row
        self.gaze_col = gaze_col
        self.gaze_row = gaze_row

        print("Loading dataset... DIRCTORY: %s" % self.data_root)

        self.csv_paths = os.path.join(self.data_root, '*/data.csv')
        self.csv_paths = glob.glob(self.csv_paths)
        print(len(self.csv_paths), self.csv_paths[0], self.csv_paths[-1])

        ### reading csv files
        self.img_path = []
        self.gaze_path = []
        self.cmd_data = []
        for csv_path in self.csv_paths:
            name = os.path.dirname(csv_path)
            #no = csv_path.split('/')[-2]
            # # debug
            # print(csv_path)
            # print(name)
            # skip_count = 0
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if float(row[2]) == 0.0:
                        skip_count += 1
                        continue
                    self.img_path.append(row[0])
                    self.gaze_path.append(os.path.join(name, row[1]))
                    self.cmd_data.append(float(row[2]))
            # # debug
            # print(skip_count)
            # print(self.img_path[0], self.img_path[-1])
            # print(self.gaze_path[0], self.gaze_path[-1])
            # print(self.cmd_data[0], self.cmd_data[-1])
            # print(len(self.img_path), len(self.gaze_path), len(self.cmd_data))
        if workspace == 'mkei':
            pass
        elif workspace == 'moriy':
            ### もりゆPCでも動かせるようにハードコーディング．．．
            self.img_path = [name.replace('/mnt/disk1/', '/home/mkei/mk151/') for name in self.img_path]
        else:
            raise ValueError("Invalid work space : %s" % workspace)

        assert len(self.img_path) == len(self.gaze_path) and len(self.gaze_path) == len(self.cmd_data)
        
        self.data_num = len(self.img_path)

        print('  - # of samples :', self.data_num)
        print('\n')
        
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        gaze_path = self.gaze_path[idx]
        cmd = self.cmd_data[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_col, self.img_row), Image.BILINEAR)
        image = np.asarray(image).transpose(2,0,1).astype(np.float32)
        image /= 255.

        gaze = Image.open(gaze_path).convert('L')
        gaze = gaze.resize((self.gaze_col, self.gaze_row), Image.BILINEAR)
        gaze = np.asarray(gaze).astype(np.float32)
        gaze = np.reshape(gaze, (1, self.gaze_row, self.gaze_col))
        gaze /= 255.
        
        return image, gaze, cmd

