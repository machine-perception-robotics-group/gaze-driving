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
# REACH_GOAL = 0.0
# GO_STRAIGHT = 5.0
# TURN_RIGHT = 4.0
# TURN_LEFT = 3.0
# LANE_FOLLOW = 2.0

class CarlaGazeDataset(data.Dataset):
    """
    Dataset with only rgb image
    """
    def __init__(self, root="./dataset", img_col=200, img_row=88, gaze_col=20, gaze_row=10, val=False):
        self.data_root = root
        self.data_num = 0
        self.img_col = img_col
        self.img_row = img_row
        self.gaze_col = gaze_col
        self.gaze_row = gaze_row

        print("Loading dataset... DIRCTORY: %s" % self.data_root)

        _, self.img_paths = find_img_folders(self.data_root)

        self.command = []
        self.gaze_path = []
        self.img_path = []

        for node in self.img_paths:
            for item in node:
                item = item.replace('csv', 'png')
                self.img_path.append(item)
                self.gaze_path.append(path.join(path.dirname(item), 
                                     "Gaze"+path.basename(item)))
                # self.gaze_path.append(item)
                # self.img_path.append(item.replace("Gaze", ""))
                self.command.append(CVT_DICT[item.split("/")[-3]])
        ## debug
        print(self.gaze_path[:3], self.img_path[:3], self.command[:3])
        
        self.data_num = len(self.gaze_path)
        print('  - # of samples :', self.data_num)
        print('\n')
        
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        gaze_path = self.gaze_path[idx]
        cmd = self.command[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_col, self.img_row), Image.ANTIALIAS)
        image = np.asarray(image).transpose(2,0,1).astype(np.float32)
        image /= 255.

        gaze = Image.open(gaze_path).convert('L')
        gaze = gaze.resize((self.gaze_col, self.gaze_row), Image.ANTIALIAS)
        gaze = np.asarray(gaze).astype(np.float32)
        gaze = np.reshape(gaze, (1, self.gaze_row, self.gaze_col))
        gaze /= 255.
        
        return image, gaze, cmd


def find_img_folders(rootdir):
    """
    直下に画像が含まれるディレクトリを所得する
    """
    dir_list = glob.glob(path.join(rootdir, "**/"), recursive=True)
    # print(dir_list)
    dir_list.sort()
    img_folders = []
    img_paths = []
    for i, directory in enumerate(dir_list):
        img_list = glob.glob(path.join(directory, '*.csv'))
        if len(img_list) > 0:
            # print(len(img_list))
            # print("append directory named: %s" % directory)
            img_folders.append(directory)
            img_list.sort()
            img_paths.append(img_list)
    # print(image_folders)
    # print(image_paths)
    return img_folders, img_paths

