# -*- coding: utf-8 -*-
'''
評価用データローダ
coiltraineのDatasets/CoILValのデータを視線評価に使う用
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

import json
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

        self.img_path, self.command = find_img_folders(self.data_root)

        ## debug
        print(self.img_path[:3], self.command[:3])
        
        self.data_num = len(self.img_path)
        print('  - # of samples :', self.data_num)
        print('\n')
        
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        cmd = self.command[idx]

        row_img = cv2.imread(img_path)
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_col, self.img_row), Image.ANTIALIAS)
        image = np.asarray(image).transpose(2,0,1).astype(np.float32)
        image /= 255.
        
        return row_img, image, cmd


def find_img_folders(rootdir):
    """
    直下に画像が含まれるディレクトリを所得する
    """
    dir_list = glob.glob(path.join(rootdir, "**/Central*.png"), recursive=True)
    # json_list = glob.glob(path.join(rootdir, "**/measurements*"), recrsive=True)
    # print(dir_list)
    dir_list.sort()
    img_folders = []
    img_paths = []
    json_paths = []
    directions = []
    count = 0
    for name in dir_list:
        count += 1
        sys.stdout.write("\r%d/%d" % (count, len(dir_list)))
        sys.stdout.flush()
        episode = name.split('/')[-2].split('_')[-1]
        frame = name.split('.')[-2].split('_')[-1]
        json_name = name.replace("CentralRGB", "measurements").replace("png", "json")
        if os.path.exists(json_name):
            img_paths.append(name)
            json_paths.append(json_name)
            with open(json_name, 'r') as f:
                measurements = json.load(f)
                directions.append(measurements["directions"])

    # print(image_folders)
    # print(image_paths)
    return img_paths, directions

