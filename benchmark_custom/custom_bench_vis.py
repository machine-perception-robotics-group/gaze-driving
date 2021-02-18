import os
import glob
import traceback
import collections
import sys
import math
import copy
import json
import random
import numpy as np

import torch
import cv2
import matplotlib.pyplot as plt

import argparse
import datetime
import csv

CMAP = {'jet': cv2.COLORMAP_JET,
        #'inferno': cv2.COLORMAP_INFERNO
        'hot': cv2.COLORMAP_HOT}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        help='experiment name'
    )
    argparser.add_argument(
        '-f',
        '--feat',
        nargs='+',
        type=str,
        help='feature folder name'
    )
    argparser.add_argument(
        '-ep',
        '--episode',
        type=str,
        help='episode name'
    )
    argparser.add_argument(
        '-fn',
        '--frame-no',
        type=int
    )
    argparser.add_argument(
        '-mm',
        '--minmax',
        action='store_true'
    )
    argparser.add_argument(
        '--no-blending',
        action='store_true'
    )
    argparser.add_argument(
        '-hr',
        '--high-res',
        action='store_true'
    )
    argparser.add_argument(
        '-cm',
        '--colormap',
        type=str,
        choices=['jet', 'inferno', 'hot'],
        default='jet'
    )

    args = argparser.parse_args()

    exp_folder = os.path.join('_benchmarks_results', args.exp)
    data_folder = os.path.join(exp_folder, '_images')
    # os.makedirs(result_base, exist_ok=True)

    ## result folderを作成
    now = datetime.datetime.now()
    basename = now.strftime('%Y%m%d_%H%M%S_')
    result_folder = 'vis_results'
    os.makedirs(result_folder, exist_ok=True)

    # #for episode_name in args.episode:
    # for episode_folder in episodes:
    episode_folder = os.path.join(data_folder, args.episode)

    # print("EPISODE: %s" % episode_name)
    print("FOLDER: %s" % episode_folder)
    ##
    # episode_folder = os.path.join(data_folder, episode_name)
    if not os.path.exists(episode_folder):
        print('EP %s does not exist.' % episode_folder)
        sys.exit()

    ## 背景に使う画像とか解像度とか
    width = 200
    height = 88
    image_name = os.path.join(episode_folder, 'image/image_%05d.jpg' % args.frame_no)
    if args.high_res:
        width = 800
        height = 395
        image_name = os.path.join(episode_folder, 'original_image/image_%05d.jpg' % args.frame_no)
    if not os.path.exists(image_name):
        print('Image %s does not exist.' % image_name)

    # ## 
    # image_name = os.path.join(episode_folder, 'image/image_%05d.jpg' % args.frame_no)
    # if not os.path.exists(image_name):
    #     print('Image %s does not exist.' % image_name)

    with open(os.path.join(episode_folder, 'episode_measurements.csv'), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if args.frame_no == int(row[0]):
                steer = row[1]
                throttle = row[2]
                brake = row[3]
                directions = row[4]
                speed_sim = row[5]
                speed_inf = row[6]

    ### log
    with open(os.path.join(result_folder, basename+'log.txt'), 'w') as f:
        f.write("%s\n" % args.exp)
        f.write("%s\n" % args.episode)
        f.write("%s\n" % args.frame_no)
        f.write("steer: %s\n" % steer)
        f.write("throttle: %s\n" % throttle)
        f.write("brake: %s\n" % brake)
        f.write("directions: %s\n" % directions)
        f.write("speed_sim: %s\n" % speed_sim)
        f.write("speed_inf: %s\n" % speed_inf)
        ##
        for feat_path in args.feat:
            print("----feature: %s" % feat_path)
            feat_folder = os.path.join(episode_folder, feat_path)
            if os.path.exists(feat_folder):
                feat_name = os.path.join(feat_folder, 'image_%05d.jpg' % args.frame_no)
            else:
                print('%s does not exist.' % feat_folder)
                continue
            #

            if os.path.basename(feat_name) != os.path.basename(image_name):
                raise ValueError("mismatched file name : %s, %s" % (feat_name, image_name))
            img = cv2.imread(image_name)
            feat = cv2.imread(feat_name)
            if args.minmax:
                feat = feat.astype(np.float)
                feat = (feat-feat.min()) / (feat.max()-feat.min())
                feat = (feat * 255.).astype(np.uint8)
            feat = cv2.cvtColor(feat, cv2.COLOR_BGR2GRAY)
            feat = cv2.resize(feat, (width, height))
            if args.colormap=='inferno':
                cmap = plt.get_cmap('inferno')
                feat = cmap(feat)
                feat = np.delete(feat, 3, 2)
                feat = (feat*255.).astype(np.uint8)
                feat = cv2.cvtColor(feat, cv2.COLOR_RGB2BGR)
                ratio = [0.5, 0.5]
            else:
                cmap = CMAP[args.colormap]
                ratio = [0.7, 0.3]
                feat = cv2.applyColorMap(feat, cmap)

            if not args.no_blending:
                result = cv2.addWeighted(img, ratio[0], feat, ratio[1], 0)
            else:
                result = feat

            cv2.imwrite(os.path.join(result_folder, basename+feat_path+'.jpg'), result)
            f.write("%s\n" % feat_path)



