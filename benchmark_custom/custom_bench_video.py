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
import csv
import matplotlib.pyplot as plt

import argparse

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
        default=[],
        help='feature folder name'
    )
    argparser.add_argument(
        '-ep',
        '--episode',
        nargs='+',
        type=str,
        help='episode name'
    )
    argparser.add_argument(
        '-t',
        '--text',
        action='store_true'
    )
    argparser.add_argument(
        '-c',
        '--print-control',
        action='store_true'
    )
    argparser.add_argument(
        '-mm',
        '--minmax',
        action='store_true'
    )
    argparser.add_argument(
        '-i',
        '--image',
        action='store_true'
    )
    argparser.add_argument(
        '-oi',
        '--org-image',
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

    ### 'all'があれば全て読み込む
    if 'all' in args.episode:
        episodes = glob.glob(os.path.join(data_folder, 'episode*'))
    ### なければ任意のフォルダーのみ
    else:
        episodes = [os.path.join(data_folder, episode_name) for episode_name in args.episode]

    ## video result folderを作成
    video_folder = os.path.join(exp_folder, 'video_results')
    os.makedirs(video_folder, exist_ok=True)

    #for episode_name in args.episode:
    for episode_folder in episodes:
        # print("EPISODE: %s" % episode_name)
        print("FOLDER: %s" % episode_folder)
        #
        # episode_folder = os.path.join(data_folder, episode_name)
        if not os.path.exists(episode_folder):
            print('EP %s does not exist.' % episode_folder)
            continue
        image_folder = os.path.join(episode_folder, 'image/*.jpg')
        image_name_list = glob.glob(image_folder)
        image_name_list.sort()
        if len(image_name_list) == 0:
            print("no images")
            continue
        #
        ## reading CSV data
        csv_data = []
        csv_path = os.path.join(episode_folder, "episode_measurements.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                st = format(float(row[1]), 'f')
                th = format(float(row[2]), 'f')
                br = format(float(row[3]), 'f')
                sp = format(float(row[5]), 'f')
                csv_data.append([row[0], str(st), str(th), str(br), row[4], str(sp), row[5]])
        #
        ##
        ### result folder 作成
        result_folder = os.path.join(video_folder, episode_folder.split('/')[-1])
        os.makedirs(result_folder, exist_ok=True)
        ##
        if args.image:
            print("----image----")
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            #video = cv2.VideoWriter('./video_l.mp4', fourcc, 15.0, (800,600))
            video_name = "image.mp4"
            video = cv2.VideoWriter(os.path.join(result_folder, video_name), fourcc, 10, (200, 88))
            for image_name in image_name_list:
                img = cv2.imread(image_name)
                video.write(img)
            video.release()
        ##
        ##original
        org_image_folder = os.path.join(episode_folder, 'original_image/*.jpg')
        org_image_name_list = glob.glob(org_image_folder)
        org_image_name_list.sort()
        if args.org_image:
            print("----original image----")
            if len(org_image_name_list) == 0:
                print("there're no original images")
                continue
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            #video = cv2.VideoWriter('./video_l.mp4', fourcc, 15.0, (800,600))
            video_name = "original_image.mp4"
            video = cv2.VideoWriter(os.path.join(result_folder, video_name), fourcc, 10, (800, 395))
            count = 0
            for image_name in org_image_name_list:
                img = cv2.imread(image_name)
                if args.text:
                    cv2.putText(img, os.path.basename(image_name), 
                                (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
                if args.print_control:
                    text = header[4]+': '+csv_data[count][4] + ', ' + header[5]+': '+csv_data[count][5]
                    cv2.putText(img, text, 
                                (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, header[1]+': '+csv_data[count][1], 
                                (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, header[2]+': '+csv_data[count][2], 
                                (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, header[3]+': '+csv_data[count][3], 
                                (10, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
                count += 1
                video.write(img)
            video.release()

        ## 背景に使う画像とか解像度とか
        back_img_name_list = image_name_list
        width = 200
        height = 88
        if args.high_res:
            assert len(org_image_name_list) > 0
            back_img_name_list = org_image_name_list
            width = 800
            height = 395
        ##
        for feat_name in args.feat:
            print("----feature: %s" % feat_name)
            feat_folder = os.path.join(episode_folder, feat_name)
            if os.path.exists(feat_folder):
                feat_paths = os.path.join(feat_folder, '*.jpg')
                # result_folder = os.path.join(video_folder, feat_name)
                # os.makedirs(result_folder, exist_ok=True)
            else:
                print('%s does not exist.' % feat_folder)
                continue

            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            #video = cv2.VideoWriter('./video_l.mp4', fourcc, 15.0, (800,600))
            video_name = "%s.mp4" % feat_name
            video = cv2.VideoWriter(os.path.join(result_folder, video_name), fourcc, 10, (width, height))

            feat_name_list = glob.glob(feat_paths)
            feat_name_list.sort()
            if len(feat_name_list) == 0:
                print("no images")
                continue
            count = 0

            for feat_name, image_name in zip(feat_name_list, back_img_name_list):
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

                result = cv2.addWeighted(img, ratio[0], feat, ratio[1], 0)
                if args.text:
                    cv2.putText(result, os.path.basename(feat_name), 
                                (0, 50), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if args.print_control:
                    text = header[4]+': '+csv_data[count][4] + ', ' + header[5]+': '+csv_data[count][5]
                    cv2.putText(result, text, 
                                (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(result, header[1]+': '+csv_data[count][1], 
                                (10, 66), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(result, header[2]+': '+csv_data[count][2], 
                                (10, 72), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(result, header[3]+': '+csv_data[count][3], 
                                (10, 78), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                count += 1
                # cv2.imwrite(os.path.join(result_base, '%05d.png' % i))
                video.write(result)

            video.release()


