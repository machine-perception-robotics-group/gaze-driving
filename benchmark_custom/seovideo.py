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

import argparse


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        help='experiment name'
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
                st = float(row[1]) #format(float(row[1]), 'f')
                th = float(row[2]) #format(float(row[2]), 'f')
                br = float(row[3]) #format(float(row[3]), 'f')
                sp = float(row[5]) #format(float(row[5]), 'f')
                csv_data.append([row[0], st, th, br, row[4], sp, row[5]])
        #
        ##
        ### result folder 作成
        result_folder = os.path.join(video_folder, episode_folder.split('/')[-1])
        os.makedirs(result_folder, exist_ok=True)

        ##
        if args.org_image:
            print("----original image----")
            org_image_folder = os.path.join(episode_folder, 'original_image/*.jpg')
            org_image_name_list = glob.glob(org_image_folder)
            org_image_name_list.sort()
            if len(org_image_name_list) == 0:
                print("there're no original images")
                continue
            gaze_folder = os.path.join(episode_folder, 'gaze')
            if os.path.exists(gaze_folder):
                gaze_paths = os.path.join(gaze_folder, '*.jpg')
            gaze_name_list = glob.glob(gaze_paths)
            gaze_name_list.sort()
            if len(gaze_name_list) == 0:
                print("no images")
                continue

            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            #video = cv2.VideoWriter('./video_l.mp4', fourcc, 15.0, (800,600))
            video_name = "vis_image.mp4"
            width, height = 800, 395
            video = cv2.VideoWriter(os.path.join(result_folder, video_name), fourcc, 10, (width, height))

            count = 0
            for gaze_name, image_name in zip(gaze_name_list, org_image_name_list):
                if os.path.basename(gaze_name) != os.path.basename(image_name):
                    raise ValueError("mismatched file name : %s, %s" % (gaze_name, image_name))
                img = cv2.imread(image_name)
                gaze = cv2.imread(gaze_name)
                if args.minmax:
                    gaze = gaze.astype(np.float)
                    gaze = (gaze-gaze.min()) / (gaze.max()-gaze.min())
                    gaze = (gaze * 255.).astype(np.uint8)
                gaze = cv2.cvtColor(gaze, cv2.COLOR_BGR2GRAY)
                gaze = cv2.resize(gaze, (width, height))
                gaze = cv2.applyColorMap(gaze, cv2.COLORMAP_JET)
                result = cv2.addWeighted(img, 0.7, gaze, 0.3, 0)
                ### ステアリング可視化用, ステアリング可視化用
                steer = csv_data[count][1]
                throttle = csv_data[count][2]
                brake = csv_data[count][3]
                #####
                centrh, centrw = int(height/2), int(width/2)
                # ステアリング
                angle_inf = -90 + (90*steer)
                # スロットル
                thr_inf = 70 * throttle

                # 外枠
                result = cv2.ellipse(result,(centrw, height-10),(100,100),0,0,-180,(255,0,0),3)
                # 推論
                result = cv2.ellipse(result,(centrw, height-10),(98,98),0,-90,angle_inf,(0,255,0),-1)

                # 外枠
                recth = 140
                sx, sy, ex, ey = 200,  height-(10+recth), 250, height-10
                result = cv2.rectangle(result,(sx,sy),(ex,ey),(230,200,200),2)
                # 推論
                c_rect = int((ey-sy)/2 + sy)
                result = cv2.rectangle(result,(sx, int(c_rect-thr_inf)),(ex, c_rect),(0,255,0),-1)

                slx, sly, elx, ely = sx, int(sy+(ey-sy)/2), ex, int(sy+(ey-sy)/2)
                # 中心線
                result = cv2.line(result,(slx,sly),(elx,ely),(200,10,10),3)

                # 文字出力
                txt1 = "steer   : {}".format(format(float(steer), 'f'))
                txt2 = "throttle: {}".format(format(float(throttle), 'f'))
                result = cv2.putText(result,txt1,(520,height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
                result = cv2.putText(result,txt2,(520,height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
                #####
                count += 1
                ##
                video.write(result)
            video.release()


            # if args.text:
            #     cv2.putText(img, os.path.basename(image_name), 
            #                 (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # if args.print_control:
            #     text = header[4]+': '+csv_data[count][4] + ', ' + header[5]+': '+csv_data[count][5]
            #     cv2.putText(img, text, 
            #                 (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            #     cv2.putText(img, header[1]+': '+csv_data[count][1], 
            #                 (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            #     cv2.putText(img, header[2]+': '+csv_data[count][2], 
            #                 (10, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            #     cv2.putText(img, header[3]+': '+csv_data[count][3], 
            #                 (10, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # ##
        # for feat_name in args.feat:
        #     print("----feature: %s" % feat_name)
        #     feat_folder = os.path.join(episode_folder, feat_name)
        #     if os.path.exists(feat_folder):
        #         feat_paths = os.path.join(feat_folder, '*.jpg')
        #         # result_folder = os.path.join(video_folder, feat_name)
        #         # os.makedirs(result_folder, exist_ok=True)
        #     else:
        #         print('%s does not exist.' % feat_folder)
        #         continue

        #     fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        #     #video = cv2.VideoWriter('./video_l.mp4', fourcc, 15.0, (800,600))
        #     video_name = "%s.mp4" % feat_name
        #     video = cv2.VideoWriter(os.path.join(result_folder, video_name), fourcc, 10, (200, 88))

        #     feat_name_list = glob.glob(feat_paths)
        #     feat_name_list.sort()
        #     if len(feat_name_list) == 0:
        #         print("no images")
        #         continue
        #     count = 0
        #     for feat_name, image_name in zip(feat_name_list, image_name_list):
        #         if os.path.basename(feat_name) != os.path.basename(image_name):
        #             raise ValueError("mismatched file name : %s, %s" % (feat_name, image_name))
        #         img = cv2.imread(image_name)
        #         feat = cv2.imread(feat_name)
        #         if args.minmax:
        #             feat = feat.astype(np.float)
        #             feat = (feat-feat.min()) / (feat.max()-feat.min())
        #             feat = (feat * 255.).astype(np.uint8)
        #         feat = cv2.cvtColor(feat, cv2.COLOR_BGR2GRAY)
        #         feat = cv2.resize(feat, (200, 88))
        #         feat = cv2.applyColorMap(feat, cv2.COLORMAP_JET)

        #         result = cv2.addWeighted(img, 0.7, feat, 0.3, 0)
        #         if args.text:
        #             cv2.putText(result, os.path.basename(feat_name), 
        #                         (0, 50), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #         if args.print_control:
        #             text = header[4]+': '+csv_data[count][4] + ', ' + header[5]+': '+csv_data[count][5]
        #             cv2.putText(result, text, 
        #                         (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        #             cv2.putText(result, header[1]+': '+csv_data[count][1], 
        #                         (10, 66), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        #             cv2.putText(result, header[2]+': '+csv_data[count][2], 
        #                         (10, 72), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        #             cv2.putText(result, header[3]+': '+csv_data[count][3], 
        #                         (10, 78), cv2.FONT_HERSHEY_PLAIN, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        #         count += 1
        #         # cv2.imwrite(os.path.join(result_base, '%05d.png' % i))
        #         video.write(result)

        #     video.release()


