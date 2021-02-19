import cv2
import glob
from os import path
import argparse
import re
# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Self-driving Training')

parser.add_argument('image_path', help='path to attention map', type=str)
# parser.add_argument('size', help='size', type=str)
args = parser.parse_args()

# 要変更箇所
WIDTH, HEIGHT = 200, 88
fps = 10.0


def make_video(video_path, img_list):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    #video = cv2.VideoWriter('./video_l.mp4', fourcc, 15.0, (800,600))
    video = cv2.VideoWriter(video_path, fourcc, fps, (WIDTH,HEIGHT))



    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = cv2.resize(img, (WIDTH,HEIGHT))
        video.write(img)


    video.release()


def main():
    #filepath = path.join(args.image_path, "video_pred.mp4")
    #img_path = path.join(args.image_path, "output_conv/pred/*.png")
    #img_list = glob.glob(img_path)
    #img_list.sort()
    #make_video(filepath, img_list)
    # ep_dir = path.join(args.image_path, "episode_?_[1-2]_*")
    # all_dir_paths = glob.glob(ep_dir)
    all_dir_paths = [args.image_path]
    count = 0
    for dir_path in all_dir_paths:
        print("%d/%d" % (count, len(all_dir_paths)))
        img_list = glob.glob(path.join(dir_path, "*.png"))
        print(len(img_list))
        img_list.sort()
        # one_ep = dir_path.split("/")[-1]
        # filepath = path.join(args.image_path, one_ep+".mp4")
        filepath = path.join(args.image_path, "result.mp4")
        make_video(filepath, img_list)
        count += 1

    # filepath = path.join(args.image_path, "video_att_10.mp4")
    # img_path = path.join(args.image_path, "output_conv/attention/*_att.png")
    # img_list = glob.glob(img_path)
    # img_list.sort()
    # make_video(filepath, img_list)

    # # import pdb;pdb.set_trace()
    # filepath = path.join(args.image_path, "video_pred.mp4")
    # img_path = path.join(args.image_path, "output_conv/*")
    # img_list = [p for p in glob.glob(img_path, recursive=False) if re.search('\d+\.png', p)]
    # img_list = [a for a in img_list if 'att' not in a]
    # img_list.sort()
    # #import pdb;pdb.set_trace()
    # make_video(filepath, img_list)

if __name__ == '__main__':
    main()

