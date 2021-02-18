import numpy as np
import scipy
import sys
import os
import glob
import torch
import csv
import cv2
import matplotlib.pyplot as plt

import argparse

from coilutils.general import create_log_folder, create_exp_path, erase_logs,\
                          erase_wrong_plotting_summaries, erase_validations

# from .coil_core import run_drive_custom
import multiprocessing

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto, get_gaze_model #, GazeModel, GazeModelMN
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, \
                                    check_loss_validation_stopped

# You could send the module to be executed and they could have the same interface.



#
# The main function maybe we could call it with a default name
def execute(gpu, base_folder, exp_batch, exp_alias, checkpoint_num, output_folder, data_info, no_blending, suppress_output=True, number_of_workers=12):
    """
        The main training function. This functions loads the latest checkpoint
        for a given, exp_batch (folder) and exp_alias (experiment configuration).
        With this checkpoint it starts from the beginning or continue some training.
    Args:
        gpu: The GPU number
        exp_batch: the folder with the experiments
        exp_alias: the alias, experiment name
        suppress_output: if the output are going to be saved on a file
        number_of_workers: the number of threads used for data loading

    Returns:
        None

    """
    try:
        # We set the visible cuda devices to select the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu #"0, 1"
        g_conf.VARIABLE_WEIGHT = {}
        # At this point the log file with the correct naming is created.
        # You merge the yaml file with the global configuration structure.
        config_path = os.path.join(base_folder, 'configs', exp_batch, exp_alias + '.yaml')
        merge_with_yaml(config_path)
        # set_type_of_process('train')

        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': gpu})

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint = torch.load(os.path.join(base_folder, '_logs', exp_batch, exp_alias,
                                'checkpoints', str(checkpoint_num)+'.pth'))
        iteration = checkpoint['iteration']
        best_loss = checkpoint['best_loss']
        best_loss_iter = checkpoint['best_loss_iter']

        ## 変更
        if g_conf.MODEL_TYPE == "coil-gaze" or g_conf.MODEL_TYPE == "coil-gaze-loss":
            gaze_model = get_gaze_model(g_conf.GAZE_MODEL_TYPE, g_conf.GAZE_CHECKPOINT, exp_batch, exp_alias, "toy", base_folder)
            # ## get loss function for feature mimicking.
            # if g_conf.MODEL_CONFIGURATION['perception']['res_g']['gtype'] == 'mimick':
            #     mimick_criterion = Loss('mimick_L2')
            # # gaze_model = GazeModelMN(exp_batch, exp_alias, "train")
        """ABN+視線用の視線モデル"""
        if g_conf.MODEL_TYPE == "abn-drive":
            if 'gmask' in g_conf.MODEL_CONFIGURATION or 'gloss' in g_conf.MODEL_CONFIGURATION:
                gaze_model = get_gaze_model(g_conf.GAZE_MODEL_TYPE, g_conf.GAZE_CHECKPOINT, exp_batch, exp_alias, "toy", base_folder)
                use_gaze_for_abn = True
            else:
                use_gaze_for_abn = False


        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()
        # optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)


        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        print ("Before the loss")


        # # attention map 用のloss
        # if g_conf.MODEL_TYPE == 'abn-drive' and 'gloss' in g_conf.MODEL_CONFIGURATION:
        #     criterion_att = Loss(g_conf.MODEL_CONFIGURATION['gloss']['loss'])

        ### 必要な情報など
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        feat_folder = []
        for i in range(4):
            name = os.path.join(output_folder, 'feat%d' % i)
            os.makedirs(name)
            feat_folder.append(name)
        ##
        gaze_folder = None
        if g_conf.MODEL_TYPE == "coil-gaze" or g_conf.MODEL_TYPE == "coil-gaze-loss":
            gaze_folder = os.path.join(output_folder, 'gaze')
            os.makedirs(gaze_folder)
        ##
        image_folder = os.path.join(output_folder, 'image')
        os.makedirs(image_folder)
        csv_out_path = os.path.join(output_folder, 'out_measurements.csv')
        ##

        # data読み込み
        with open(data_info['csv_path'], 'r') as f:
            frame_no = []
            all_directions = []
            all_speed = []
            reader = csv.reader(f)
            header = next(reader)
            print(header[0], header[4], header[5])
            for row in reader:
                frame_no.append(int(row[0]))
                all_directions.append(float(row[4]))
                all_speed.append(float(row[5]))
        with open(csv_out_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['frame', 'steer', 'throttle', 'brake'])

        model.eval()

        with torch.no_grad():
            #for img_path in image_path:
            for i in range(len(frame_no)):
                sys.stdout.write('\r%d/%d' % (i, len(frame_no)))
                sys.stdout.flush()
                """
                    ####################################
                        Main optimization loop
                    ####################################
                """
                ## data : image data path
                # assert os.path.exists(img_path)
                # info_path = img_path.replace('png', 'csv')
                # assert info_path
                # out_path = info_path.replace('.csv', '_out.csv')
                # print(info_path, out_path)
                # with open(info_path, 'r') as f:
                #     reader = csv.reader(f)
                #     row = next(reader)
                #     speed = np.array(float(row[0])).reshape(1,1) / g_conf.SPEED_FACTOR
                #     control = np.array(CVT_DICT[row[1]]).reshape(1,1)

                frame = frame_no[i]
                img_path = os.path.join(data_info['img_path'], 'image_%05d.jpg' % frame)
                control = np.array(all_directions[i]).reshape(1,1)
                speed = np.array(all_speed[i]).reshape(1,1) / g_conf.SPEED_FACTOR

                org_image = cv2.imread(img_path)
                image = org_image.transpose(2, 0, 1).astype(np.float) / 255.
                ch, h, w = image.shape
                image = image.reshape(1,ch,h,w)

                # g_conf.SPEED_FACTOR
                image_tensor = torch.tensor(image, dtype=torch.float).cuda()
                speed_tensor = torch.tensor(speed, dtype=torch.float).cuda()
                control_tensor = torch.tensor(control, dtype=torch.float).cuda()

                # print(image_tensor.shape)
                # print(speed_tensor.shape)
                # print(control_tensor.shape)


                # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
                model.zero_grad()
                if g_conf.MODEL_TYPE == "coil-icra" or g_conf.MODEL_TYPE == "normal-drive":
                    # branches = model(torch.squeeze(data['rgb'].cuda()),
                    #              dataset.extract_inputs(data).cuda())
                    branches = model(image_tensor, speed_tensor)
                elif g_conf.MODEL_TYPE == "coil-gaze":
                    gaze_map = gaze_model.run_step(image_tensor, control_tensor)
                    branches = model(image_tensor, speed_tensor, gaze_map)

                elif g_conf.MODEL_TYPE == "coil-gaze-loss":
                    gaze_map = gaze_model.run_step(image_tensor, control_tensor)
                    branches = model(image_tensor, speed_tensor)

                elif g_conf.MODEL_TYPE == "abn-drive":
                    if use_gaze_for_abn:
                        gaze_map = gaze_model.run_step(image_tensor, control_tensor)
                    ##### 普通のabnの場合にerrorなるかも？(~~before assignment)その場合は下のバージョンを使う
                    items = [image_tensor, speed_tensor, gaze_map] \
                            if 'gmask' in g_conf.MODEL_CONFIGURATION else \
                            [image_tensor, speed_tensor]
                    branches = model(*items)

                    ab_branches = branches[4:8] # attention branchの制御値出力
                    # att_maps = branches[8:12] ##
                    branches = branches[0:4] + [branches[-1]] ## perception branch の出力と速度推定出力
                control_tensor = control_tensor.reshape([1,])
                output = model.extract_branch(torch.stack(branches[0:4]), control_tensor)
                output = output.cpu().numpy()

                save_attentions(org_image, model.inter, feat_folder, frame, no_blending)
                if gaze_folder is not None:
                    save_gaze(org_image, gaze_map, gaze_folder, frame, no_blending)

                with open(csv_out_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame] + output[0].tolist())
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('finish.')

def save_attentions(image, featmaps, out_folder, frame, no_blending, layers=[1,2,3,4]):
    # all_layers = self._model.get_perception_layers(self.latest_image_tensor)
    # all_layers = self._model.inter ## convモデルとresnetモデルからinterを受け取れるように変更したらon
    # cmap = plt.get_cmap('inferno')
    cmap = plt.get_cmap('gray')
    attentions = []
    for layer, folder in zip(layers, out_folder):
        y = featmaps[layer]
        att = torch.abs(y).mean(1)[0].data.cpu().numpy()
        ### Matplot版
        # att = att / att.max()
        att = (att - att.min()) / (att.max() - att.min())
        # att = cmap(imresize(att, [88, 200]))# att = imresize(att, [88, 200])# 
        att = cmap(att)
        att = np.delete(att, 3, 2)
        att = (att*255.).astype(np.uint8)
        att = cv2.cvtColor(att, cv2.COLOR_RGB2GRAY)
        att = cv2.resize(att, (200, 88))
        att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        if no_blending:
            result = att
        else:
            result = cv2.addWeighted(image, 0.7, att, 0.3, 0)
        cv2.imwrite(os.path.join(folder, '%05d.jpg' % frame), result)
        # cv2.imwrite(os.path.join(folder, '%s_%05d.jpg' % (folder.split('/')[-1], frame)))

def save_gaze(image, gaze, out_folder, frame, no_blending):
    cmap = plt.get_cmap('gray')

    att = torch.abs(gaze)[0][0].data.cpu().numpy()
    ### Matplot版
    att = att / att.max()
    # att = (att - att.min()) / (att.max() - att.min())
    att = cmap(att)
    att = np.delete(att, 3, 2)
    att = (att*255.).astype(np.uint8)
    att = cv2.cvtColor(att, cv2.COLOR_RGB2GRAY)
    att = cv2.resize(att, (200, 88))
    att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    if no_blending:
        result = att
    else:
        result = cv2.addWeighted(image, 0.7, att, 0.3, 0)
    cv2.imwrite(os.path.join(out_folder, '%05d.jpg' % frame), result)




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--gpus',
        #nargs='+',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '-d',
        '--data-path',
        dest='data_path',
        default=None
        # dest='data_path',
        # nargs='+',
        # default=[]
    )
    argparser.add_argument(
        '-o', '--output_folder',
        metavar='P',
        default=None,
        type=str,
        help='The folder to store images received by the network and its activations'
    )
    argparser.add_argument(
        '-cp', '--checkpoint',
        metavar='P',
        default=100000,
        type=int,
        help='The checkpoint used for the model visualization'
    )
    argparser.add_argument(
        '-bf', '--base-folder',
        default="./",
        type=str,
        help='Base folder including the model running'
    )
    argparser.add_argument(
        '--no-blending',
        action='store_true'
    )

    args = argparser.parse_args()

    # Check if the vector of GPUs passed are valid.
    for gpu in args.gpus:
        try:
            int(gpu)
        except ValueError:  # Reraise a meaningful error.
            raise ValueError("GPU is not a valid int number")

    # Check if the mandatory folder argument is passed
    if args.folder is None:
        raise ValueError("You should set a folder name where the experiments are placed")


    # # This is the folder creation of the
    # create_log_folder(args.folder)
    # erase_logs(args.folder)

    # The definition of parameters for driving
    drive_params = {
        "suppress_output": True,
        "base_folder": args.base_folder
    }
    ####
    # MODE 1: Single Process. Just execute a single experiment alias.
    ####

    if args.exp is None:
        raise ValueError(" You should set the exp alias when using single process")

    # create_exp_path(args.folder, args.exp)

    drive_params['suppress_output'] = False
    # execute_drive(args.gpus, args.folder, args.exp, list(args.driving_environments)[0], drive_params, args.save_images)

    checkpoint_number = args.checkpoint
    print('a')
    # create_exp_path(args.folder, args.exp)
    csv_path = os.path.join(args.data_path, 'episode_measurements.csv')
    img_path = os.path.join(args.data_path, 'image/')
    assert os.path.exists(csv_path) and os.path.exists(img_path)
    data_info = {
        "csv_path": csv_path,
        "img_path": img_path
    }
    execute(args.gpus, args.base_folder, args.folder, args.exp, checkpoint_number, args.output_folder, data_info, args.no_blending)

