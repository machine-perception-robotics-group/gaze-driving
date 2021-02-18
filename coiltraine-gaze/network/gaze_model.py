"""
    A simple factory module that returns instances of possible modules 

"""

## 必要に応じて変更
from .models.gaze_models import ConvModel, ConvModelCMD
from .models.gaze_models import MobileNetV2, MobileNetV2CMD
from .models.gaze_models import CoConvModel

from PIL import Image
import torch.nn.functional as F
import torch
import os
from collections import OrderedDict

from configs import g_conf


def get_gaze_model(model_name, checkpoint, exp_batch, exp_alias, process):

    if model_name == 'mobilenet-v2':
        return GazeModelMN(checkpoint, exp_batch, exp_alias, process)
    elif model_name == 'vgg':
        return GazeModel(checkpoint, exp_batch, exp_alias, process)
    elif model_name == 'vgg-cmd':
        return GazeModel(checkpoint, exp_batch, exp_alias, process, True)
    elif model_name == 'mobilenet-v2-cmd':
        return GazeModelMN(checkpoint, exp_batch, exp_alias, process, True)
    elif model_name == 'co-convnet':
        return CoGazeModel(checkpoint, exp_batch, exp_alias, process)
    else:
        raise ValueError("Invalid gaze_model option %s" % model_name)



def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

class GazeModel(object):
    def __init__(self, checkpoint, exp_batch, exp_alias, process, cmd_input=False):

        self.checkpoint_base = "gaze_checkpoints/"
        self.checkpoint_name = checkpoint# "model_100_vgg.pth"

        cmdlog = 'cmd' if cmd_input else 'no cmd'
        #log_path = '_log/' + exp_batch + '/' exp_alias + '/' + '%s_gaze_conf.log' % process
        log_path = os.path.join('_logs/', exp_batch, exp_alias, '%s_gaze_conf.log' % process)
        with open(log_path, 'w') as f:
            f.write("Gaze model used: GazeModel (VGG) with %s\n" % cmdlog)
            f.write("Checkpoint path loaded: %s" % (self.checkpoint_base+self.checkpoint_name))

        ## 必要に応じて変更
        if cmd_input:
            self.model = ConvModelCMD()
        else:
            self.model = ConvModel()
        # self.model = torch.nn.DataParallel(self.model)
        # self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_base, self.checkpoint_name)))
        self.model.load_state_dict(fix_model_state_dict(
                torch.load(os.path.join(self.checkpoint_base, self.checkpoint_name))
            ))
        self.model.cuda()
        self.model.eval()

        # self.i_size = (400, 176) # w, h #### 11/31やばいミス…
        self.i_size = (176, 400) # w, h
        # except:
        #     import traceback
        #     traceback.print_exc()

        # self.g_size = () # w, h モデルのサイズに応じでリサイズする

    def run_step(self, rgb_image, cmd):
        # # PILのresizeデフォがbicubic ### 5.3.0ではnearestでした．
        # ToDo: optionで変更できるようにする．__init__でmode名を宣言しておくなど
        # rgb_image = F.interpolate(rgb_image, size=self.i_size, mode='bilinear', align_corners=False)
        # rgb_image = F.interpolate(rgb_image, size=self.i_size, mode='nearest', align_corners=False) ## nearestに非対応
        rgb_image = F.upsample(rgb_image, size=self.i_size, mode='nearest')
        with torch.no_grad():
            gaze_map = self.model(rgb_image, cmd)
        # gaze_map = F.interpolate(gaze_map, size=self.g_size, mode='bicubic')
        return gaze_map

class GazeModelMN(object):
    def __init__(self, checkpoint, exp_batch, exp_alias, process, cmd_input=False):

        self.checkpoint_base = "gaze_checkpoints/"
        self.checkpoint_name = checkpoint# "model_100_mnv2.pth"

        cmdlog = 'cmd' if cmd_input else 'no cmd'
        #log_path = '_log/' + exp_batch + '/' + exp_alias + '/' + '%s_gaze_conf.log' % process
        log_path = os.path.join('_logs/', exp_batch, exp_alias, '%s_gaze_conf.log' % process)
        with open(log_path, 'w') as f:
            f.write("Gaze model used: GazeModelMN with %s\n" % cmdlog)
            f.write("Checkpoint path loaded: %s" % (self.checkpoint_base+self.checkpoint_name))

        ## 必要に応じて変更
        if cmd_input:
            self.model = MobileNetV2CMD()
        else:
            self.model = MobileNetV2()
        # self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_base, self.checkpoint_name)))
        # self.model.load_state_dict(fix_model_state_dict(
        #         torch.load(os.path.join(self.checkpoint_base, self.checkpoint_name))
        #     ))
        self.model.cuda()
        self.model.eval()

        # self.i_size = (400, 176) # w, h #### 11/31やばいミス…
        self.i_size = (176, 400) # w, h
        # except:
        #     import traceback
        #     traceback.print_exc()
        
        # self.g_size = () # w, h モデルのサイズに応じでリサイズする

    def run_step(self, rgb_image, cmd):
        # # PILのresizeデフォがbicubic ### 5.3.0ではnearestでした．
        # ToDo: optionで変更できるようにする．__init__でmode名を宣言しておくなど
        # rgb_image = F.interpolate(rgb_image, size=self.i_size, mode='bilinear', align_corners=False)
        rgb_image = F.interpolate(rgb_image, size=self.i_size, mode='nearest', align_corners=False)
        with torch.no_grad():
            gaze_map = self.model(rgb_image, cmd)
        # gaze_map = F.interpolate(gaze_map, size=self.g_size, mode='bicubic')
        return gaze_map


class CoGazeModel(object):
    def __init__(self, checkpoint, exp_batch, exp_alias, process):#, interpolate='bilinear'):

        self.checkpoint_base = "gaze_checkpoints/"
        self.checkpoint_name = checkpoint# "model_100_vgg.pth"

        # cmdlog = 'cmd' if cmd_input else 'no cmd'
        #log_path = '_log/' + exp_batch + '/' exp_alias + '/' + '%s_gaze_conf.log' % process
        log_path = os.path.join('_logs/', exp_batch, exp_alias, '%s_gaze_conf.log' % process)
        with open(log_path, 'w') as f:
            f.write("Gaze model used: ConditionalGazeModel (VGG)")
            f.write("Checkpoint path loaded: %s" % (self.checkpoint_base+self.checkpoint_name))

        # self.interpolate = interpolate
        self.model = CoConvModel()
        # self.model = torch.nn.DataParallel(self.model)
        # self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_base, self.checkpoint_name)))
        self.model.load_state_dict(fix_model_state_dict(
                torch.load(os.path.join(self.checkpoint_base, self.checkpoint_name))
            ))
        self.model.cuda()
        self.model.eval()

        # self.i_size = (400, 176) # w, h #### 11/31やばいミス…
        self.i_size = (176, 400) # w, h
        # except:
        #     import traceback
        #     traceback.print_exc()

        # self.g_size = () # w, h モデルのサイズに応じでリサイズする

    def run_step(self, rgb_image, cmd):
        # # PILのresizeデフォがbicubic ### 5.3.0ではnearestでした．
        # ToDo: optionで変更できるようにする．__init__でmode名を宣言しておくなど
        # rgb_image = F.interpolate(rgb_image, size=self.i_size, mode='bilinear', align_corners=False)
        # rgb_image = F.interpolate(rgb_image, size=self.i_size, mode='nearest', align_corners=False) ## nearestに非対応
        rgb_image = F.interpolate(rgb_image, size=self.i_size, mode='bilinear', align_corners=False) #mode=self.interpolate)
        with torch.no_grad():
            gaze_map = self.model.forward_branch(rgb_image, cmd)
        # gaze_map = F.interpolate(gaze_map, size=self.g_size, mode='bicubic')
        return gaze_map