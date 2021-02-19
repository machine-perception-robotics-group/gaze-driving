# -*- coding: utf-8 -*-
'''
ネットワークモ
'''

from . import CoConvModel
from . import ConvModelCMD
from . import ConvModelCMD_v2

def get_model(model_name=None):
    if model_name == 'co-conv':
        return CoConvModel()
    elif model_name == 'conv-cmd':
        return ConvModelCMD()
    elif model_name == 'conv-cmd-v2':
        return ConvModelCMD_v2()
    else:
        raise ValueError("Invalid model name: %s" % model_name)