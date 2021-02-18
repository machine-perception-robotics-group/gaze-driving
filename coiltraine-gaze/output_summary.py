import argparse
import sys
import os
import glob
import torch
# First thing we should try is to import two CARLAS depending on the version


from drive import CoILAgent
from configs import g_conf, merge_with_yaml, set_type_of_process


# Control for CARLA 9

if __name__ == '__main__':


    argparser = argparse.ArgumentParser(description=__doc__)

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
        '-cp', '--checkpoint',
        metavar='P',
        default=100000,
        type=int,
        help='The checkpoint used for the model visualization'
    )
    argparser.add_argument(
        '-de',
        '--drive-envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-o', '--output_folder',
        metavar='P',
        default=None,
        type=str,
        help='The folder to store images received by the network and its activations'
    )

    args = argparser.parse_args()
    merge_with_yaml(os.path.join('configs', args.folder, args.exp + '.yaml'))


    summary_target_path = '%s_%s_%d_drive_control_output_%s' % (args.folder, args.exp, args.checkpoint, args.driving_envs)

