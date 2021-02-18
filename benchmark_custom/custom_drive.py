import argparse

from coil_core import execute_drive, folder_execute
from coilutils.general import create_log_folder, create_exp_path, erase_logs,\
                          erase_wrong_plotting_summaries, erase_validations

# from .coil_core import run_drive_custom
import coil_core.run_drive_custom as run_drive_custom

import multiprocessing

# You could send the module to be executed and they could have the same interface.

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
    # argparser.add_argument(
    #     '-vd',
    #     '--val-datasets',
    #     dest='validation_datasets',
    #     nargs='+',
    #     default=[]
    # )
    # argparser.add_argument(
    #     '--no-train',
    #     dest='is_training',
    #     action='store_false'
    # )
    argparser.add_argument(
        '-de',
        '--drive-envs',
        dest='driving_environments',
        nargs='+',
        choices=['NocrashCustom_Town01', 'NocrashCustomNewTown_Town02']
        # default=[]
    )
    # argparser.add_argument(
    #     '-v', '--verbose',
    #     action='store_true',
    #     dest='debug',
    #     help='print debug information')

    argparser.add_argument(
        '-ns', '--no-screen',
        action='store_true',
        dest='no_screen',
        help='Set to carla to run offscreen'
    )
    # argparser.add_argument(
    #     '-ebv', '--erase-bad-validations',
    #     action='store_true',
    #     dest='erase_bad_validations',
    #     help='erase the bad validations (Incomplete)'
    # )
    # argparser.add_argument(
    #     '-rv', '--restart-validations',
    #     action='store_true',
    #     dest='restart_validations',
    #     help='Set to carla to run offscreen'
    # )
    # argparser.add_argument(
    #     '-gv',
    #     '--gpu-value',
    #     dest='gpu_value',
    #     type=float,
    #     default=3.5
    # )
    # argparser.add_argument(
    #     '-nw',
    #     '--number-of-workers',
    #     dest='number_of_workers',
    #     type=int,
    #     default=12
    # )
    argparser.add_argument(
        '-dk', '--docker',
        dest='docker',
        default='carlasim/carla:0.8.4',
        type=str,
        help='Set to run carla using docker'
    )
    argparser.add_argument(
        '-rc', '--record-collisions',
        action='store_true',
        dest='record_collisions',
        help='Set to run carla using docker'
    )
    argparser.add_argument( # こっちは使わない
        '-si', '--save-images',
        action='store_true',
        dest='save_images',
        help='save images during benchmark.'
    )
    argparser.add_argument(
        '-sf', '--save-features',
        action='store_true',
        dest='save_features',
        help='save features (also input images) during benchmark.'
    )
    argparser.add_argument(
        '-w',
        '--weather',
        type=int,
        choices=[1, 3, 6, 8, 10, 14],
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-ep',
        '--episode_number',
        type=int,
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-tf',
        '--traffic',
        type=str, 
        default='all',
        choices=['all', 'empty', 'normal', 'cluttered']
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

    # Check if the driving parameters are passed in a correct way
    if args.driving_environments is not None:
        for de in list(args.driving_environments):
            if len(de.split('_')) < 2:
                raise ValueError("Invalid format for the driving environments should be Suite_Town")

    # This is the folder creation of the
    create_log_folder(args.folder)
    erase_logs(args.folder)

    # The definition of parameters for driving
    drive_params = {
        "suppress_output": True,
        "no_screen": args.no_screen,
        "docker": args.docker,
        "record_collisions": args.record_collisions,
        "host": "127.0.0.1",
        "weather": args.weather,
        "traffic": args.traffic,
        "ep_number": args.episode_number,
        "base_folder": args.base_folder
    }
    ####
    # MODE 1: Single Process. Just execute a single experiment alias.
    ####

    if args.exp is None:
        raise ValueError(" You should set the exp alias when using single process")

    create_exp_path(args.folder, args.exp)

    drive_params['suppress_output'] = False
    # execute_drive(args.gpus, args.folder, args.exp, list(args.driving_environments)[0], drive_params, args.save_images)

    checkpoint_number = args.checkpoint

    create_exp_path(args.folder, args.exp)
    p = multiprocessing.Process(target=run_drive_custom.execute,
                                args=(args.gpus, args.base_folder, args.folder, args.exp, checkpoint_number, list(args.driving_environments)[0],
                                      drive_params, args.save_images, args.save_features))

    p.start()




