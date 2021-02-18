import os
import sys
import random
import time
import traceback
import torch
import torch.optim as optim

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto, get_gaze_model #, GazeModel, GazeModelMN
from input import CoILDataset, Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, \
                                    check_loss_validation_stopped

# Seed固定用
import numpy as np

# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12):
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
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        set_type_of_process('train')
        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': gpu})

        # Put the output to a separate file if it is the case

        if suppress_output:
            if not os.path.exists('_output_logs'):
                os.mkdir('_output_logs')
            sys.stdout = open(os.path.join('_output_logs', exp_alias + '_' +
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a",
                              buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_'+g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            checkpoint = torch.load(os.path.join('_logs', g_conf.PRELOAD_MODEL_BATCH,
                                                  g_conf.PRELOAD_MODEL_ALIAS,
                                                 'checkpoints',
                                                 str(g_conf.PRELOAD_MODEL_CHECKPOINT)+'.pth'))


        #### モデルのパラメータ初期化をシード固定で行う（それ以前の処理に，学習結果に関わるランダム性がないことを前提にここに置いてます）
        random.seed(g_conf.MAGICAL_SEED)
        np.random.seed(g_conf.MAGICAL_SEED)
        torch.manual_seed(g_conf.MAGICAL_SEED)
        torch.cuda.manual_seed_all(g_conf.MAGICAL_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Fixed random seed (config:torch) %d : %d" % (g_conf.MAGICAL_SEED, torch.initial_seed()))
        ####

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file is not None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                    'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0


        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        # Instantiate the class used to read a dataset. The coil dataset generator
        # can be found
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                               + 'hours_' + g_conf.TRAIN_DATASET_NAME)
        print ("Loaded dataset")

        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)
        ## 変更
        if g_conf.MODEL_TYPE == "coil-gaze":
            gaze_model = get_gaze_model(g_conf.GAZE_MODEL_TYPE, g_conf.GAZE_CHECKPOINT, exp_batch, exp_alias, "train")
            ## get loss function for feature mimicking.
            if g_conf.MODEL_CONFIGURATION['perception']['res_g']['gtype'] == 'mimick':
                mimick_criterion = Loss('mimick_L2')
            # gaze_model = GazeModelMN(exp_batch, exp_alias, "train")
        ## GazeLossモデル用のGazeモデル
        elif g_conf.MODEL_TYPE == 'coil-gaze-loss':
            gaze_model = get_gaze_model(g_conf.GAZE_MODEL_TYPE, g_conf.GAZE_CHECKPOINT, exp_batch, exp_alias, "train")
        ## ABN+視線用の視線モデル
        elif g_conf.MODEL_TYPE == "abn-drive":
            if 'gmask' in g_conf.MODEL_CONFIGURATION or 'gloss' in g_conf.MODEL_CONFIGURATION:
                gaze_model = get_gaze_model(g_conf.GAZE_MODEL_TYPE, g_conf.GAZE_CHECKPOINT, exp_batch, exp_alias, "train")
                use_gaze_for_abn = True
            else:
                use_gaze_for_abn = False

        
        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)

        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accumulated_time = checkpoint['total_time']
            loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            loss_window = []

        print ("Before the loss")

        criterion = Loss(g_conf.LOSS_FUNCTION)
        # attention map 用のloss
        if g_conf.MODEL_TYPE == 'abn-drive' and 'gloss' in g_conf.MODEL_CONFIGURATION:
            criterion_att = Loss(g_conf.MODEL_CONFIGURATION['gloss']['loss'])
        # gaze cil用のloss
        if g_conf.MODEL_TYPE == 'coil-gaze-loss':
            ## loss計算箇所が定義されているか確認
            assert len(g_conf.MODEL_CONFIGURATION['perception']['res']['gpos']) > 0
            assert len(g_conf.MODEL_CONFIGURATION['perception']['res']['gpos']) == \
                   len(g_conf.MODEL_CONFIGURATION['perception']['res']['weight'])
            criterion_gloss = Loss(g_conf.MODEL_CONFIGURATION['perception']['res']['loss'])

        # Loss time series window
        for data in data_loader:

            # Basically in this mode of execution, we validate every X Steps, if it goes up 3 times,
            # add a stop on the _logs folder that is going to be read by this process
            if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                    check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                break
            """
                ####################################
                    Main optimization loop
                ####################################
            """

            iteration += 1
            if iteration % 1000 == 0:
                adjust_learning_rate_auto(optimizer, loss_window)

            # get the control commands from float_data, size = [120,1]

            capture_time = time.time()
            controls = data['directions']

            ## 追加: total loss variable
            total_loss = 0.

            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
            model.zero_grad()
            if g_conf.MODEL_TYPE == "coil-icra" or g_conf.MODEL_TYPE == "normal-drive":
                branches = model(torch.squeeze(data['rgb'].cuda()),
                             dataset.extract_inputs(data).cuda())
            
            ###### Gaze-CIL (Mask ver.) ######
            elif g_conf.MODEL_TYPE == "coil-gaze":
                gaze_map = gaze_model.run_step(torch.squeeze(data['rgb'].cuda()),
                             controls.cuda())
                ## option
                if 'option' in g_conf.MODEL_CONFIGURATION['perception']['res_g']:
                    if 'norm' in g_conf.MODEL_CONFIGURATION['perception']['res_g']['option']:
                        gaze_map = torch.div(gaze_map, gaze_map.max(3, keepdim=True)[0].max(2, keepdim=True)[0])
                branches = model(torch.squeeze(data['rgb'].cuda()),
                             dataset.extract_inputs(data).cuda(),
                             gaze_map)

            ###### Gaze-CIL (Loss ver.) ######
            elif g_conf.MODEL_TYPE == 'coil-gaze-loss':
                gaze_map = gaze_model.run_step(torch.squeeze(data['rgb'].cuda()),
                             controls.cuda())
                branches = model(torch.squeeze(data['rgb'].cuda()),
                             dataset.extract_inputs(data).cuda())
                ## Additional Loss (gaze loss)
                gloss_function_params = {
                    'main': [model.inter[i] for i in g_conf.MODEL_CONFIGURATION['perception']['res']['gpos']],
                    'target': gaze_map,
                    'variable_weight': g_conf.MODEL_CONFIGURATION['perception']['res']['weight']
                }
                g_loss, plot_feats = criterion_gloss(gloss_function_params)
                total_loss += g_loss

            ###### ABN-CIL ######
            elif g_conf.MODEL_TYPE == "abn-drive":
                if use_gaze_for_abn:
                    gaze_map = gaze_model.run_step(torch.squeeze(data['rgb'].cuda()),
                                 controls.cuda())
                ##### 普通のabnの場合にerrorなるかも？(~~before assignment)その場合は下のバージョンを使う
                items = [torch.squeeze(data['rgb'].cuda()), dataset.extract_inputs(data).cuda(), gaze_map] \
                        if 'gmask' in g_conf.MODEL_CONFIGURATION else \
                        [torch.squeeze(data['rgb'].cuda()), dataset.extract_inputs(data).cuda()]
                branches = model(*items)
                ##
                ab_branches = branches[4:8] # attention branchの制御値出力
                attention_maps = branches[8:12] ##
                branches = branches[0:4] + [branches[-1]] ## perception branch の出力と速度推定出力
                # attention_mapsの損失計算用
                if 'gloss' in g_conf.MODEL_CONFIGURATION:
                    # attention_maps = branches[8:12] ##
                    attmap_loss_function_params = {
                        'branches': attention_maps,
                        'targets': gaze_map,
                        'controls': controls.cuda(),
                        'branch_weights': g_conf.BRANCH_LOSS_WEIGHT[:-1],# speedを除く
                        'variable_weights': g_conf.MODEL_CONFIGURATION['gloss']['loss_weight'],
                        'gaze_norm': g_conf.MODEL_CONFIGURATION['gloss']['norm']
                    }
                    am_loss, _ = criterion_att(attmap_loss_function_params)
                    total_loss += am_loss
                # ab_branchesの損失計算用
                loss_function_params = {
                    'branches': ab_branches,
                    'targets': dataset.extract_targets(data).cuda(),
                    'controls': controls.cuda(),
                    'inputs': dataset.extract_inputs(data).cuda(),
                    'branch_weights': g_conf.BRANCH_LOSS_WEIGHT[:-1],# speedを除く
                    'variable_weights': g_conf.VARIABLE_WEIGHT
                }
                ab_loss, _ = criterion(loss_function_params, True)# フラグを立ててattention branch用に
                total_loss += ab_loss
                
            loss_function_params = {
                'branches': branches,
                'targets': dataset.extract_targets(data).cuda(),
                'controls': controls.cuda(),
                'inputs': dataset.extract_inputs(data).cuda(),
                'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                'variable_weights': g_conf.VARIABLE_WEIGHT
            }
            loss, _ = criterion(loss_function_params)
            # loss.backward()
            # optimizer.step()
            # 追加
            total_loss += loss
            total_loss.backward()
            optimizer.step()
            """
                ####################################
                    Saving the model if necessary
                ####################################
            """

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter
                }
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'checkpoints', str(iteration) + '.pth'))

            """
                ################################################
                    Adding tensorboard logs.
                    Making calculations for logging purposes.
                    These logs are monitored by the printer module.
                #################################################
            """
            coil_logger.add_scalar('Loss', loss.data, iteration)
            ## todo: mimickを今後使うことがあるならmimick用のadd_scalar文を追加してもよし
            ### GazeLossの追加
            if g_conf.MODEL_TYPE == "coil-gaze-loss":
                coil_logger.add_scalar('GazeLoss', g_loss.data, iteration)
                #### tensorboardに画像も表示したい件
                coil_logger.add_image('Gaze', plot_feats[0], iteration)
                for i in range(1, len(plot_feats)):
                    coil_logger.add_image('Features%d'%i, plot_feats[i], iteration)
            ### ABNの場合はABのLossも追加
            if g_conf.MODEL_TYPE == "abn-drive":
                coil_logger.add_scalar('ABLoss', ab_loss.data, iteration)
            coil_logger.add_image('Image', torch.squeeze(data['rgb']), iteration)
            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, len(data) - 1)

            output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(data).cuda())

            accumulated_time += time.time() - capture_time

            coil_logger.add_message('Iterating',
                                    {'Iteration': iteration,
                                     'Loss': loss.data.tolist(),
                                     'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
                                     'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                     'Output': output[position].data.tolist(),
                                     'GroundTruth': dataset.extract_targets(data)[
                                         position].data.tolist(),
                                     'Error': error[position].data.tolist(),
                                     'Inputs': dataset.extract_inputs(data)[
                                         position].data.tolist()},
                                    iteration)
            loss_window.append(loss.data.tolist())
            coil_logger.write_on_error_csv('train', loss.data)
            ## write gaze loss on csv
            if g_conf.MODEL_TYPE == "coil-gaze-loss":
                coil_logger.write_on_error_csv('train_gloss', g_loss.data)
            ## write mimicking loss on csv
            if g_conf.MODEL_TYPE == "coil-gaze":
                if g_conf.MODEL_CONFIGURATION['perception']['res_g']['gtype'] == 'mimick':
                    coil_logger.write_on_error_csv('train_mimick', mimick_loss.data)
            ## write control loss for attention branch on csv (and loss for attention map)
            if g_conf.MODEL_TYPE == "abn-drive":
                coil_logger.write_on_error_csv('train_ab', ab_loss.data)
                if 'gloss' in g_conf.MODEL_CONFIGURATION:
                    coil_logger.write_on_error_csv('train_am', am_loss.data)
            print("Iteration: %d  Loss: %f" % (iteration, loss.data))

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except RuntimeError as e:

        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
