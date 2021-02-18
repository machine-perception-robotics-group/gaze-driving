from . import loss_functional as LF
import torch

import torch.nn.functional as F

def l1(params, use_speed=False):
    if not use_speed:
        return branched_loss(LF.l1_loss, params)
    elif use_speed:
        return branched_loss(LF.l1_loss_wo_vel, params, use_speed)


def l2(params, use_speed=False):
    if not use_speed:
        return branched_loss(LF.l2_loss, params)
    elif use_speed:
        return branched_loss(LF.l2_loss_wo_vel, params, use_speed)


def ml2(params):
    """
    params: all the parameters, including
            main: main task's feature maps
            target: sub-task's feature maps including several layers
            weight: coefficient balancing mimicking loss
    """
    criterion = torch.nn.MSELoss()
    loss = 0.
    for i in range(len(params['target'])):
        b, c, h, w = params['target'][i].size()
        features = F.interpolate(params['main'][i], size=(h, w), mode='bilinear', align_corners=False)
        loss += criterion(features, params['target'][i]) * params['weight'][i]

    return loss

def l2p(params):
    return branched_loss_plane(LF.l2_loss_wo_vel, params)



def branched_loss(loss_function, params, use_speed=False):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    controls_mask = LF.compute_branches_masks(params['controls'],
                                              params['branches'][0].shape[1])
    # Update the dictionary to add also the controls mask.
    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to  have 4 branches not using speed.

    for i in range(4): ## 提案: len(params['branches'])-1じゃいかんのか？
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] * params['variable_weights']['Steer'] \
                               + loss_branches_vec[i][:, 1] * params['variable_weights']['Gas'] \
                               + loss_branches_vec[i][:, 2] * params['variable_weights']['Brake']

    loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    if use_speed: ## for attention branch
        return torch.sum(loss_function) / (params['branches'][0].shape[0]),\
           plotable_params

    ## for others (calculate the speed loss)
    speed_loss = loss_branches_vec[4]/(params['branches'][0].shape[0])

    return torch.sum(loss_function) / (params['branches'][0].shape[0])\
                + torch.sum(speed_loss) / (params['branches'][0].shape[0]),\
           plotable_params


def branched_loss_plane(loss_function, params):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                variable weights: The weights for each of the variables used
                gaze_norm: gazemapの正規化するかどうか

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """
    ## gaze_mapをattmapと同サイズにするハードコード
    b, c, h, w = params['branches'][0].shape

    gazemap = params['targets'].detach()
    ### バッチ単位での正規化だけど…forは速度に問題が…とりあえずで．
    if params["gaze_norm"]:
        gazemap = gazemap / gazemap.max()
    gazemap = F.interpolate(gazemap, size=(h, w), mode='bilinear', align_corners=False)
    params['targets'] = gazemap
    ##
    controls_mask = LF.compute_branches_masks(params['controls'],
                                              params['branches'][0].shape[1])

    ## control maskのshapeをここで変える
    for i in range(4):
        controls_mask[i] = controls_mask[i].view(b, 1, 1, 1)
    # Update the dictionary to add also the controls mask.

    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    for i in range(4):
        loss_branches_vec[i] = torch.sum(loss_branches_vec[i])

    loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    return (loss_function / (params['branches'][0].shape[0])) * params['variable_weights'],\
           plotable_params


def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if loss_name == 'L1':

        return l1

    elif loss_name == 'L2':

        return l2

    elif loss_name == 'mimick_L2':

        return ml2

    elif loss_name == 'L2_plane':

        return l2p

    else:
        raise ValueError(" Not found Loss name")


