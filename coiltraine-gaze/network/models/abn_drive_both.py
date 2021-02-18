
from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F

import importlib

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join

"""
簡易実装のABN．細かい設定はNoCrashの設定に合わせてハードコード済み
ToDo: 拡張性を考えて場合はCoILICRAベースに作り直す
"""


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)


class WGP(nn.Module):
    def __init__(self, csize, ksize):
        super(WGP, self).__init__()
        wgp = []
        for i in range(csize):
            p = nn.Sequential(
                    nn.Conv2d(1, 1, ksize, 1),
                    nn.Tanh()
                )
            wgp.append(p)
        self.wgp = nn.ModuleList(wgp)

    def forward(self, x):
        b, c, h, w = x.shape
        ans = []
        for i in range(c):
            ans.append(self.wgp[i](x[:,i,:,:].view(b, 1, h, w)).view(b))
        result = torch.stack(ans, dim=1)

        # # catでやる場合
        # result = self.wgp[0](x[:,0,:,:].view(b, 1, h, w)).view(b)
        # for i in range(1,c):
        #     ans = self.wgp[i](x[:,i,:,:].view(b, 1, h, w)).view(b,1)
        #     result = torch.cat((result, ans), dim=1)

        return result


class AttentionBranch(nn.Module):
    def __init__(self, params, n_shape, n_out):
        super(AttentionBranch, self).__init__()
        ### ハードコーディング ToDo: YAMLファイルから設計できるようにする（conv.pyを改良して使用）
        b,c,h,w = n_shape
        self.convs = nn.Sequential(
                    nn.Conv2d(c, 500, 1, 1), # 元のabnでは96,1000,100,classだったが変更 -> 96,500,100,class
                    nn.BatchNorm2d(500),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(500, 100, 1, 1),
                    nn.BatchNorm2d(100),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(100, n_out, 1, 1),
                    nn.ReLU(inplace=True)
            )
        self.att_mask = nn.Sequential(
                nn.Conv2d(n_out, 1, 1),
                nn.Sigmoid()
            )
        n_size, n_shape = get_conv_output(self.convs, (c,h,w))
        #self.wgp = WGP(n_out, (4, 18))
        self.wgp = WGP(n_out, n_shape[2:])

    def forward(self, x):
        h = self.convs(x)
        attention_map = self.att_mask(h)
        outputs = self.wgp(h)

        return attention_map, outputs

class PerceptionBranch(nn.Module):
    def __init__(self, params, n_size, out_size):
        super(PerceptionBranch, self).__init__()
        # if resnet:
        #     ### fcの入力サイズはハードコーディング
        #     self.fc = nn.Sequential(
        #                         nn.AvgPool2d(2, stride=0),
        #                         FC(params={'neurons': [1536] + 
        #                                      params['extractor']['fc']['neurons'],
        #                                    'dropouts': params['extractor']['fc']['dropouts'],
        #                                    'end_layer': True}) ## なぜかCILRSのこの部分には活性化関数がなかったので一応合わせるためTrue
        #                 )
        # else:
        self.fc = FC(params={'neurons': [n_size] + 
                                         params['extractor']['fc']['neurons'],
                                       'dropouts': params['extractor']['fc']['dropouts'],
                                       'end_layer': True}) ## なぜかCILRSのこの部分には活性化関数がなかったので一応合わせるためTrue

        self.join = Join(
                params={'after_process':
                        FC(params={'neurons':
                                    [params['measurements']['fc']['neurons'][-1] +
                                     out_size] +
                                    params['join']['fc']['neurons'],
                                 'dropouts': params['join']['fc']['dropouts'],
                                 'end_layer': False}),
                        'mode': 'cat'
                    }
            )

        self.out_fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.Dropout2d(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(256, len(g_conf.TARGETS)) # 出力層にactivationはない模様
            )

    def forward(self, x, m):
        x = self.fc(x)
        j = self.join(x, m)
        outputs = self.out_fc(j)

        return outputs


#### Attention branch用のbranchクラス．PerceptionBranchクラスと紛らわしいから注意
class AttentionBranching(nn.Module):

    def __init__(self, branched_modules=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """
        # TODO: Make an auto naming function for this.

        super(AttentionBranching, self).__init__()

        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)




    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):
        # get only the speeds from measurement labels



        # TODO: we could easily place this speed outside

        branches_attnmap = []
        branches_outputs = []
        for branch in self.branched_modules:
            attnmap, outputs = branch(x)
            branches_attnmap.append(attnmap)
            branches_outputs.append(outputs)

        return branches_attnmap, branches_outputs


    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        coil_logger.add_message('Loading', {
                    "Model": {"Loaded checkpoint: " + str(checkpoint) }

                })



#### Perception branch用のbranchクラス．PerceptionBranchクラスと紛らわしいから注意
class PerceptionBranching(nn.Module):

    def __init__(self, branched_modules=None, pool=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """
        # TODO: Make an auto naming function for this.

        super(PerceptionBranching, self).__init__()

        # self.resnet = resnet ## flag
        ## ave_poolは重みパラがないのでモジュールにしない
        if pool is not None: # poolにはカーネルサイズを格納; Noneの場合はresnetで行う
            self.pooling = nn.Sequential(nn.AvgPool2d(pool, stride=0) for i in range(4))
        else:
            self.pooling = nn.Sequential()

            # ### branch数をハードコーディング
            # pooling_modules = [nn.AvgPool2d(2, stride=0) for i in range(4)]
            # self.pooling_modules = nn.ModuleList(pooling_modules)


        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)

    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, m, attn):
        # get only the speeds from measurement labels

        branches_outputs = []
        for i, branch in enumerate(self.branched_modules):
            fvec = x * attn[i]
            fvec = self.pooling(fvec) # pooling (入っていれば)
            b = fvec.shape[0]
            fvec = fvec.view(b, -1)
            branches_outputs.append(branch(fvec, m))

        return branches_outputs


    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .
        """
        coil_logger.add_message('Loading', {
                    "Model": {"Loaded checkpoint: " + str(checkpoint) }

                })



class CoABNDrive(nn.Module):
    def __init__(self, params):
        super(CoABNDrive, self).__init__()

        n_out = len(g_conf.TARGETS) # 出力の数

        # 
        number_first_layer_channels = 0
        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        resnet = 'resnet' in params['extractor']

        if 'conv' in params['extractor']:
            # Convolutional Layers
            self.extractor = nn.Sequential( # in-place: Trueならコピーを作らず直接変数を書き換える
                    nn.Conv2d(3, 24, 5, 2),
                    nn.BatchNorm2d(24),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(24, 36, 5, 2),
                    nn.BatchNorm2d(36),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(36, 48, 5, 2),
                    nn.BatchNorm2d(48),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(48, 64, 3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 96, 3),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True)
                )
        elif 'res' in params['extractor']:  # pre defined residual networks
            if 'normal' == params['extractor']['res']['type']:
                resnet_module = importlib.import_module('network.models.building_blocks.resnet_abn')
                resnet_module = getattr(resnet_module, params['extractor']['res']['name'])
                self.extractor = resnet_module(pretrained=g_conf.PRE_TRAINED)
            elif 'cifar' == params['extractor']['res']['type']:
                pooling = params['extractor']['res']['pooling']# if params['extractor']['res']['pool_before'] else None
                resnet_module = importlib.import_module('network.models.building_blocks.resnet_cifar')
                resnet_module = getattr(resnet_module, params['extractor']['res']['name'])
                self.extractor = resnet_module(pretrained=g_conf.PRE_TRAINED, pool=pooling) #poolがnoneなら2x2のpoolingを追加
            else:
                raise ValueError("invalid resnet type")
        else:

            raise ValueError("invalid convolution layer type")


        # Fully Connected Layers
        n_size, n_shape = get_conv_output(self.extractor, sensor_input_shape)
        if resnet:
            if 'normal' == params['extractor']['res']['type']:
                n_size = 1536 # ハードコーディング
        print(n_size, n_shape)
        

        ### Attention branch
        ##### 技術的な問題点：学習時に，conditional attention map をバッチ毎に…(ry => 変更
        branch_ab_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_ab_vector.append(
                    AttentionBranch(params, n_shape, n_out)
                )
        self.attention_branches = AttentionBranching(branch_ab_vector)


        ### 元のコピー
        self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] +
                                                   params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})


        # number_output_neurons = params['perception']['fc']['neurons'][-1]
        number_output_neurons = 512 ## ハードコーディング

        # CILRSと構造・位置を変更
        if resnet:
            self.speed_branch = nn.Sequential(
                                nn.AvgPool2d(2, stride=0),
                                FC(params={'neurons': [n_size] + 
                                                      params['speed_branch']['fc']['neurons'] + [1],
                                           'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                           'end_layer': True})
                            )
        else:
            self.speed_branch = FC(params={'neurons': [n_size] + 
                                                      params['speed_branch']['fc']['neurons'] + [1],
                                           'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                           'end_layer': True})
        ###

        ### Perception branch ## Joinをbranch用Sequentialにぶちこむ（動くかな？）
        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(PerceptionBranch(params, n_size, number_output_neurons))

        ## poolingの位置を設定
        pooling = None
        if resnet:
            # normal resnetの場合のみ
            if 'normal' in params['extractor']['res']['type']:
                pooling = 2

        self.perception_branches = PerceptionBranching(branch_fc_vector, pooling)

        ## initialization
        if 'conv' in params['extractor']:
            self.extractor.apply(init_weights)
            self.attention_branches.apply(init_weights)
            self.measurements.apply(init_weights)
            self.speed_branch.apply(init_weights)
            self.perception_branches.apply(init_weights)
        else:
            self.attention_branches.apply(init_weights)
            self.measurements.apply(init_weights)
            self.speed_branch.apply(init_weights)
            self.perception_branches.apply(init_weights)

        ## 最初にローカル変数を用意してるが，面倒だから別で用意
        self.use_resnet = 'res' in params['extractor']


    def forward(self, x, a):
        # Convolutional Layers
        if self.use_resnet:
            x, self.inter = self.extractor(x)
        else:
            self.inter = []
            #### スーパーハードコーディング
            for i in range(5):
                x = self.extractor[i*3:i*3+3](x)
                self.inter.append(x)
        # x = self.extractor(x)

        b = x.shape[0]
        x_vec = x.view(b, -1)
        
        ## Attention branch
        att_map, ab_outputs = self.attention_branches(x)

        # b,c,y,x = x.shape
        # h = x.view(b, -1)
        # h = self.fc(h)
        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        # """ Join measurements and perception"""
        # j = self.join(h, m)

        branch_outputs = self.perception_branches(x, m, att_map)

        """Speed Regularization"""
        speed_branch_output = self.speed_branch(x_vec)

        # We concatenate speed with the rest.
        return branch_outputs + ab_outputs + att_map + [speed_branch_output]


    def forward_branch(self, x, a, branch_number):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            branch_number: the branch number to be returned

        Returns:
            the forward operation on the selected branch

        """
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        
        ## 0-3: pbの出力, 4-7: abの出力, 8-11: attention map, 12: 推定速度
        outputs = self.forward(x, a)
        output_att = torch.stack(outputs[8:12])
        output_vec = torch.stack(outputs[0:4])
        # output_vec = torch.stack(self.forward(x, a)[0:4])

        return self.extract_branch(output_vec, branch_number), self.extract_branch(output_att, branch_number), outputs[-1]


    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :] ### branch_number[0]:バッチ毎の選択ブランチ番号 branch_number[1]:0から順に制御値をとる



def get_conv_output(func, shape):
    """
       By inputing the shape of the input, simulate what is the ouputsize.
    """

    bs = 1
    input = torch.autograd.Variable(torch.rand(bs, *shape))
    output_feat = func.forward(input)
    if type(output_feat) == tuple:
        output_feat, _ = output_feat
    # output_feat = func.forward(input)
    n_shape = output_feat.data.shape
    n_size = output_feat.data.view(bs, -1).size(1)
    return n_size, n_shape





# AB無し

## ResNetを使った処理の記述がABNDriveと異なるため注意（処理内容は変わらないはず）

# AB無しのPerceptin branch部分
class NormalBranch(nn.Module):
    def __init__(self, params, n_size, out_size):
        super(NormalBranch, self).__init__()
        self.fc = FC(params={'neurons': [n_size] + 
                                         params['extractor']['fc']['neurons'],
                                       'dropouts': params['extractor']['fc']['dropouts'],
                                       'end_layer': True}) ## なぜかCILRSのこの部分には活性化関数がなかったので一応合わせるためTrue

        self.join = Join(
                params={'after_process':
                        FC(params={'neurons':
                                    [params['measurements']['fc']['neurons'][-1] +
                                     out_size] +
                                    params['join']['fc']['neurons'],
                                 'dropouts': params['join']['fc']['dropouts'],
                                 'end_layer': False}),
                        'mode': 'cat'
                    }
            )

        self.out_fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.Dropout2d(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(256, len(g_conf.TARGETS)) # 出力層にactivationはない模様
            )

    def forward(self, x):#, m):
        x, m = x
        x = self.fc(x)
        j = self.join(x, m)
        outputs = self.out_fc(j)

        return outputs



class NormalDrive(nn.Module):
    def __init__(self, params):
        super(NormalDrive, self).__init__()

        n_out = len(g_conf.TARGETS) # 出力の数

        # 
        number_first_layer_channels = 0
        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        # Convolutional Layers
        # resnet = 'resnet' in params['extractor']
        if 'conv' in params['extractor']:
            # Convolutional Layers
            self.extractor = nn.Sequential( # in-place: Trueならコピーを作らず直接変数を書き換える
                    nn.Conv2d(3, 24, 5, 2),
                    nn.BatchNorm2d(24),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(24, 36, 5, 2),
                    nn.BatchNorm2d(36),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(36, 48, 5, 2),
                    nn.BatchNorm2d(48),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(48, 64, 3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 96, 3),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True)
                )
        elif 'res' in params['extractor']:  # pre defined residual networks
            resnet_module = importlib.import_module('network.models.building_blocks.resnet_abn')
            resnet_module = getattr(resnet_module, params['extractor']['res']['name'])
            resnet_extractor = resnet_module(pretrained=g_conf.PRE_TRAINED)
            self.extractor = nn.Sequential(
                    resnet_extractor,
                    nn.AvgPool2d(2, stride=0) ## Poolingに学習パラメータはないはずなので…ブランチ分けせず．
                )
        else:

            raise ValueError("invalid convolution layer type")


        ### 元のコピー
        self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] +
                                                   params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})


        # Fully Connected Layers
        n_size, n_shape = get_conv_output(self.extractor, sensor_input_shape)
        print(n_size, n_shape)

        # number_output_neurons = params['perception']['fc']['neurons'][-1]
        number_output_neurons = 512 ## ハードコーディング

        # CILRSと構造・位置を変更
        self.speed_branch = FC(params={'neurons': [n_size] + 
                                                  params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})
        ###

        ### Perception branch ## Joinをbranch用Sequentialにぶちこむ（動くかな？）
        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(NormalBranch(params, n_size, number_output_neurons))

        self.perception_branches = Branching(branch_fc_vector)


        ## initialization
        if 'conv' in params['extractor']:
            self.extractor.apply(init_weights)
            self.measurements.apply(init_weights)
            self.speed_branch.apply(init_weights)
            self.perception_branches.apply(init_weights)
        else:
            self.measurements.apply(init_weights)
            self.speed_branch.apply(init_weights)
            self.perception_branches.apply(init_weights)

        ## 最初にローカル変数を用意してるが，面倒だから別で用意
        self.use_resnet = 'res' in params['extractor']


    def forward(self, x, a):
        # Convolutional Layers
        if self.use_resnet:
            x, self.inter = self.extractor(x)
        else:
            self.inter = []
            #### スーパーハードコーディング
            for i in range(5):
                x = self.extractor[i*3:i*3+3](x)
                self.inter.append(x)
        # x = self.extractor(x)

        b = x.shape[0]
        x_vec = x.view(b, -1)
        
        # ## Attention branch
        # att_map, ab_outputs = self.attention_branches(x)

        # b,c,y,x = x.shape
        # h = x.view(b, -1)
        # h = self.fc(h)
        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        # """ Join measurements and perception"""

        branch_outputs = self.perception_branches([x_vec, m])

        """Speed Regularization"""
        speed_branch_output = self.speed_branch(x_vec)

        # We concatenate speed with the rest.
        return branch_outputs + [speed_branch_output]


    def forward_branch(self, x, a, branch_number):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            branch_number: the branch number to be returned

        Returns:
            the forward operation on the selected branch

        """
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        
        ## 0-3: pbの出力, 4-7: abの出力, 8-11: attention map, 12: 推定速度
        outputs = self.forward(x, a)
        output_vec = torch.stack(self.forward(x, a)[0:4])

        return self.extract_branch(output_vec, branch_number), outputs[-1]


    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :] ### branch_number[0]:バッチ毎の選択ブランチ番号 branch_number[1]:0から順に制御値をとる


