import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


######################################
##### 通常の視線導入ResNet (gaze1) #####
######################################

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=0)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, g):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        b, c, h, w = x1.size() # 
        g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
        x1 = (x1 * g) + x1 # 視線情報付与

        x2 = self.layer2(x1)
        b, c, h, w = x2.size() # 
        g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
        x2 = (x2 * g) + x2 # 視線情報付与

        x3 = self.layer3(x2)
        b, c, h, w = x3.size() # 
        g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
        x3 = (x3 * g) + x3 # 視線情報付与

        x4 = self.layer4(x3)
        b, c, h, w = x4.size() # 
        g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
        x4 = (x4 * g) + x4 # 視線情報付与

        x = self.avgpool(x4) # 

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x, g):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        b, c, h, w = x.size()
        g = F.interpolate(g, size=(h, w), mode='bicubic', align_corners=False)
        x6 = (x5 * g) + x5 # 視線情報付与
        x = x6.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers


######################################
##### 入力箇所複数試す用 (gaze1ex)　#####
######################################

class ResNet_ex(nn.Module):

    def __init__(self, block, layers, num_classes=1000, extype=None):
        self.inplanes = 64
        super(ResNet_ex, self).__init__()
        ## 追加：視線追加位置の指定用変数
        # 設定忘れ回避
        if not extype in ['A', 'B', 'C', 'D']:
            raise ValueError("Invalid letter: extype must be 'A', 'B', 'C', or 'D'.")
        print("...setting resnet structure, EXTYPE: %s" % extype)
        self.extype = extype

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=0)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, g):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        if self.extype == 'A':
            b, c, h, w = x1.size() # 
            g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
            x1 = (x1 * g) + x1 # 視線情報付与

        x2 = self.layer2(x1)
        if self.extype == 'B':
            b, c, h, w = x2.size() # 
            g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
            x2 = (x2 * g) + x2 # 視線情報付与

        x3 = self.layer3(x2)
        if self.extype == 'C':
            b, c, h, w = x3.size() # 
            g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
            x3 = (x3 * g) + x3 # 視線情報付与

        x4 = self.layer4(x3)
        if self.extype == 'D':
            b, c, h, w = x4.size() # 
            g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False)
            x4 = (x4 * g) + x4 # 視線情報付与

        x = self.avgpool(x4) # 

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x, g):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        b, c, h, w = x.size()
        g = F.interpolate(g, size=(h, w), mode='bicubic', align_corners=False)
        x6 = (x5 * g) + x5 # 視線情報付与
        x = x6.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers

######################################
######################################
######################################




def make_resnet(block, layers, gtype, **kwargs):
    # 必要なResNetを選択して返す
    if gtype=='gaze1':
        return ResNet(block, layers, **kwargs)
    elif gtype=='gaze1ex':
        return ResNet_ex(block, layers, **kwargs)

