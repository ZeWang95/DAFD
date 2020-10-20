import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
from torch.autograd import Variable

from torch.utils.data import DataLoader
import torch
from torch import nn

import torch.nn.functional as F
import pdb
import cv2

import time

from torch.nn.parameter import Parameter
import math
from fb import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}



class Conv_DCF(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
        num_bases=6, bias=True,  base_grad=False, initializer='FB'):
        super(Conv_DCF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.edge = (kernel_size-1)/2
        self.stride = stride
        self.padding = padding
        self.kernel_list = {}
        self.num_bases = num_bases


        assert initializer in ['FB', 'random'], 'Initializer should be either FB or random, other methods are not implemented yet'

        if initializer == 'FB':
            if kernel_size % 2 == 0:
                raise Exception('Kernel size for FB initialization only supports odd number for now.')
            base_np, _, _ = calculate_FB_bases(int((kernel_size-1)/2))
            if num_bases > base_np.shape[1]:
                raise Exception('The maximum number of bases for kernel size = %d is %d' %(kernel_size, base_np.shape[1]))
            base_np = base_np[:, :num_bases]
            base_np = base_np.reshape(kernel_size, kernel_size, num_bases)
            base_np = np.expand_dims(base_np.transpose(2,0,1), 1)

        else:
            base_np = np.random.random((num_bases, 1, kernel_size, kernel_size))-0.5

        self.bases = Parameter(torch.Tensor(base_np), requires_grad=base_grad)


        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels*num_bases, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        FE_SIZE = input.size()
        feature_list = []
        input = input.view(FE_SIZE[0]*FE_SIZE[1], 1, FE_SIZE[2], FE_SIZE[3])
        

        feature = F.conv2d(input, self.bases,
            None, self.stride, self.padding, dilation=1)

        feature = feature.view(
            FE_SIZE[0], FE_SIZE[1]*self.num_bases, 
            int((FE_SIZE[2]-2*self.edge+2*self.padding)/self.stride), 
            int((FE_SIZE[3]-2*self.edge+2*self.padding)/self.stride))

        feature_out = F.conv2d(feature, self.weight, self.bias, 1, 0)

        return feature_out


class Conv_DCF_db(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
        num_bases=6, bias=True, base_grad0=True, base_grad1=True, initializer='FB'):
        super(Conv_DCF_db, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.edge = (kernel_size-1)/2
        self.stride = stride
        self.padding = padding
        self.kernel_list = {}
        self.num_bases = num_bases


        assert initializer in ['FB', 'random', 'db_rand'], 'Initializer should be either FB or random, other methods are not implemented yet'

        if initializer == 'FB':
            if kernel_size % 2 == 0:
                raise Exception('Kernel size for FB initialization only supports odd number for now.')
            base_np, _, _ = calculate_FB_bases(int((kernel_size-1)/2))
            if num_bases > base_np.shape[1]:
                raise Exception('The maximum number of bases for kernel size = %d is %d' %(kernel_size, base_np.shape[1]))
            base_np = base_np[:, :num_bases]
            base_np = base_np.reshape(kernel_size, kernel_size, num_bases)
            base_np = np.expand_dims(base_np.transpose(2,0,1), 1)
            base_np0 = base_np
            base_np1 = base_np

        elif initializer == 'random':
            base_np = np.random.random((num_bases, 1, kernel_size, kernel_size))-0.5
            base_np0 = base_np
            base_np1 = base_np
        else:
            base_np0 = np.random.random((num_bases, 1, kernel_size, kernel_size))-0.5
            base_np1 = np.random.random((num_bases, 1, kernel_size, kernel_size))-0.5

        self.bases0 = Parameter(torch.Tensor(base_np0), requires_grad=base_grad0)
        self.bases1 = Parameter(torch.Tensor(base_np1), requires_grad=base_grad1)

        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels*num_bases, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # pdb.set_trace()
        batch_size = input.size(0)
        # if batch_size == 2:
        #     input0 = input[:int(batch_size/2)].unsqueeze(0)
        #     input1 = input[int(batch_size/2):].unsqueeze(0)
        # else:
        input0 = input[:int(batch_size/2)]
        input1 = input[int(batch_size/2):]
        FE_SIZE = input0.size()
        feature_list = []
        input0 = input0.view(FE_SIZE[0]*FE_SIZE[1], 1, FE_SIZE[2], FE_SIZE[3])
        input1 = input1.view(FE_SIZE[0]*FE_SIZE[1], 1, FE_SIZE[2], FE_SIZE[3])
        

        feature0 = F.conv2d(input0, self.bases0,
            None, self.stride, self.padding, dilation=1)
        feature1 = F.conv2d(input1, self.bases1,
            None, self.stride, self.padding, dilation=1)

        feature0 = feature0.view(
            FE_SIZE[0], FE_SIZE[1]*self.num_bases, 
            int((FE_SIZE[2]-2*self.edge+2*self.padding)/self.stride), 
            int((FE_SIZE[3]-2*self.edge+2*self.padding)/self.stride))

        feature1 = feature1.view(
            FE_SIZE[0], FE_SIZE[1]*self.num_bases, 
            int((FE_SIZE[2]-2*self.edge+2*self.padding)/self.stride), 
            int((FE_SIZE[3]-2*self.edge+2*self.padding)/self.stride))

        self.feature0 = feature0
        self.feature1 = feature1
        # pdb.set_trace()

        feature_all = torch.cat([feature0, feature1], 0)

        # feature_out0 = F.conv2d(feature0, self.weight, self.bias, 1, 0)
        # feature_out1 = F.conv2d(feature1, self.weight, self.bias, 1, 0)
        feature_out = F.conv2d(feature_all, self.weight, self.bias, 1, 0)

        # self.feature_out0 = feature_out0
        # self.feature_out1 = feature_out1

        # feature_out = torch.cat([feature_out0, feature_out1], 0)
        return feature_out

# NUM_BASES = 14
class LeNet(nn.Module):
    def __init__(self, num_bases=14):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv1 = Conv_DCF_db(3, 6, kernel_size=5, num_bases=num_bases)
        self.conv2 = Conv_DCF_db(6, 16, kernel_size=5, num_bases=num_bases)
        # self.conv1 = Conv_DCF_db(3, 6, kernel_size=5, num_bases=num_bases, initializer='db_rand')
        # self.conv2 = Conv_DCF_db(6, 16, kernel_size=5, num_bases=num_bases, initializer='db_rand')
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.conv_feat0 = x
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        self.conv_feat1 = x
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        self.feature = x
        x = self.fc3(x)
        return x



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(512, 1024)
        # self.linear = nn.Linear(512*7*7, 4096)
        # self.linear1 = nn.Linear(4096, 4096)
        # self.classifier = nn.Linear(1024, num_classes)
        self.classifier = nn.Linear(256, num_classes)
        self.drop0 = nn.Dropout(0.2)
        self.drop1 = nn.Dropout(0.2)

        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        

        self.feature = x
        # x = self.drop1(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

# def joint_max_pooling()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_dcf(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    cm = 0
    for v in cfg:
        if v == 'M':
            cm += 1
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # if cm < 2:
            conv2d = Conv_DCF(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_dcf_db(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    cm = 0
    for v in cfg:
        if v == 'M':
            cm += 1
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if cm < 2:
                conv2d = Conv_DCF_db(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = Conv_DCF(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'F': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


def vgg11_dcf(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg13_dcf(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg16_dcf(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg19_dcf(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_dcf_db(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf_db(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg13_dcf_db(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf_db(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg16_dcf_db(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf_db(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg19_dcf_db(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf_db(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vggs_dcf_db(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers_dcf_db(cfg['F'], batch_norm=False), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model