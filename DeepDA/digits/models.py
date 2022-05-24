# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier1(nn.Module):
    """1 FC layer"""

    def __init__(self, nclass=10, in_feature=128):
        super(Classifier1, self).__init__()
        self.fc = nn.Linear(in_feature, nclass)
        self.__in_features = nclass

    def forward(self, x):
        x = self.fc(x)
        return x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [{"params": self.fc.parameters(), "lr_mult": 10, "decay_mult": 2}]

        return parameter_list


def weights_init(m):
    """Weight init function for layers"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


def call_bn(bn, x):
    """call batch norm layer"""
    return bn(x)


class SVHN_generator(nn.Module):
    """Generator for SVHN dataset"""

    def __init__(self, input_channel=3):
        super(SVHN_generator, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(128 * 4 * 4, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        self.__in_features = 128

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.relu(call_bn(self.bn1, h))
        h = self.c2(h)
        h = F.relu(call_bn(self.bn2, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.c3(h)
        h = F.relu(call_bn(self.bn3, h))
        h = self.c4(h)
        h = F.relu(call_bn(self.bn4, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.c5(h)
        h = F.relu(call_bn(self.bn5, h))
        h = self.c6(h)
        h = F.relu(call_bn(self.bn6, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit = torch.sigmoid(self.linear1(h))
        return logit

    def output_num(self):
        return self.__in_features


class USPS_generator(nn.Module):
    """Generator for USPS dataset"""

    def __init__(self, input_channel=1):
        super(USPS_generator, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 32, kernel_size=5, stride=1, padding=0)
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.linear1 = nn.Linear(128 * 4 * 4, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.__in_features = 128

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.relu(call_bn(self.bn1, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.c2(h)
        h = F.relu(call_bn(self.bn2, h))

        h = self.c3(h)
        h = F.relu(call_bn(self.bn3, h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit = torch.sigmoid(self.linear1(h))
        return logit

    def output_num(self):
        return self.__in_features
