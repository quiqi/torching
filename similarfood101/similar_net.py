# -*- encoding: utf-8 -*-
import torch.nn as nn


class SiameseTrainNet(nn.Module):
    def __init__(self, net: nn.Module):
        super(SiameseTrainNet, self).__init__()
        self.net = net

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        y1 = self.net.forward(x1)
        y2 = self.net.forward(x2)
        return [y1, y2]
