import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random
import numpy as np
from itertools import combinations
import utils

class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, t_feat, s_feat):
        if t_feat.dim() == 2:
            t_feat = t_feat.unsqueeze(2).unsqueeze(3)
            s_feat = s_feat.unsqueeze(2).unsqueeze(3)

        return (self.transform(t_feat) - s_feat).pow(2).mean()


class AttentionTransfer(nn.Module):
    def forward(self, t_feat, s_feat):
        s_attention = F.normalize(t_feat.pow(2).mean(1).view(t_feat.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(s_feat.pow(2).mean(1).view(s_feat.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()