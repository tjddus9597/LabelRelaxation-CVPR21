import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet18, resnet34, resnet50, resnet101
import torch.utils.model_zoo as model_zoo

class Resnet18(nn.Module):
    def __init__(self,embedding_size, pretrained=True, l2_norm=True, bn_freeze = True):
        super(Resnet18, self).__init__()

        self.model = resnet18(pretrained)
        self.l2_norm = l2_norm
        self.embedding_size = embedding_size
        self.model.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.model.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        f1 = self.model.layer1(x)
        f2 = self.model.layer2(f1)
        f3 = self.model.layer3(f2)
        f4 = self.model.layer4(f3)

        avg_x = self.model.gap(f4)
        max_x = self.model.gmp(f4)

        x = max_x + avg_x
        feat = x.view(x.size(0), -1)
        emb = self.model.embedding(feat)
        
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=1)
        
        return [f1, f2, f3, f4, feat], emb

    def _initialize_weights(self):
        init.orthogonal_(self.model.embedding.weight)
        init.constant_(self.model.embedding.bias, 0)

class Resnet34(nn.Module):
    def __init__(self,embedding_size, pretrained=True, l2_norm=True, bn_freeze = True):
        super(Resnet34, self).__init__()

        self.model = resnet34(pretrained)
        self.l2_norm = l2_norm
        self.embedding_size = embedding_size
        self.model.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.model.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        f1 = self.model.layer1(x)
        f2 = self.model.layer2(f1)
        f3 = self.model.layer3(f2)
        f4 = self.model.layer4(f3)

        avg_x = self.model.gap(f4)
        max_x = self.model.gmp(f4)

        x = max_x + avg_x
        feat = x.view(x.size(0), -1)
        emb = self.model.embedding(feat)
        
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=1)
        
        return [f1, f2, f3, f4, feat], emb

    def _initialize_weights(self):
        init.orthogonal_(self.model.embedding.weight)
        init.constant_(self.model.embedding.bias, 0)

class Resnet50(nn.Module):
    def __init__(self,embedding_size, pretrained=True, l2_norm=True, bn_freeze = True):
        super(Resnet50, self).__init__()

        self.model = resnet50(pretrained)
        self.l2_norm = l2_norm
        self.embedding_size = embedding_size
        self.model.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.model.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        f1 = self.model.layer1(x)
        f2 = self.model.layer2(f1)
        f3 = self.model.layer3(f2)
        f4 = self.model.layer4(f3)

        avg_x = self.model.gap(f4)
        max_x = self.model.gmp(f4)

        x = max_x + avg_x
        feat = x.view(x.size(0), -1)
        emb = self.model.embedding(feat)
        
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=1)
        
        return [f1, f2, f3, f4, feat], emb

    def _initialize_weights(self):
        init.orthogonal_(self.model.embedding.weight)
        init.constant_(self.model.embedding.bias, 0)

class Resnet101(nn.Module):
    def __init__(self,embedding_size, pretrained=True, l2_norm=True, bn_freeze = True):
        super(Resnet101, self).__init__()

        self.model = resnet101(pretrained)
        self.l2_norm = l2_norm
        self.embedding_size = embedding_size
        self.model.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.model.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        f1 = self.model.layer1(x)
        f2 = self.model.layer2(f1)
        f3 = self.model.layer3(f2)
        f4 = self.model.layer4(f3)

        avg_x = self.model.gap(f4)
        max_x = self.model.gmp(f4)

        x = max_x + avg_x
        feat = x.view(x.size(0), -1)
        emb = self.model.embedding(feat)
        
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=1)
        
        return [f1, f2, f3, f4, feat], emb

    def _initialize_weights(self):
        init.orthogonal_(self.model.embedding.weight)
        init.constant_(self.model.embedding.bias, 0)