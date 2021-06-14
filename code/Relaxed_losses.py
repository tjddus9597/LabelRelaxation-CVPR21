import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random
import numpy as np
from itertools import combinations
import utils

def pdist(A, B, squared = False, eps = 1e-12):
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)
    
    if not squared:
        D = D.sqrt()
        
    if torch.equal(A,B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0
        
    return D
    
class Relaxed_Contra(nn.Module):
    """Relaxed Contrative loss function. """
    def __init__(self, sigma=1, delta=1):
        super(Relaxed_Contra, self).__init__()
        
        self.sigma = sigma
        self.delta = delta
        
    def forward(self, t_emb, s_emb):
        s_emb = F.normalize(s_emb, p=2, dim=1)
        
        T_dist = pdist(t_emb, t_emb, False)
        dist_mean = T_dist.mean(1, keepdim=True)
        T_dist = T_dist / dist_mean
            
        with torch.no_grad():
            S_dist = pdist(s_emb, s_emb, False)
            P = torch.exp(-S_dist.pow(2) / self.sigma)
        
        pos_weight = P
        neg_weight = 1-P
        
        pull_losses = torch.relu(T_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - T_dist).pow(2) * neg_weight

        pull_losses = pull_losses[T_dist>0]
        push_losses = push_losses[T_dist>0]
        loss = (pull_losses.sum() + push_losses.sum()) / len(t_emb)

        return loss
    
class Relaxed_MS(nn.Module):
    """Relaxed MS loss function. """
    def __init__(self, sigma=1, delta=1):
        super(Relaxed_MS, self).__init__()
        
        self.sigma = sigma
        self.delta = delta
        self.scale_pos = 1
        self.scale_neg = 2

    def forward(self, t_emb, s_emb):
        s_emb = F.normalize(s_emb, p=2, dim=1)
        
        losses = []
        T_dist = pdist(t_emb, t_emb, False)
        T_dist = T_dist / T_dist.mean(1, keepdim=True)
        
        S_dist = pdist(s_emb, s_emb, False)
        P = torch.exp(-S_dist.pow(2) / self.sigma)
        
        batch_size = len(student_col) // 2
        for i in range(batch_size):
            P[i,i+batch_size] = 1
            P[i+batch_size,i] = 1
        
        for i in range(len(student_col)):
            dist_i = torch.cat([T_dist[i][0:i], T_dist[i][i+1:]])
            P_i = torch.cat([P[i][0:i], P[i][i+1:]])
            
            pos_weight = P_i
            neg_weight = (1-P_i)
            
            pos_exp = pos_weight * torch.exp(self.scale_pos * (dist_i))
            neg_exp = neg_weight * torch.exp(self.scale_neg * (self.delta - dist_i))
            
            P_sim_sum = pos_exp.sum()
            N_sim_sum = neg_exp.sum()

            pulling_loss = 1 / self.scale_pos * torch.log(1 + P_sim_sum)
            pushing_loss = 1 / self.scale_neg * torch.log(1 + N_sim_sum)
            
            losses.append(pulling_loss + pushing_loss)

        losses = torch.stack(losses)
        loss = losses[losses>0].mean()
        
        return loss