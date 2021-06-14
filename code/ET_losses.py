import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random
import numpy as np
from DML_losses import TripletLoss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RKdAngle(nn.Module):
    def forward(self, t_emb, s_emb):
        with torch.no_grad():
            sd = (s_emb.unsqueeze(0) - s_emb.unsqueeze(1))
            norm_sd = F.normalize(td, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        td = (t_emb.unsqueeze(0) - t_emb.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(t_angle, s_angle, reduction='mean')
        return loss

class RkdDistance(nn.Module):
    def forward(self, t_emb, s_emb):
        with torch.no_grad():
            sd = pdist(s_emb, squared=False)
            mean_sd = sd[sd>0].mean()
            sd = sd / mean_sd

        td = pdist(t_emb, squared=False)
        mean_td = td[td>0].mean()
        td = td / mean_td

        loss = F.smooth_l1_loss(td, sd, reduction='mean')
        return loss
    
class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self):
        super(PKT, self).__init__()
        self.eps = 0.0000001

    def forward(t_emb, s_emb):
        # Normalize each vector by its norm
        t_norm = torch.sqrt(torch.sum(t_emb ** 2, dim=1, keepdim=True))
        t_emb = t_emb / (t_norm + self.eps)
        t_emb[t_emb != t_emb] = 0

        s_norm = torch.sqrt(torch.sum(s_emb ** 2, dim=1, keepdim=True))
        s_emb = s_emb / (s_norm + self.eps)
        s_emb[s_emb != s_emb] = 0

        # Calculate the cosine similarity
        t_similarity = torch.mm(t_emb, target.transpose(0, 1))
        s_similarity = torch.mm(s_emb, source.transpose(0, 1))

        # Scale cosine similarity to 0..1
        t_similarity = (t_similarity + 1.0) / 2.0
        s_similarity = (s_similarity + 1.0) / 2.0

        # Transform them into probabilities
        t_similarity = t_similarity / torch.sum(t_similarity, dim=1, keepdim=True)
        s_similarity = s_similarity / torch.sum(s_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(s_similarity * torch.log((s_similarity + self.eps) / (t_similarity + self.eps)))

        return loss
    
class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len
        self.triplet_factor = 1
        self.triplet_loss = TripletLoss(margin=0.2) # NOTE Followed hyper-parameter setting used in RKD

    def forward(self, t_emb, s_emb, label):
        t_emb = F.normalize(t_emb, p=2, dim=1)
        s_emb = F.normalize(s_emb, p=2, dim=1)
        score_s_emb = -1 * self.alpha * pdist(s_emb, squared=False).pow(self.beta)
        score_t_emb = -1 * self.alpha * pdist(t_emb, squared=False).pow(self.beta)

        permute_idx = score_s_emb.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_t_emb = torch.gather(score_t_emb, 1, permute_idx)

        log_prob = (ordered_t_emb - torch.stack([torch.logsumexp(ordered_t_emb[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()
        loss += self.triplet_factor * self.triplet_loss(t_emb, label)
        
        return loss