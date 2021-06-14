import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import math

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    _, J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def evaluate_euclid(model, dataloader, k_list, num_buffer):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    
    # get predictions by assigning nearest K neighbors with Euclidean distance
    K = max(k_list)
    Y = []
    xs = []
    for x in X:
        if len(xs) < num_buffer:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            
            dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
            dist_emb = X.pow(2).sum(1) + dist_emb.t()

            y = T[dist_emb.topk(1 + K, largest = False)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
    dist_emb = X.pow(2).sum(1) + dist_emb.t()
    y = T[dist_emb.topk(1 + K, largest = False)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ K
    recall = []
    for k in k_list:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall