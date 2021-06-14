import torch, math, argparse, os
import numpy as np
import random, dataset, utils, net

from net.resnet import *
from net.bn_inception import *
import torch.nn.functional as F

from tqdm import *

parser = argparse.ArgumentParser(description=
    'Official implementation of `Embedding Transfer with Label Relaxation for Improved Metric Learning`'  
    + 'Our code is based on `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/`'
)
parser.add_argument('--gpu-id', default = 0, type = int, help = 'ID of GPU that is used for training.')
parser.add_argument('--workers', default = 20, type = int, dest = 'nb_workers', help = 'Number of workers for dataloader.')

# Dataset
parser.add_argument('--DATA_DIR',  default='./data', help = 'Path of data')
parser.add_argument('--dataset', default='cub', help = 'Training dataset, e.g. cub, cars, SOP', choices=['cub', 'cars', 'SOP'])

# For DML evaluation
parser.add_argument('--embedding-size', default = 512, type = int, dest = 'sz_embedding', help = 'Size of embedding that is appended to backbone model.')
parser.add_argument('--batch-size', default = 150, type = int, dest = 'sz_batch', help = 'Number of samples per batch.')
parser.add_argument('--model', default = 'bn_inception', help = 'Model for training', 
                    choices=['bn_inception', 'resnet18', 'resnet50', 'resnet101']) 
parser.add_argument('--bn-freeze', default = 1, type = int, help = 'Batch normalization parameter freeze')
parser.add_argument('--target-norm', default = 0, type = int, help = 'Target L2 normalization')
parser.add_argument('--source-norm', default = 1, type = int, help = 'Source L2 normalization')
parser.add_argument('--ckpt', default = '', help = 'Loading checkpoint')

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

DATA_DIR = os.path.abspath(args.DATA_DIR)

# Dataset Loader and Sampler

ev_dataset = dataset.load(
        name = args.dataset,
        root = DATA_DIR,
        mode = 'eval',
        transform = dataset.utils.MultiTransforms(
            is_train = False, 
            is_inception = (args.model == 'bn_inception'),
            view = 1
        ))

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)

# Backbone Model
if args.model.find('bn_inception')+1:
    model_target = bn_inception(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet18')+1:
    model_target = Resnet18(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet50')+1:
    model_target = Resnet50(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet101')+1:
    model_target = Resnet101(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)

model_target = model_target.cuda()
    
if os.path.isfile(args.ckpt):
    print('=> Loading Checkpoint {} for Target Model'.format(args.ckpt))
    checkpoint = torch.load(args.ckpt, map_location='cuda:{}'.format(0))
    model_target.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> No Checkpoint for Target Model!!!'.format(args.checkpoint))
    
if args.gpu_id == -1:
    model_target = nn.DataParallel(model_target)
    
#  Evaluation
print("**Evaluating...**")
with torch.no_grad():
    if args.dataset != 'SOP':
        k_list = [2**i for i in range(4)]
    else:
        k_list = [10**i for i in range(4)]
    Recalls = utils.evaluate_euclid(model_target, dl_ev, k_list, 5000)