import torch, math, argparse, os
import numpy as np
import random, dataset, utils, net
import Relaxed_losses, CRD_loss, DML_losses, ET_losses

from net.resnet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

from tqdm import *
import wandb

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of Embedding Transfer with Label Relaxation for Improved Metric Learning'  
    + 'Our code is based on `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/`')

parser.add_argument('--LOG_DIR',  default='./logs', help = 'Path to log')
parser.add_argument('--save', default = 0, type = int, help = 'Save checkpoint or not')
parser.add_argument('--workers', default = 8, type = int, dest = 'nb_workers', help = 'Number of workers for dataloader.')
parser.add_argument('--gpu-id', default = 0, type = int, help = 'ID of GPU that is used for training.')

# Dataset
parser.add_argument('--DATA_DIR',  default='./data', help = 'Path of data')
parser.add_argument('--dataset', default='cub', help = 'Training dataset, e.g. cub, cars, SOP', choices=['cub', 'cars', 'SOP'])

# Optimization
parser.add_argument('--embedding-size', default = 512, type = int, dest = 'sz_embedding', help = 'Size of embedding appended to target model')
parser.add_argument('--batch-size', default = 90, type = int, dest = 'sz_batch', help = 'Number of samples per batch')
parser.add_argument('--epochs', default = 90, type = int, dest = 'nb_epochs', help = 'Number of training epochs')
parser.add_argument('--model', default = 'bn_inception', help = 'Model for training', 
                    choices=['bn_inception', 'resnet18', 'resnet50', 'resnet101']) 
parser.add_argument('--loss', default = 'Relaxed_Contra', help = 'Criterion for training',
                   choices=['Relaxed_Contra', 'Relaxed_MS', 'RKD', 'PKT', 'DarkRank', 'Attention', 'FitNet', 'CRD'])
parser.add_argument('--optimizer', default = 'adamw', help = 'Optimizer setting')
parser.add_argument('--lr', default = 1e-4, type =float, help = 'Learning rate setting')
parser.add_argument('--weight-decay', default = 1e-4, type =float, help = 'Weight decay setting')

# For DML optimization
parser.add_argument('--IPC', type = int, help = 'Balanced sampling, images per class')
parser.add_argument('--bn-freeze', default = 0, type = int, help = 'Batch normalization parameter freeze')
parser.add_argument('--warm', default = 0, type = int, help = 'Warmup training epochs')
parser.add_argument('--target-norm', default = 0, type = int, help = 'Target model L2 normalization')
parser.add_argument('--source-norm', default = 1, type = int, help = 'Source model L2 normalization')

# For Embedding Transfer
parser.add_argument('--source-ckpt', default = '', help = 'Loading checkpoint')
parser.add_argument('--view', default = 1, type = int, help = 'Choose number of view for multi-view data augmentation')

# Hyperparameters of Relaxed Losses
parser.add_argument('--delta', default = 1, type = float, help = 'Delta in relaxed contrastive loss')
parser.add_argument('--sigma', default = 1, type = float, help = 'Sigma in relaxed contrastive loss')
                    
parser.add_argument('--remark', default = '', help = 'Any reamrk')

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model, args.loss, args.sz_embedding, 
                                                                               args.optimizer, args.lr, args.sz_batch, args.remark)
LOG_DIR = os.path.abspath(LOG_DIR)
DATA_DIR = os.path.abspath(args.DATA_DIR)

# Wandb Initialization
wandb.init(project='EmbeddingTransfer', notes=LOG_DIR, name = args.remark)
wandb.config.update(args)

# Dataset Loader and Sampler
trn_dataset = dataset.load(
        name = args.dataset,
        root = DATA_DIR,
        mode = 'train',
        transform = dataset.utils.MultiTransforms(
            is_train = True, 
            is_inception = (args.model == 'bn_inception'),
            view = args.view
        ), is_CRD = args.loss == 'CRD')

if args.IPC:
    balanced_sampler = sampler.ClassBalancedBatchSampler(trn_dataset, batch_size=args.sz_batch, images_per_class = args.IPC)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = balanced_sampler
    )
    print('Balanced Sampling')
    
else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )
    print('Random Sampling')

ev_dataset = dataset.load(
        name = args.dataset,
        root = DATA_DIR,
        mode = 'eval',
        transform = dataset.utils.make_transform(
            is_train = False, 
            is_inception = (args.model == 'bn_inception')
        ))

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)

nb_classes = trn_dataset.nb_classes()

# Backbone Model
if args.model.find('bn_inception')+1:
    model_target = bn_inception(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet18')+1:
    model_target = Resnet18(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet50')+1:
    model_target = Resnet50(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet101')+1:
    model_target = Resnet101(embedding_size=args.sz_embedding, pretrained=True, l2_norm=args.target_norm, bn_freeze = args.bn_freeze)
    
if args.model.find('resnet')+1:
    model_source = Resnet50(embedding_size=512, pretrained=True, l2_norm=args.source_norm, bn_freeze = args.bn_freeze)
else:
    model_source = bn_inception(embedding_size=512, pretrained=True, l2_norm=args.source_norm, bn_freeze = args.bn_freeze)

model_target = model_target.cuda()
model_source = model_source.cuda()
for param in list(set(model_source.parameters())):
    param.requires_grad = False
    
if os.path.isfile(args.source_ckpt):
    print('=> Loading Checkpoint {} for Source Model!'.format(args.source_ckpt))
    checkpoint = torch.load(args.source_ckpt, map_location='cuda:{}'.format(0))
    model_source.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> No Checkpoint for Source Model!!!'.format(args.source_ckpt))
    
if args.gpu_id == -1:
    model_source = nn.DataParallel(model_source)
    model_target = nn.DataParallel(model_target)
                    
# Our Losses with Label Relaxation
if args.loss == 'Relaxed_Contra':
    relaxedcontra_criterion = Relaxed_losses.Relaxed_Contra(delta = args.delta, sigma = args.sigma).cuda()
elif args.loss == 'Relaxed_MS':
    relaxedMS_criterion = Relaxed_losses.Relaxed_MS(delta = args.delta, sigma = args.sigma).cuda()

# Embedding Transfer Losses
elif args.loss == 'RKD':                    
    dist_criterion = ET_losses.RkdDistance().cuda()
    angle_criterion = ET_losses.RKdAngle().cuda()
elif args.loss == 'PKT':
    PKT_criterion = ET_losses.PKT().cuda()
elif args.loss == 'DarkRank':
    Hard_DarkRank_criterion = KD_losses.HardDarkRank()
                    
# Knowledge Distillation Losses
elif args.loss == 'CRD':
    CRD_criterion = CRD_loss.CRDLoss(sz_embed = args.sz_embedding, n_data = trn_dataset.__len__()).cuda()
elif args.loss == 'Attention':
    AT_criterion = KD_losses.AttentionTransfer().cuda()
elif args.loss == 'FitNet':
    if args.model == 'bn_inception':
        FitNet_criterions = [
            KD_losses.FitNet(576, 576).cuda(), 
            KD_losses.FitNet(1056, 1056).cuda(), 
            KD_losses.FitNet(1024, 1024).cuda(), 
            KD_losses.FitNet(args.sz_embedding, 512).cuda()
        ]
    elif args.model == 'resnet18':
        FitNet_criterions = [
            KD_losses.FitNet(64, 256).cuda(), 
            KD_losses.FitNet(128, 512).cuda(), 
            KD_losses.FitNet(256, 1024).cuda(), 
            KD_losses.FitNet(512, 2048).cuda(), 
            KD_losses.FitNet(args.sz_embedding, 512).cuda()
        ]

# Proxy Anchor loss
ProxyAnchor_criterion = DML_losses.Proxy_Anchor(nb_classes = trn_dataset.nb_classes(), sz_embed = args.sz_embedding, mrg = 0.1, alpha = 32).cuda()

# Train Parameters
param_groups = [{'params': model_target.parameters() if args.gpu_id != -1 else model_target.module.parameters()}]
             
# Additional Parameters for KD Losses
if args.loss == 'CRD':
    param_groups.append({'params': CRD_criterion.embed_t.parameters(), 'lr':float(args.lr)})
    param_groups.append({'params': CRD_criterion.embed_s.parameters(), 'lr':float(args.lr)})
if args.loss == 'FitNet':
    fit_param_dict = [{'params':f.parameters(), 'lr':float(args.lr)} for f in FitNet_criterions]
    param_groups += fit_param_dict
                    
# NOTE: KD methods require additional DML loss (this case, Proxy-Anchor)
if args.loss in ['Attention', 'FitNet', 'CRD']:
    param_groups.append({'params': ProxyAnchor_criterion.parameters(), 'lr':float(args.lr) * 100})

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, eps=1)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.nb_epochs)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall=[0]
best_epoch = 0

for epoch in range(0, args.nb_epochs):
    model_target.train()
    model_source.eval()
    
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model_target.model.modules() if args.gpu_id != -1 else model_target.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    
    # Applying warm-up strategy following Proxy-Anchor loss, only new parameters are trained
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(ProxyAnchor_criterion.parameters()) + list(model_target.model.embedding.parameters())
        else:
            unfreeze_model_param = list(ProxyAnchor_criterion.parameters()) + list(model_target.module.model.embedding.parameters())

        if epoch == 0:
            for param in list(set(model_target.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        elif epoch == args.warm:
            for param in list(set(model_target.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True
                
    pbar = tqdm(enumerate(dl_tr))
    for batch_idx, data in pbar:
        if args.loss == 'CRD':
            x, y, index, contrast_idx = data
            index = index.cuda()
            contrast_idx = contrast_idx.cuda()
        else:
            x, y = data
        y = y.squeeze().cuda()
        
        # Multi-view Augmentation Strategy
        if args.view > 1:
            x = torch.cat(x, dim=0)
            y = torch.cat([y]*args.view)
            
        # t_feats: feature-maps and last feature vector (output of last pooling layer) of target embedding model
        # t_emb: embedding vector of target embedding model
        t_feats, t_emb = model_target(x.squeeze().cuda())
                    
        # s_feats: feature-maps and last feature vector (output of last pooling layer) of source embedding model
        # s_emb: embedding vector of source embedding model
        with torch.no_grad():
            s_feats, s_emb = model_source(x.squeeze().cuda())
        
        # Relaxed Losses
        if args.loss == 'Relaxed_Contra':
            loss = relaxedcontra_criterion(t_emb, s_emb)
    
        elif args.loss == 'Relaxed_MS':
            loss = relaxedMS_criterion(t_emb, s_emb)
        
        # Embedding Trasnfer Losses
        elif args.loss == 'RKD':
            dist_loss = 1 * dist_criterion(t_emb, s_emb)
            angle_loss = 2 * angle_criterion(t_emb, s_emb)
            loss = dist_loss + angle_loss
            
        elif args.loss == 'PKT':
            loss = PKT_criterion(t_emb, s_emb)
        
        elif args.loss == 'DarkRank':
            loss = Hard_DarkRank_criterion(t_emb, s_emb, y)
            
        # Knowledge Distillation Losses
        elif args.loss == 'CRD':
            loss = CRD_criterion(t_feats[-1], s_feats[-1], index, contrast_idx)
                    
        elif args.loss == 'Attention':
            assert(len(t_feats)!=1 and len(s_feats)!=1)
            loss = AT_criterion(t_feats[0], s_feats[0]) + AT_criterion(t_feats[1], s_feats[1]) + AT_criterion(t_feats[2], s_feats[2])
            if args.model == 'resnet18':
                loss += AT_criterion(t_feats[3], s_feats[3])
            
        elif args.loss == 'FitNet':
            assert(len(t_feats)!=1 and len(s_feats)!=1)
            loss = FitNet_criterions[0](t_feats[0], s_feats[0]) + FitNet_criterions[1](t_feats[1], s_feats[1]) + \
                    FitNet_criterions[2](t_feats[2], s_feats[2]) + FitNet_criterions[-1](t_feats, s_feats)
            if args.model == 'resnet18':
                loss += FitNet_criterions[3](t_feats[3], s_feats[3])

        # NOTE: KD methods require additional DML loss
        if args.loss in ['Attention', 'FitNet', 'CRD']:
            loss += ProxyAnchor_criterion(t_emb, y)
        
        opt.zero_grad()
        loss.backward()

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))
        
    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()
    
    if epoch >= 0:
        print("**Evaluating...**")
        if args.dataset != 'SOP':
            k_list = [2**i for i in range(4)]
        else:
            k_list = [10**i for i in range(4)]
        with torch.no_grad():
            Recalls = utils.evaluate_euclid(model_target, dl_ev, k_list, 5000)
            
        # Logging Evaluation Score
        for i, k in enumerate(k_list):
            wandb.log({"R@{}".format(k): Recalls[i]}, step=epoch)
        
        # Best model is saved
        if best_recall[0] < Recalls[0] and args.save:
            print('Save Best Model!')
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists('{}'.format(LOG_DIR)):
                os.makedirs('{}'.format(LOG_DIR))
            torch.save({'model_state_dict':model_target.state_dict() if args.gpu_id != -1 else model_target.module.state_dict()}, 
                       '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))
            with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                for i, k in enumerate(k_list):
                    f.write("Best Recall@{}: {:.4f}\n".format(k, best_recall[i] * 100))