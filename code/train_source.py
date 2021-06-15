import torch, math, argparse, os
import numpy as np
import random, dataset, utils, net, DML_losses

from net.resnet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of Embedding Transfer with Label Relaxation for Improved Metric Learning'  
    + 'Our code is based on `https://github.com/dichotomies/proxy-nca`')

parser.add_argument('--LOG_DIR',  default='./pretrained_source', help = 'Path to log')
parser.add_argument('--save', default = 0, type = int, help = 'Save checkpoint or not')
parser.add_argument('--workers', default = 8, type = int, dest = 'nb_workers', help = 'Number of workers for dataloader.')
parser.add_argument('--gpu-id', default = 0, type = int, help = 'ID of GPU that is used for training.')

# Dataset
parser.add_argument('--DATA_DIR',  default='./data', help = 'Path of data')
parser.add_argument('--dataset', default='cub', help = 'Training dataset, e.g. cub, cars, SOP', choices=['cub', 'cars', 'SOP'])

# Optimization
parser.add_argument('--embedding-size', default = 512, type = int, dest = 'sz_embedding', help = 'Size of embedding appended to backbone model')
parser.add_argument('--batch-size', default = 90, type = int, dest = 'sz_batch', help = 'Number of samples per batch')
parser.add_argument('--epochs', default = 90, type = int, dest = 'nb_epochs', help = 'Number of training epochs')
parser.add_argument('--model', default = 'bn_inception', help = 'Model for training', 
                    choices=['bn_inception', 'resnet18', 'resnet50', 'resnet101']) 
parser.add_argument('--loss', default = 'Proxy_Anchor', help = 'Criterion for training',
                   choices=['Proxy_Anchor', 'Proxy_NCA', 'MS', 'Contrastive', 'Triplet', 'NPair'])
parser.add_argument('--optimizer', default = 'adamw', help = 'Optimizer setting')
parser.add_argument('--lr', default = 1e-4, type =float, help = 'Learning rate setting')
parser.add_argument('--weight-decay', default = 1e-4, type =float, help = 'Weight decay setting')
parser.add_argument('--lr-decay-step', default = 10, type =int, help = 'Learning decay step setting')
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float, help = 'Learning decay gamma setting')

# For DML optimization
parser.add_argument('--IPC', type = int, help = 'Balanced sampling, images per class')
parser.add_argument('--bn-freeze', default = 0, type = int, help = 'Batch normalization parameter freeze')
parser.add_argument('--warm', default = 0, type = int, help = 'Warmup training epochs')
parser.add_argument('--l2-norm', default = 0, type = int, help = 'Model L2 normalization')

# Hyperparameters of Proxy-Anchor Loss
parser.add_argument('--alpha', default = 32, type = float, help = 'Scaling Parameter setting')
parser.add_argument('--mrg', default = 0.1, type = float, help = 'Margin parameter setting')
                    
parser.add_argument('--remark', default = '', help = 'Any reamrk')

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/{}'.format(args.model)
LOG_DIR = os.path.abspath(LOG_DIR)
DATA_DIR = os.path.abspath(args.DATA_DIR)

# Wandb Initialization
wandb.init(project='Source_Pretraining', notes=LOG_DIR, name=args.remark)
wandb.config.update(args)

# Dataset Loader and Sampler
trn_dataset = dataset.load(
        name = args.dataset,
        root = DATA_DIR,
        mode = 'train',
        transform = dataset.utils.make_transform(
            is_train = True, 
            is_inception = (args.model == 'bn_inception'))
)

if args.IPC:
    balanced_sampler = sampler.ClassBalancedBatchSampler(trn_dataset, batch_size=args.sz_batch, images_per_class = args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.sz_batch, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
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
            is_inception = (args.model == 'bn_inception'))
)

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
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet18')+1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet50')+1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
elif args.model.find('resnet101')+1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

# DML Losses
if args.loss == 'Proxy_Anchor':
    criterion = DML_losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = args.sz_embedding, mrg = args.mrg, alpha = args.alpha).cuda()
elif args.loss == 'Proxy_NCA':
    criterion = DML_losses.Proxy_NCA().cuda()
elif args.loss == 'MS':
    criterion = DML_losses.MultiSimilarityLoss().cuda()
elif args.loss == 'Contrastive':
    criterion = DML_losses.ContrastiveLoss().cuda()
elif args.loss == 'Triplet':
    criterion = DML_losses.TripletLoss().cuda()
elif args.loss == 'NPair':
    criterion = DML_losses.NPairLoss().cuda()

# Train Parameters
param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else 
                 list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
    {'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(args.lr) * 1},
]
if args.loss == 'Proxy_Anchor':
    param_groups.append({'params': criterion.proxies, 'lr':float(args.lr) * 100})

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall=[0]
best_epoch = 0

for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    
    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
        else:
            unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:
        _, m = model(x.squeeze().cuda())
        loss = criterion(m, y.squeeze().cuda())
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if args.loss == 'Proxy_Anchor':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

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
        with torch.no_grad():
            if args.dataset != 'SOP':
                k_list = [2**i for i in range(4)]
            else:
                k_list = [10**i for i in range(4)]
            Recalls = utils.evaluate_euclid(model, dl_ev, k_list, 5000)
                
        # Logging Evaluation Score
        for i, k in enumerate(k_list):
            wandb.log({"R@{}".format(k): Recalls[i]}, step=epoch)
                        
        # Best model save
        if best_recall[0] < Recalls[0] and args.save:
            print('Save Best Model!')
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists('{}'.format(LOG_DIR)):
                os.makedirs('{}'.format(LOG_DIR))
            torch.save({'model_state_dict':model.state_dict() if args.gpu_id != -1 else model.module.state_dict()}, 
                       '{}/{}_{}_{}dim_{}_ckpt.pth'.format(LOG_DIR, args.dataset, args.model, args.sz_embedding, args.loss))
            with open('{}/{}_{}_{}dim_{}_results.txt'.format(LOG_DIR, args.dataset, args.model, args.sz_embedding, args.loss), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                for i, k in enumerate(k_list):
                    f.write("Best Recall@{}: {:.4f}\n".format(k, best_recall[i] * 100))

    