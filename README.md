
# Embedding Transfer with Label Relaxation for Improved Metric Learning

Official PyTorch implementation of CVPR 2021 paper [**Embedding Transfer with Label Relaxation for Improved Metric Learning**](https://arxiv.org/abs/2103.14908). 

Embedding trnasfer with **Relaxed Contrastive Loss** improves performance, or reduces sizes and output dimensions of embedding model effectively.

This repository provides source code of experiments on three datasets (CUB-200-2011, Cars-196 and Stanford Online Products) 
including **relaxed contrastive loss**, **relaxed MS loss**, and 6 other knowledge distillation or embedding transfer methods such as:
- *FitNet*, Fitnets: hints for thin deep nets
- *Attention*, Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
- *CRD*, Contrastive Representation Distillation
- *DarkRank*, Darkrank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer
- *PKT*, Learning Deep Representations with Probabilistic Knowledge Transfer
- *RKD*, Relational Knowledge Distillation

## Overview

### Relaxed Contrastive Loss
- Relaxed contrastive loss exploits pairwise similarities between samples in the source embedding space as relaxed labels, 
  and transfers them through a contrastive loss used for learning target embedding models.
  
<p align="center"><img src="misc/overview.png" alt="graph" width="80%"></p>

### Experimental Restuls
- Our method achieves the state of the art when embedding dimension is 512, and is as competitive as recent metric learning models 
  even with a substantially smaller embedding dimension. In all experiments, it is superior to other embedding transfer techniques. 

<p align="center"><img src="misc/Recalls_ET.png" alt="graph" width="90%"></p>

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Prepare Datasets

1. Download three public benchmarks for deep metric learning.
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))

2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)

## Prepare Pretrained Source models
1. Download the pretrained source models.

```bash
sh scripts/download_pretrained_source_models.sh
```

## Training Target Embedding Network with Relaxed Contrastive Loss
### Self-transfer Setting
- Transfer the knowledge of source model to target model with the same architecture and embedding dimension for performance improvement.
- Source Embedding Network (Inception-BN, 512 dim) ⮕ Target Embedding Network (Inception-BN, 512 dim)

#### CUB-200-2011

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model bn_inception \
--embedding-size 512 --batch-size 90 --IPC 3 --dataset cub --epochs 90 \
--source-ckpt ./pretrained_source/bn_inception/cub_bn_inception_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

#### Cars-196

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model bn_inception \ 
--embedding-size 512 --batch-size 90 --IPC 3 --dataset cars --epochs 90 \
--source-ckpt ./pretrained_source/bn_inception/cars_bn_inception_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

#### SOP

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model bn_inception \
--embedding-size 512 --batch-size 90 --IPC 3 --dataset SOP --epochs 150 \
--source-ckpt ./pretrained_source/bn_inception/SOP_bn_inception_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

<table>
<thead>
<tr>
<th align="center"colspan="2"></th>
<th align="center"colspan="3"><strong>CUB-200-2011</strong></th>
<th align="center"colspan="3"><strong>Cars-196</strong></th>
<th align="center"colspan="3"><strong>SOP</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Method</td>
<td align="center">Backbone</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
</tr>
<tr>
<td align="center"><em>Source</em>: PA</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">69.1</td>
<td align="center">78.9</td>
<td align="center">86.1</td>
<td align="center">86.4</td>
<td align="center">91.9</td>
<td align="center">95.0</td>
<td align="center">79.2</td>
<td align="center">90.7</td>
<td align="center">96.2</td>
</tr>
<tr>
<td align="center">FitNet</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">69.9</td>
<td align="center">79.5</td>
<td align="center">86.2</td>
<td align="center">87.6</td>
<td align="center">92.2</td>
<td align="center">95.6</td>
<td align="center">78.7</td>
<td align="center">90.4</td>
<td align="center">96.1</td>
</tr>
<tr>
<td align="center">Attention</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">66.3</td>
<td align="center">76.2</td>
<td align="center">84.5</td>
<td align="center">84.7</td>
<td align="center">90.6</td>
<td align="center">94.2</td>
<td align="center">78.2</td>
<td align="center">90.4</td>
<td align="center">96.2</td>
</tr>
<tr>
<td align="center">CRD</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">67.7</td>
<td align="center">78.1</td>
<td align="center">85.7</td>
<td align="center">85.3</td>
<td align="center">91.1</td>
<td align="center">94.8</td>
<td align="center">78.1</td>
<td align="center">90.2</td>
<td align="center">95.8</td>
</tr>
<tr>
<td align="center">DarkRank</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">66.7</td>
<td align="center">76.5</td>
<td align="center">84.8</td>
<td align="center">84.0</td>
<td align="center">90.0</td>
<td align="center">93.8</td>
<td align="center">75.7</td>
<td align="center">88.3</td>
<td align="center">95.3</td>
</tr>
<tr>
<td align="center">PKT</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">69.1</td>
<td align="center">78.8</td>
<td align="center">86.4</td>
<td align="center">86.4</td>
<td align="center">91.6</td>
<td align="center">94.9</td>
<td align="center">78.4</td>
<td align="center">90.2</td>
<td align="center">96.0</td>
</tr>
<tr>
<td align="center">RKD</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">70.9</td>
<td align="center">80.8</td>
<td align="center">87.5</td>
<td align="center">88.9</td>
<td align="center">93.5</td>
<td align="center">96.4</td>
<td align="center">78.5</td>
<td align="center">90.2</td>
<td align="center">96.0</td>
</tr>
<tr>
<td align="center"><strong>Ours</strong></td>
<td align="center">BN<sup>512</sup></td>
<td align="center"><strong>72.1</strong></td>
<td align="center"><strong>81.3</strong></td>
<td align="center"><strong>87.6</strong></td>
<td align="center"><strong>89.6</strong></td>
<td align="center"><strong>94.0</strong></td>
<td align="center"><strong>96.5</strong></td>
<td align="center"><strong>79.8</strong></td>
<td align="center"><strong>91.1</strong></td>
<td align="center"><strong>96.3</strong></td>
</tr>
</tbody>
</table>


### Dimensionality Reduction Setting

- Transfer to the same architecture with a lower embedding dimension for efficient image retrieval. 
- Source Embedding Network (Inception-BN, 512 dim) ⇨ Target Embedding Network (Inception-BN, 64 dim)

#### CUB-200-2011

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model bn_inception \
--embedding-size 64 --batch-size 90 --IPC 3 --dataset cub --epochs 90 \
--source-ckpt ./pretrained_source/bn_inception/cub_bn_inception_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

#### Cars-196

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model bn_inception \
--embedding-size 64 --batch-size 90 --IPC 3 --dataset cars --epochs 90 \
--source-ckpt ./pretrained_source/bn_inception/cars_bn_inception_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

#### SOP

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model bn_inception \
--embedding-size 64 --batch-size 90 --IPC 3 --dataset SOP --epochs 150 \
--source-ckpt ./pretrained_source/bn_inception/SOP_bn_inception_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

<table>
<thead>
<tr>
<th align="center"colspan="2"></th>
<th align="center"colspan="3"><strong>CUB-200-2011</strong></th>
<th align="center"colspan="3"><strong>Cars-196</strong></th>
<th align="center"colspan="3"><strong>SOP</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Method</td>
<td align="center">Backbone</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
</tr>
<tr>
<td><em>Source</em>: PA</td>
<td align="center">BN<sup>512</sup></td>
<td align="center">69.1</td>
<td align="center">78.9</td>
<td align="center">86.1</td>
<td align="center">86.4</td>
<td align="center">91.9</td>
<td align="center">95.0</td>
<td align="center">79.2</td>
<td align="center">90.7</td>
<td align="center">96.2</td>
</tr>
<tr>
<td>FitNet</td>
<td align="center">BN<sup>64</sup></td>
<td align="center">62.3</td>
<td align="center">73.8</td>
<td align="center">83.0</td>
<td align="center">81.2</td>
<td align="center">87.7</td>
<td align="center">92.5</td>
<td align="center"><strong>76.6</strong></td>
<td align="center"><strong>89.3</strong></td>
<td align="center"><strong>95.4</strong></td>
</tr>
<tr>
<td>Attention</td>
<td align="center">BN<sup>64</sup></td>
<td align="center">58.3</td>
<td align="center">69.4</td>
<td align="center">79.1</td>
<td align="center">79.2</td>
<td align="center">86.7</td>
<td align="center">91.8</td>
<td align="center">76.3</td>
<td align="center">89.2</td>
<td align="center">95.4</td>
</tr>
<tr>
<td>CRD</td>
<td align="center">BN<sup>64</sup></td>
<td align="center">60.9</td>
<td align="center">72.7</td>
<td align="center">81.7</td>
<td align="center">79.2</td>
<td align="center">87.2</td>
<td align="center">92.1</td>
<td align="center">75.5</td>
<td align="center">88.3</td>
<td align="center">95.3</td>
</tr>
<tr>
<td>DarkRank</td>
<td align="center">BN<sup>64</sup></td>
<td align="center">63.5</td>
<td align="center">74.3</td>
<td align="center">83.1</td>
<td align="center">78.1</td>
<td align="center">85.9</td>
<td align="center">91.1</td>
<td align="center">73.9</td>
<td align="center">87.5</td>
<td align="center">94.8</td>
</tr>
<tr>
<td>PKT</td>
<td align="center">BN<sup>64</sup></td>
<td align="center">63.6</td>
<td align="center">75.8</td>
<td align="center">84.0</td>
<td align="center">82.2</td>
<td align="center">88.7</td>
<td align="center">93.5</td>
<td align="center">74.6</td>
<td align="center">87.3</td>
<td align="center">94.2</td>
</tr>
<tr>
<td>RKD</td>
<td align="center">BN<sup>64</sup></td>
<td align="center">65.8</td>
<td align="center">76.7</td>
<td align="center">85.0</td>
<td align="center">83.7</td>
<td align="center">89.9</td>
<td align="center">94.1</td>
<td align="center">70.2</td>
<td align="center">83.8</td>
<td align="center">92.1</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td align="center">BN<sup>64</sup></td>
<td align="center"><strong>67.4</strong></td>
<td align="center"><strong>78.0</strong></td>
<td align="center"><strong>85.9</strong></td>
<td align="center"><strong>86.5</strong></td>
<td align="center"><strong>92.3</strong></td>
<td align="center"><strong>95.3</strong></td>
<td align="center">76.3</td>
<td align="center">88.6</td>
<td align="center">94.8</td>
</tr>
</tbody>
</table>

### 3. Model compression Setting
- Transfer to a smaller network with a lower embedding dimension for usage in low-power and resource limited devices.
- Source Embedding Network (ResNet50, 512 dim) ⮕ Target Embedding Network (ResNet18, 128 dim)

#### CUB-200-2011

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model resnet18 \
--embedding-size 128 --batch-size 90 --IPC 3 --dataset cub --epochs 90 \
--source-ckpt ./pretrained_source/resnet50/cub_resnet50_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

#### Cars-196

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model resnet18 \
--embedding-size 128 --batch-size 90 --IPC 3 --dataset cars --epochs 90 \
--source-ckpt ./pretrained_source/resnet50/cars_resnet50_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

#### SOP

```bash
python code/train_target.py --gpu-id 0 --loss Relaxed_Contra --model resnet18 \
--embedding-size 128 --batch-size 90 --IPC 3 --dataset SOP --epochs 150 \
--source-ckpt ./pretrained_source/resnet50/SOP_resnet50_512dim_Proxy_Anchor_ckpt.pth \
--view 2 --sigma 1 --save 1
```

<table>
<thead>
<tr>
<th align="center"colspan="2"></th>
<th align="center"colspan="3"><strong>CUB-200-2011</strong></th>
<th align="center"colspan="3"><strong>Cars-196</strong></th>
<th align="center"colspan="3"><strong>SOP</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Method</td>
<td align="center">Backbone</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
<td align="center">R@1</td>
<td align="center">R@2</td>
<td align="center">R@4</td>
</tr>
<tr>
<td><em>Source</em>: PA</td>
<td align="center">R50<sup>512</sup></td>
<td align="center">69.9</td>
<td align="center">79.6</td>
<td align="center">88.6</td>
<td align="center">87.7</td>
<td align="center">92.7</td>
<td align="center">95.5</td>
<td align="center">80.5</td>
<td align="center">91.8</td>
<td align="center">98.8</td>
</tr>
<tr>
<td>FitNet</td>
<td align="center">R18<sup>128</sup></td>
<td align="center">61.0</td>
<td align="center">72.2</td>
<td align="center">81.1</td>
<td align="center">78.5</td>
<td align="center">86.0</td>
<td align="center">91.4</td>
<td align="center">76.7</td>
<td align="center">89.4</td>
<td align="center">95.5</td>
</tr>
<tr>
<td>Attention</td>
<td align="center">R18<sup>128</sup></td>
<td align="center">61.0</td>
<td align="center">71.7</td>
<td align="center">81.5</td>
<td align="center">78.6</td>
<td align="center">85.9</td>
<td align="center">91.0</td>
<td align="center">76.4</td>
<td align="center">89.3</td>
<td align="center">95.5</td>
</tr>
<tr>
<td>CRD</td>
<td align="center">R18<sup>128</sup></td>
<td align="center">62.8</td>
<td align="center">73.8</td>
<td align="center">83.2</td>
<td align="center">80.6</td>
<td align="center">87.9</td>
<td align="center">92.5</td>
<td align="center">76.2</td>
<td align="center">88.9</td>
<td align="center">95.3</td>
</tr>
<tr>
<td>DarkRank</td>
<td align="center">R18<sup>128</sup></td>
<td align="center">61.2</td>
<td align="center">72.5</td>
<td align="center">82.0</td>
<td align="center">75.3</td>
<td align="center">83.6</td>
<td align="center">89.4</td>
<td align="center">72.7</td>
<td align="center">86.7</td>
<td align="center">94.5</td>
</tr>
<tr>
<td>PKT</td>
<td align="center">R18<sup>128</sup></td>
<td align="center">65.0</td>
<td align="center">75.6</td>
<td align="center">84.8</td>
<td align="center">81.6</td>
<td align="center">88.8</td>
<td align="center">93.4</td>
<td align="center">76.9</td>
<td align="center">89.2</td>
<td align="center">95.5</td>
</tr>
<tr>
<td>RKD</td>
<td align="center">R18<sup>128</sup></td>
<td align="center">65.8</td>
<td align="center">76.3</td>
<td align="center">84.8</td>
<td align="center">84.2</td>
<td align="center">90.4</td>
<td align="center">94.3</td>
<td align="center">75.7</td>
<td align="center">88.4</td>
<td align="center">95.1</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td align="center">R18<sup>128</sup></td>
<td align="center"><strong>66.6</strong></td>
<td align="center"><strong>78.1</strong></td>
<td align="center"><strong>85.9</strong></td>
<td align="center"><strong>86.0</strong></td>
<td align="center"><strong>91.6</strong></td>
<td align="center"><strong>95.3</strong></td>
<td align="center"><strong>78.4</strong></td>
<td align="center"><strong>90.4</strong></td>
<td align="center"><strong>96.1</strong></td>
</tr>
</tbody>
</table>


## Evaluating Image Retrieval

Follow the below steps to evaluate the trained model. 

Trained best model will be saved in the `./logs/folder_name`.

```bash
# The parameters should be changed according to the model to be evaluated.
python evaluate.py --gpu-id 0 \
                   --batch-size 120 \
                   --model bn_inception \
                   --embedding-size 512 \
                   --dataset cub \
                   --ckpt /set/your/model/path/best_model.pth
```

## Acknowledgements

Our code is modified and adapted on these great repositories:

- [Proxy Anchor Loss for Deep Metric Learning](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)
- [No Fuss Distance Metric Learning using Proxies](https://github.com/dichotomies/proxy-nca)
- [PyTorch Metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)


## Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{kim2021embedding,
      title={Embedding Transfer with Label Relaxation for Improved Metric Learning},
      author={Kim, Sungyeon and Kim, Dongwon and Cho, Minsu and Kwak, Suha},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year={2021}
    }

