import torch
import torchvision
from torchvision import transforms
import PIL.Image
import random

class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class print_shape():
    def __call__(self, im):
        print(im.size)
        return im

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im
    
class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor

    
class MultiTransforms:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, is_train = True, is_inception = True, view=1):
        self.is_train = is_train
        self.is_inception = is_inception
        self.view = view
        
        resnet_sz_resize = 256
        resnet_sz_crop = 224
        resnet_mean = [0.485, 0.456, 0.406]
        resnet_std = [0.229, 0.224, 0.225]
        self.resnet_transform = transforms.Compose([
            transforms.RandomResizedCrop(resnet_sz_crop) if is_train else Identity(),
            transforms.RandomHorizontalFlip() if is_train else Identity(),
            transforms.Resize(resnet_sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(resnet_sz_crop) if not is_train else Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std)
        ])

        inception_sz_resize = 256
        inception_sz_crop = 224
        inception_mean = [104, 117, 128]
        inception_std = [1, 1, 1]
        self.inception_transform = transforms.Compose(
           [
            RGBToBGR(),
            transforms.RandomResizedCrop(inception_sz_crop) if is_train else Identity(),
            transforms.RandomHorizontalFlip() if is_train else Identity(),
            transforms.Resize(inception_sz_resize) if not is_train else Identity(),
            transforms.CenterCrop(inception_sz_crop) if not is_train else Identity(),
            transforms.ToTensor(),
            ScaleIntensities([0, 1], [0, 255]),
            transforms.Normalize(mean=inception_mean, std=inception_std)
           ])

        if is_inception:
            self.transform = self.inception_transform
        else:
            self.transform = self.resnet_transform

    def __call__(self, x):
        if self.is_train:
            transform = []
            for i in range(self.view):
                transform = transform + [self.transform(x)]
            return transform
        else:
            return self.transform(x)

def make_transform(is_train = True, is_inception = True):
    # Resolution Resize List : 146, 182, 256, 292, 361, 512
    # Resolution Crop List: 128, 160, 224, 256, 324, 448
    
    resnet_sz_resize = 256
    resnet_sz_crop = 224
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    resnet_transform = transforms.Compose([
        transforms.RandomResizedCrop(resnet_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(resnet_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(resnet_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

    inception_sz_resize = 256
    inception_sz_crop = 224
    inception_mean = [104, 117, 128]
    inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
       [
        RGBToBGR(),
        transforms.RandomResizedCrop(inception_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(inception_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(inception_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities([0, 1], [0, 255]),
        transforms.Normalize(mean=inception_mean, std=inception_std)
       ])
    
    return inception_transform if is_inception else resnet_transform