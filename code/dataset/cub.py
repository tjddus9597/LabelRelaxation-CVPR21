from .base import *

class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform = None, is_CRD = False):
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.transform = transform
        self.is_CRD = is_CRD
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform, self.is_CRD)
        index = 0
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1
        
        # For CRD Training
        num_classes = len(self.classes)
        num_samples = len(self.I)
        label = self.ys

        if self.mode == 'train':
            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]