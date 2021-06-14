from .base import *
import scipy.io

class Cars(BaseDataset):
    def __init__(self, root, mode, transform = None, is_CRD = False):
        self.root = root + '/cars196'
        self.mode = mode
        self.transform = transform
        self.is_CRD = is_CRD
        if self.mode == 'train':
            self.classes = range(0,98)
        elif self.mode == 'eval':
            self.classes = range(98,196)
                
        BaseDataset.__init__(self, self.root, self.mode, self.transform, self.is_CRD)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)
                self.I += [index]
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
