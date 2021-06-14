from .base import *

class SOP(BaseDataset):
    def __init__(self, root, mode, transform = None, is_CRD = False):
        self.root = root + '/Stanford_Online_Products'
        self.mode = mode
        self.transform = transform
        self.is_CRD = is_CRD
        if self.mode == 'train':
            self.classes = range(0,11318)
        elif self.mode == 'eval':
            self.classes = range(11318,22634)
            
        BaseDataset.__init__(self, self.root, self.mode, self.transform, self.is_CRD)
        metadata = open(os.path.join(self.root, 'Ebay_train.txt' if self.classes == range(0, 11318) else 'Ebay_test.txt'))
        for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id)-1 in self.classes:
                    self.ys += [int(class_id)-1]
                    self.I += [int(image_id)-1]
                    self.im_paths.append(os.path.join(self.root, path))
                    
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
                    else:
                        self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]