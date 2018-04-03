import torch
from torch.utils import data
import numpy as np

def create_dataset(opt):

    name = opt.dataset_type
    if name == 'hairstyle':
        from .hairstyle_dataset import HairStyleDataset
        return HairStyleDataset(opt)
    elif name == 'sketchx':
        from .sketchx_dataset import SketchXDataset
        return SketchXDataset(opt)
    elif name == 'sketchy':
        from .sketchy_dataset import SketchyDataset
        return SketchyDataset(opt)
    elif name == 'imagenet':
        from .imagenet_edgemap_dataset import ImageNetEdgeMapDataset
        return ImageNetEdgeMapDataset(opt)
    elif name == 'imagenet_hed':
        from .imagenet_hed_dataset import ImageNetHEDDataset
        return ImageNetHEDDataset(opt)
    elif name == 'tuberlin':
        from .tuberlin_dataset import TUBerlinDataset
        return TUBerlinDataset(opt)
    elif name == 'coco':
        from .coco_edgemap_dataset import CoCoEdgeMapDataset
        return CoCoEdgeMapDataset(opt)
    return None

class ModerateNegativeBatchSampler(data.sampler.Sampler):
    """
    A sampler for moderate negative sampling in batch form
    """
    def __init__(self, labels_dict):
        #print(labels_dict)
        self.labels_dict = labels_dict
        self.P = 8
        self.K = 4
        self.num = np.sum([len(self.labels_dict[label]) for label in self.labels_dict])
        print('num',self.num,int(1.0* self.num // (self.P * self.K)))
    def __iter__(self):
        for idx in range(len(self)):
            labels_set = np.random.choice(len(self.labels_dict), size=self.P, replace=False)
            indices = []
            for label in labels_set:
                label_num = len(self.labels_dict[label])
                label_inds = np.random.choice(label_num, size=self.K, replace=False)
                indices.extend([self.labels_dict[label][i] for i in label_inds])
            #print(indices)
            yield indices

    def __len__(self):
        return int(1.0* self.num // (self.P * self.K))



class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.initialize(self.opt)

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        #BaseDataLoader.initialize(self, opt)
        self.dataset = create_dataset(opt)
        if opt.phase == 'train':
            batch_size = opt.batch_size
        else:
            if not opt.retrieval_once:
                batch_size = opt.batch_size
            

            else:
                batch_size = len(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.n_threads))        
        if opt.loss_type[0] == "moderate_triplet":
            self.batch_sampler = ModerateNegativeBatchSampler(self.dataset.labels_dict)


    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
