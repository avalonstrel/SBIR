import torch
from torch.utils import data
import numpy as np
from torch.autograd import Variable
def sample_negative(ind, x, search_collection, sample_num, distance_fun):
    distance_collection = []
    num = len(search_collection)
    for i in range(sample_num[1]):
        sample_ind = np.random.randint(num)
        while sample_ind == ind:
            sample_ind = np.random.randint(num)
        dist = distance_fun(x, search_collection[sample_ind])
        distance_collection.append((sample_ind, dist))
    sorted_dist = sorted(distance_collection, key=lambda x:x[1])
    negative_inds = [term[0] for term in sorted_dist[:sample_num[0]]]
    return negative_inds

def hard_negative_mining(model, dataset, query_what, distance_fun, sample_num=(10,50)):
    if query_what == 'image':
        dataset.query_image()
    elif query_what == 'sketch':
        dataset.query_sketch()

    search_collection = []
    query_collection = []
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=20,
                shuffle=False) 
    for i, batch_data in enumerate(dataloader):
        x0, x1, x2, attr, fg_label, label = batch_data
        x0 = Variable(x0.cuda())
        x1 = Variable(x1.cuda())
        x2 = Variable(x2.cuda())

        #print(x0, x1, x2)
        output0, output1, output2 = model(x0, x1, x2)
        
        output0 = output0.data.cpu()
        output1 = output1.data.cpu()
        for j in range(output0.size(0)):
            query_collection.append(output0[j])
            search_collection.append(output1[j])
        
    # query_collection = query_collection.data.cpu()
    # search_collection = search_collection.data.cpu()
    query_imgs, search_neg_imgs, search_imgs, attributes, fg_labels, labels = [],[],[],[],[],[]

    for i, query in enumerate(query_collection):
        negative_inds = sample_negative(i, query, search_collection, sample_num, distance_fun)
        for negative_ind in negative_inds:
            
            query_imgs.append(dataset.query_imgs[i])
            search_imgs.append(dataset.search_imgs[i])
            search_neg_imgs.append(dataset.search_imgs[negative_ind])
            fg_labels.append(dataset.fg_labels[i])
            labels.append(dataset.labels[i])
            if dataset.name == 'hairstyle':
                attributes.append(dataset.attributes[i])
    dataset.query_imgs, dataset.search_neg_imgs, dataset.search_imgs, dataset.fg_labels, dataset.labels = query_imgs, search_neg_imgs, search_imgs, fg_labels, labels
    print(len(dataset.query_imgs), len(dataset.search_neg_imgs), len(dataset.search_imgs), len(dataset.fg_labels), len(dataset.labels))
    if dataset.name == 'hairstyle':
        dataset.attributes = attributes
    print('Hard negative mining Finish.')

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
