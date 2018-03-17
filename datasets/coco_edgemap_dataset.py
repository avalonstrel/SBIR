from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb, load_bndbox
import os
import re
from PIL import Image
import json
import cv2
import pickle
"""Sketch Dataset"""


class CoCoEdgeMapDataset(data.Dataset):
    # augment_types=[""], levels="cs", mode="train",flag="two_loss", train_split=20,  pair_inclass_num=5,pair_outclass_num=0,edge_map=True):
    def __init__(self,  opt):
        # parameters setting
        self.opt = opt
        self.root = opt.data_root
        photo_types = opt.sketchy_photo_types
        sketch_types = opt.sketchy_sketch_types
        mode = opt.phase
        self.photo_imgs = []
        self.photo_neg_imgs = []
        self.fg_labels = []
        self.labels = []
        self.bndboxes = []
        tri_mode = mode

        if mode == "train":
            start, end = 0, 1200
        elif mode == 'test':
            start, end = 1200, 10000

        
        root = os.path.join(self.root, mode + '2017')
        annotation_root = self.opt.annotation_root
        self.creat_index(annotation_root)
        for i, img_id in enumerate(self.anns.keys()):
            self.photo_imgs.append(self.id2path(img_id))
            self.photo_neg_imgs.append(self.id2path(img_id))
            self.fg_labels.append(i)
            self.labels.append(self.cats[img_id])
            self.bndboxes.append(self.anns[img_id]['bbox'])
        self.n_labels = len(self.catToImgs)
        self.n_fg_labels = len(self.fg_labels)

        print('Total COCO Class:{} Total Num:{}'.format(self.n_labels, self.n_fg_labels))



        save_filename = "coco_image_list.pkl"
        pickle.dump({'photo_imgs': self.photo_imgs, 'photo_neg_imgs': self.photo_neg_imgs,
                     'fg_labels': self.fg_labels, 'labels': self.labels, 'bndboxes': self.bndboxes,
                     'n_labels': self.n_labels, 'n_fg_labels': self.n_fg_labels}, open(save_filename, 'wb'))
        pair_inclass_num, pair_outclass_num = self.opt.pair_num

        if tri_mode == "train" and not self.opt.model == 'cls_model':
            self.generate_triplet(pair_inclass_num, pair_outclass_num)

        print("{} pairs loaded. After generate triplet".format(len(self.photo_imgs)))

    def id2path(self, img_id, root):
        path = os.path.join(root, '0' * (12 - len(img_id)) + str(img_id) + '.png')
        return path

    def creat_transform(self):
        transforms_list = []
        if self.opt.random_crop:
            transforms_list.append(transforms.Resize((256, 256)))
            transforms_list.append(transforms.RandomCrop(
                (self.opt.scale_size, self.opt.scale_size)))
        else:
            transforms_list.append(transforms.Resize(
                (self.opt.scale_size, self.opt.scale_size)))
        if self.opt.flip:
            transforms_list.append(transforms.RandomVerticalFlip())
        #transforms_list.append(transforms.Resize((self.opt.scale_size, self.opt.scale_size)))
        transforms_list.append(transforms.ToTensor())
        self.transform_fun = transforms.Compose(transforms_list)
        self.test_transform_fun = transforms.Compose([transforms.Resize(
            (self.opt.scale_size, self.opt.scale_size)), transforms.ToTensor()])

    def creat_index(self, annotation_path):
        print('loading annotations into memory...')
        tic = time.time()
        annotations = json.load(open(annotation_file, 'r'))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in annotations:
            for ann in annotations['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in annotations:
            for img in annotations['images']:
                imgs[img['id']] = img

        if 'categories' in annotations:
            for cat in annotations['categories']:
                cats[cat['id']] = cat

        if 'annotations' in annotations and 'categories' in annotations:
            for ann in annotations['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.cats = cats
        self.catToImgs = catToImgs
        self.imgToAnns = imgToAnns

    def transform(self, pil, bndbox):

        # print(np.array(pil).shape)
        if self.opt.image_type == 'GRAY':
            pil = pil.convert('L')
        else:
            pil = pil.convert('RGB')
        pil_numpy = np.array(pil)
        pil_numpy = self.crop(pil_numpy, bndbox)
        #pil_numpy = cv2.resize(pil_numpy,(self.opt.scale_size,self.opt.scale_size))

        # if self.opt.image_type == 'GRAY':
        #    pil_numpy = pil_numpy.reshape(pil_numpy.shape + (1,))
        if self.transform_fun is not None:
            pil = Image.fromarray(pil_numpy)
            pil_numpy = self.transform_fun(pil)

        return pil_numpy

    def crop(self, pil_numpy, bb):
        # print(pil_numpy.shape)
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        if len(pil_numpy.shape) == 3:
            return pil_numpy[bndbox[y1]:bndbox[y2], bndbox[x1]:bndbox[x2], :]
        elif len(pil_numpy.shape) == 2:
            return pil_numpy[bndbox[y1]:bndbox[y2], bndbox[x1]:bndbox[x2]]

    def load_sketch(self, pil, bndbox):
        pil = pil.convert('L')
        pil_numpy = np.array(pil)
        # print(pil_numpy.shape)
        pil_numpy = self.crop(pil_numpy, bndbox)
        # print(pil_numpy.shape)
        if not self.opt.sketch_type == 'RGB':
            edge_map = cv2.Canny(pil_numpy, 0, 200)
        # print(edge_map.shape)
        #edge_map = cv2.resize(edge_map,(self.opt.scale_size,self.opt.scale_size))
        # if self.opt.sketch_type == 'RGB':
        #    edge_map = to_rgb(edge_map)
        # elif self.opt.sketch_type == 'GRAY':
        #    edge_map = edge_map.reshape(edge_map.shape + (1,))
        if self.transform_fun is not None:
            edge_map = Image.fromarray(edge_map)
            edge_map = self.transform_fun(edge_map)

        return edge_map

    def __len__(self):
        return len(self.photo_imgs)

    def __getitem__(self, index):
        photo_img, photo_neg_img, fg_label, label = self.photo_imgs[
            index], self.photo_neg_imgs[index], self.fg_labels[index], self.labels[index]
        bndbox = self.bndboxes[index]

        photo_pil, photo_neg_pil = Image.open(
            photo_img), Image.open(photo_neg_img)
        # if self.transform is not None:
        sketch_pil = self.load_sketch(photo_pil, bndbox)
        photo_pil = self.transform(photo_pil, bndbox)
        photo_neg_pil = self.transform(photo_neg_pil, bndbox)
        #print(sketch_pil.size(), photo_pil.size())
        #print(label, fg_label)
        return sketch_pil, photo_pil, photo_neg_pil, label, fg_label, label

    def generate_triplet(self, pair_inclass_num, pair_outclass_num=0):
        photo_neg_imgs, photo_imgs, fg_labels, labels = [], [], [], []

        labels_dict = [[] for i in range(self.n_labels)]
        for i, label in enumerate(self.labels):
            labels_dict[label].append(i)
        fg_labels_dict = [[] for i in range(self.n_fg_labels)]
        for i, fg_label in enumerate(self.fg_labels):
            fg_labels_dict[fg_label].append(i)

        for i, (photo_img, fg_label, label) in enumerate(zip(self.photo_imgs, self.fg_labels, self.labels)):
            num = len(labels_dict[label])
            inds = [labels_dict[label].index(i)]
            for j in range(pair_inclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                photo_neg_imgs.append(self.photo_imgs[labels_dict[label][ind]])
                photo_imgs.append(photo_img)
                fg_labels.append(fg_label)
                labels.append(label)

        num = len(self.photo_imgs)
        for i, (photo_img, fg_label, label) in enumerate(zip(self.photo_imgs, self.fg_labels, self.labels)):
            inds = [i]
            for j in range(pair_outclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label] or ind in labels_dict[label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                photo_neg_imgs.append(self.photo_imgs[ind])
                photo_imgs.append(photo_img)
                fg_labels.append(fg_label)
                labels.append(label)

        self.photo_neg_imgs, self.photo_imgs, self.fg_labels, self.labels = photo_neg_imgs, photo_imgs, fg_labels, labels
