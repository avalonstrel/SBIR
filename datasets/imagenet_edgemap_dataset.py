from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb, load_bndbox
import os, re
from PIL import Image
import json
import cv2

"""Sketch Dataset"""
class ImageNetEdgeMapDataset(data.Dataset):
    def __init__(self,  opt): #augment_types=[""], levels="cs", mode="train",flag="two_loss", train_split=20,  pair_inclass_num=5,pair_outclass_num=0,edge_map=True):
        # parameters setting
        self.opt = opt
        self.root = opt.data_root
        photo_types = opt.sketchy_photo_types
        sketch_types = opt.sketchy_sketch_types
        mode = opt.phase
        transforms_list = []
        if self.opt.random_crop:
            transforms_list.append(transforms.Resize((256,256)))
            transforms_list.append(transforms.RandomCrop((self.opt.scale_size, self.opt.scale_size)))
        if self.opt.flip:
            transforms_list.append(transforms.RandomVerticalFlip())
        #transforms_list.append(transforms.Resize((self.opt.scale_size, self.opt.scale_size)))
        transforms_list.append(transforms.ToTensor())
        self.transform_fun = transforms.Compose(transforms_list)
        self.test_transform_fun = transforms.Compose([transforms.Resize((self.opt.scale_size, self.opt.scale_size)), transforms.ToTensor()])
        self.photo_imgs = []
        self.photo_neg_imgs = []
        #self.transform_fun = transforms.Compose([transforms.ToTensor()])
        self.fg_labels = []
        self.labels = []
        self.bndboxes = []
        tri_mode = mode

        if mode == "train":
            start, end = 0, 100
        elif mode == 'test':
            start, end = 100, 105
            mode = 'train'

        #photo_roots = [root+photo_type for photo_type in photo_types]
        #print(photo_roots)
        root = os.path.join(self.root, mode)
        annotation_root = os.path.join(self.root, 'Annotation')
        fg_label, label = 0, 0

        for cls_root, subFolders, files in os.walk(root):
            photo_pat = re.compile("n.+\.JPEG")
            photo_imgs = list(filter(lambda fname:photo_pat.match(fname), files))
            annotation_pre_path = os.path.join(annotation_path, cls_root)
            if len(photo_imgs) == 0 or not os.path.exists(annotation_pre_path):
                print(cls_root)
                continue

            for i, photo_img in enumerate(photo_imgs, start=0):
                if i < start or i >= end:
                    continue
                photo_label = photo_img[:(len(photo_img)-5)]
                img_path = os.path.join(root, cls_root, photo_img)
                annotation_path = os.path.join(annotation_path, cls_root, 'Annotation', cls_root, photo_label + '.xml')
                if not os.path.exists(annotation_path):
                    continue
                self.photo_imgs.append(img_path)
                self.photo_neg_imgs.append(img_path)
                self.fg_labels.append(fg_label)
                self.labels.append(label)
                self.bndboxes.append(annotation_path)
                fg_label += 1
            label += 1
        print('Total ImageNet Class:{} Total Num:{}'.format(label, fg_label))
        self.n_labels = label
        self.n_fg_labels = fg_label
        pair_inclass_num, pair_outclass_num = self.opt.pair_num

        if tri_mode == "train" and not self.opt.model == 'cls_model':
            self.generate_triplet(pair_inclass_num,pair_outclass_num)

        print("{} pairs loaded. After generate triplet".format(len(self.photo_imgs)))

    def transform(self, pil, bndbox):

        #print(np.array(pil).shape)
        if self.opt.image_type == 'GRAY':
            pil = pil.convert('L')
        else:
            pil = pil.convert('RGB')
        pil_numpy = np.array(pil)
        pil_numpy = self.crop(pil_numpy, bndbox)
        #pil_numpy = cv2.resize(pil_numpy,(self.opt.scale_size,self.opt.scale_size))

        #if self.opt.image_type == 'GRAY':
        #    pil_numpy = pil_numpy.reshape(pil_numpy.shape + (1,))
        if self.transform_fun is not None:
            pil = Image.fromarray(pil_numpy)
            pil_numpy = self.transform_fun(pil)

        return pil_numpy

    def crop(self, pil_numpy, bndbox):
        return pil_numpy[bndbox['xmin']:bndbox['xmax'],bndbox['ymax']:bndbox['ymax'],:]

    def load_sketch(self, pil, bndbox):
        pil = pil.convert('L')
        pil_numpy = np.array(pil)
        pil_numpy = self.crop(pil_numpy, bndbox)
        edge_map = cv2.Canny(pil_numpy, 100, 200)
        #edge_map = cv2.resize(edge_map,(self.opt.scale_size,self.opt.scale_size))
        if self.opt.sketch_type == 'RGB':
            edge_map = to_rgb(edge_map)
        #elif self.opt.sketch_type == 'GRAY':
        #    edge_map = edge_map.reshape(edge_map.shape + (1,))
        if self.transform_fun is not None:
            edge_map = Image.fromarray(edge_map)
            edge_map = self.transform_fun(edge_map)

        return edge_map
    def __len__(self):
        return len(self.photo_imgs)

    def __getitem__(self,index):
        photo_img,photo_neg_img,fg_label,label = self.photo_imgs[index], self.photo_neg_imgs[index], self.fg_labels[index], self.labels[index]
        bndbox_path = self.bndbox[index]
        bndbox = load_bndbox(bndbox_path)
        photo_pil,photo_neg_pil = Image.open(photo_img), Image.open(photo_neg_img)
        #if self.transform is not None:
        sketch_pil = self.load_sketch(photo_pil, bndbox)
        photo_pil = self.transform(photo_pil, bndbox)
        photo_neg_pil = self.transform(photo_neg_pil, bndbox)
        #print(sketch_pil.size(), photo_pil.size())
        #print(label, fg_label)
        return sketch_pil, photo_pil, photo_neg_pil, label, fg_label, label


    def generate_triplet(self, pair_inclass_num,pair_outclass_num=0):
        photo_neg_imgs, photo_imgs, fg_labels, labels = [],[],[],[]

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
