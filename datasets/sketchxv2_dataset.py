from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb
import os, re, json
import scipy.io as sio
from PIL import Image
import cv2

#SketchX dataset
class SketchXDataset(data.Dataset):
    def name(self):
        return "sketchx"
    def __init__(self, opt):# root,thing_type="chairs",levels="cs", mode="train", flag="two_loss"):
        self.opt = opt
        # Parameters Setting
        root = opt.data_root
        mode = opt.phase
        self.mode = mode
        sketch_root = os.path.join(root,  "ShoeV2_sketch_png")
        image_root = os.path.join(root, "ShoeV2_photo")
        self.query_what = self.opt.query_what
        self.flag = opt.loss_flag
        self.levels = opt.sketch_levels

        transforms_list = []
        if self.opt.random_crop:
            transforms_list.append(transforms.Resize((256,256)))
            transforms_list.append(transforms.RandomCrop((self.opt.scale_size, self.opt.scale_size)))
        else:
            transforms_list.append(transforms.Resize((self.opt.scale_size, self.opt.scale_size)))
        if self.opt.flip:
            transforms_list.append(transforms.RandomVerticalFlip())
        #transforms_list.append(transforms.Resize((self.opt.scale_size, self.opt.scale_size)))
        transforms_list.append(transforms.ToTensor())

        self.transform_fun = transforms.Compose(transforms_list)
        self.test_transform_fun = transforms.Compose([transforms.Resize((self.opt.scale_size, self.opt.scale_size)), transforms.ToTensor()])


        cls_dict = {}
        with open(os.path.join(root, "photo_{}.txt".format(mode)),"r") as f:
            photo_imgs = f.read().splitlines()

        # Define a class dictionary
        cls_ind = 0
        for img_path in photo_imgs:
            if img_path[:-4] not in cls_dict:
                cls_dict[img_path[:-4]] = cls_ind
                cls_ind += 1

        with open(os.path.join(root, "sketch_{}.txt".format(mode)),"r") as f:
            sketch_imgs = f.read().splitlines()

        # Refactor the sketch imgs and photo_imgs
        self.sketch_imgs, self.photo_imgs = [], []
        self.labels = []
        self.fg_labels = []
        self.attributes = []
        self.attribute_size = 2
        for sketch_name in sketch_imgs:
            i = sketch_name.find("_")
            cls_name = sketch_name[:i]
            j = sketch_name.find('.')
            self.sketch_imgs.append(os.path.join(sketch_root, sketch_name[:j]+".png"))
            self.photo_imgs.append(os.path.join(image_root, cls_name+".png"))
            self.attributes.append(np.array([1.,0.]))
            self.fg_labels.append(cls_dict[cls_name])
            self.labels.append(cls_dict[cls_name])

        self.n_labels = len(cls_dict)
        self.n_fg_labels = len(cls_dict)
        print(self.n_labels)

        self.ori_sketch_imgs = self.sketch_imgs.copy()
        self.ori_photo_imgs = self.photo_imgs.copy()
        self.ori_labels = self.labels.copy()
        self.ori_fg_labels = self.fg_labels.copy()
        self.ori_attributes = self.attributes.copy()
        print("{} images loaded.".format(len(self.photo_imgs)))

        # For generate triplet
        if self.query_what == "image":
            self.query_image()
        elif self.query_what == "sketch":
            self.query_sketch()

        self.generate_triplet_all()
        print("Total Sketchy Class:{}, fg class: {}".format(self.n_labels, self.n_fg_labels))
        print("{} images loaded. After generate triplet".format(len(self.query_imgs)))

    def generate_triplet_all(self):
        pair_inclass_num, pair_outclass_num = self.opt.pair_num
        if self.opt.triplet_type == 'annotation':
            self.generate_annotation_triplet()
        elif self.opt.triplet_type == 'random':
            if self.opt.task == 'fg_sbir' and self.opt.phase == 'train':
                self.generate_triplet(pair_inclass_num, pair_outclass_num)
            elif self.opt.task == 'fg_sbir' and self.opt.phase == 'test':
                self.generate_triplet(0,1)


    def generate_triplet(self, pair_inclass_num, pair_outclass_num=0):
        query_imgs, search_neg_imgs, search_imgs,attributes, fg_labels, labels = [],[],[],[],[],[]
        num = len(self.search_imgs)
        for i, (query_img, search_img, fg_label, label,attribute) in enumerate(zip(self.query_imgs, self.search_imgs, self.fg_labels, self.labels,self.attributes)):
            inds = [i]
            for j in range(pair_outclass_num):
                ind = np.random.randint(num)
                while ind in inds:
                    ind = np.random.randint(num)
                inds.append(ind)
                query_imgs.append(query_img)
                search_neg_imgs.append(self.search_imgs[ind])
                search_imgs.append(search_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)


        self.query_imgs, self.search_neg_imgs, self.search_imgs, self.fg_labels, self.labels, self.attributes = query_imgs, search_neg_imgs, search_imgs, fg_labels, labels, attributes

    def query_image(self):
        self.query_imgs = self.ori_sketch_imgs.copy()
        self.search_imgs = self.ori_photo_imgs.copy()
        self.search_neg_imgs = self.ori_photo_imgs.copy()
        self.labels = self.ori_labels.copy()
        self.fg_labels = self.ori_fg_labels.copy()

        self.load_search = self.load_image
        self.load_query = self.load_sketch
        print("Query is Sketch Search Image")
    def query_sketch(self):
        self.query_imgs = self.ori_photo_imgs.copy()
        self.search_imgs = self.ori_sketch_imgs.copy()
        self.search_neg_imgs = self.ori_sketch_imgs.copy()
        self.labels = self.ori_labels.copy()
        self.fg_labels = self.ori_fg_labels.copy()

        self.load_query = self.load_image
        self.load_search = self.load_sketch
        print("Query is Image Search Sketch")

    def load_image(self, pil):
        def show(mode, pil_numpy):
            print(mode, len(",".join([str(i) for i in pil_numpy.flatten() if i != 0])))

        if self.opt.image_type == 'RGB':
            pil = pil.convert('RGB')
            pil_numpy = np.array(pil)
        elif self.opt.image_type == 'GRAY':
            pil = pil.convert('L')
            pil_numpy = np.array(pil)
        elif self.opt.image_type == 'EDGE':
            pil = pil.convert('L')
            pil_numpy = np.array(pil)
            #show('edge', pil_numpy)
            pil_numpy = cv2.Canny(pil_numpy, 0, 200)

        if 'densenet' in self.opt.feature_model and not self.opt.image_type == 'RGB':
            pil_numpy = to_rgb(pil_numpy)

        transform_fun = self.transform_fun if self.mode == 'train' else self.test_transform_fun
        if transform_fun is not None :
            pil = Image.fromarray(pil_numpy)
            pil_numpy = transform_fun(pil)
        return pil_numpy

    def load_sketch(self, pil):
        def show(mode, pil_numpy):
            print(mode, len(",".join([str(i) for i in pil_numpy.flatten() if i != 0])))
        pil = pil.convert('L')
        pil_numpy = np.array(pil)

        if self.opt.sketch_type == 'RGB' or 'densenet' in self.opt.feature_model:
            pil_numpy = to_rgb(pil_numpy)

        transform_fun = self.transform_fun if self.mode == 'train' else self.test_transform_fun
        if transform_fun is not None:
            pil = Image.fromarray(pil_numpy)
            pil_numpy = transform_fun(pil)

        return pil_numpy


    def transform(self, pil):
        pil = pil.convert('RGB')
        pil_numpy = np.array(pil)
        if len(pil_numpy.shape) == 2:
            pil_numpy = to_rgb(pil_numpy)
        elif pil_numpy.shape[2] == 4:
            pil_numpy = to_rgb(pil_numpy[:,:,3])
        pil_numpy = cv2.resize(pil_numpy,(self.opt.scale_size, self.opt.scale_size))
        if self.transform_fun is not None:
            pil = Image.fromarray(pil_numpy)
            pil_numpy = self.transform_fun(pil)
        return pil_numpy

    def __len__(self):
        return len(self.query_imgs)

    def __getitem__(self,index):
        search_img,query_img,search_neg_img,fg_label,label, attribute = self.search_imgs[index], self.query_imgs[index], self.search_neg_imgs[index], self.fg_labels[index], self.labels[index], self.attributes[index]
        if self.levels == "stack":
            query_s_pil, query_c_pil = self.load_query(Image.open(query_img[0])), self.load_query(Image.open(query_img[1]))
            query_s_pil[:,:,1] = query_c_pil[:,:,0]
            query_pil = query_s_pil
        else:
            query_pil = Image.open(query_img)
            query_pil = self.load_query(query_pil)
        search_pil, search_neg_pil = Image.open(search_img), Image.open(search_neg_img)


        search_pil = self.load_search(search_pil)
        search_neg_pil = self.load_search(search_neg_pil)
        return query_pil, search_pil, search_neg_pil, attribute, fg_label, label
