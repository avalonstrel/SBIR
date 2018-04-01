from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb
import os, re
from PIL import Image
import json
import cv2 
"""Sketch Dataset"""
class HairStyleDataset(data.Dataset):
    def __init__(self,  opt): #augment_types=[""], levels="cs", mode="train",flag="two_loss", train_split=20,  pair_inclass_num=5,pair_outclass_num=0,edge_map=True):
        # parameters setting
        self.opt = opt
        self.root = opt.data_root
        self.flag = opt.loss_flag
        if opt.image_type == 'EDGE':
            self.edge_map = True
        else:
            self.edge_map = False
        self.levels = opt.sketch_levels

        self.attributes = []
        self.attributes_dict, self.attribute_size = load_attribute("/home/lhy/datasets/hairstyle_attribute.txt")
        self.transform_fun = transforms.Compose([transforms.ToTensor()])
        train_split = 20
        mode = opt.phase
        augment_types = opt.augment_types
        #print(self.flag, self.edge_map, self.levels, augment_types)
        # Data Initialization
        self.hair_imgs = []
        self.sketch_imgs = []
        self.hair_neg_imgs = []
        self.fg_labels = []
        self.labels = []

        label = 0
        fg_label = 0

        if self.levels == "stack":
            self.levels = "s"

        # load pictures
        for root,subFolders,files in os.walk(self.root):
            hair_pat = re.compile("cropped_\w+.*\d+.*\.jpg")
            hair_imgs = list(filter(lambda fname:hair_pat.match(fname),files))
            if len(hair_imgs) == 0:
                print(root)
                continue
            sketch_imgs=[]
            cls_name = root[root.rfind('/')+1:]
            for i, hair_img in enumerate(hair_imgs):
                digit = re.findall("\d+",hair_img)[0]
                if (mode == "train" and i < train_split) or (mode == "test" and i >= train_split):
                    for level in self.levels:
                        for augment_type in augment_types:

                            flag = "_" if mode == "train" and augment_type != "" else ""
                            sketch_pat = re.compile("cropped_"+augment_type+flag+str(digit)+level+".*\.png")
                            sketch_imgs = list(filter(lambda fname:sketch_pat.match(fname),files))
                            for sketch_img in sketch_imgs:
                                
                                self.hair_imgs.append(os.path.join(root,hair_img))
                                if self.levels == "stack":
                                    sketch_other_img = sketch_img.replace("s.","c.")
                                    sketch_ohter_img = sketch_other_img.replace("s_","c_")
                                    self.sketch_imgs.append([os.path.join(root,sketch_img),os.path.join(root,sketch_other_img)])
                                else:

                                    self.sketch_imgs.append(os.path.join(root,sketch_img))
                                self.hair_neg_imgs.append(os.path.join(root,hair_img))
                                
                                self.attributes.append(self.attributes_dict[cls_name])
                                self.fg_labels.append(fg_label)
                                self.labels.append(label)
                    fg_label += 1
            label += 1
        print("Total :",label)
        self.n_labels = label
        self.n_fg_labels = fg_label
        print("FG TOTAL:",fg_label,len(self.hair_imgs))
        print("{} images loaded.".format(len(self.hair_imgs)))
        self.labels_dict = {i:[] for i in range(self.n_labels)}
        for i, label in enumerate(self.labels):
            self.labels_dict[label].append(i)
            self.fg_labels_dict = {i:[] for i in range(self.n_fg_labels)}
        for i, fg_label in enumerate(self.fg_labels):
            self.fg_labels_dict[fg_label].append(i)

        pair_inclass_num, pair_outclass_num = opt.pair_num
        if mode == "train":
            self.generate_triplet(pair_inclass_num,pair_outclass_num)


    def __len__(self):
        return len(self.hair_imgs)



    def transform(self, pil, mode="sketch"):
        def show(mode, pil_numpy):
            print(mode, ",".join([str(i) for i in pil_numpy.flatten() if i != 0]))
        pil_numpy = np.array(pil)
        
        if len(pil_numpy.shape) == 2:
            if self.edge_map and mode == "image":
                pil_numpy = cv2.Canny(pil_numpy, 100, 200)
            #show("edge",pil_numpy)
            pil_numpy = to_rgb(pil_numpy)
            #pil_numpy = np.tile(pil_numpy,3).reshape(pil_numpy.shape+(-1,))
        elif pil_numpy.shape[2] == 4:

            #show("sketch",pil_numpy[:,:,3])
            pil_numpy = to_rgb(pil_numpy[:,:,3])
            #pil_numpy = np.tile(pil_numpy[:,:,3],3).reshape(pil_numpy.shape[0:2]+(-1,))
            #pil_numpy[:,:,2] = 0
        if self.opt.image_type == 'EDGE':
            gray_pil = Image.fromarray(pil_numpy)
            pil_numpy = np.array(gray_pil.convert('L'))
        pil_numpy = cv2.resize(pil_numpy,(self.opt.scale_size,self.opt.scale_size))
        if self.transform_fun is not None:
            pil_numpy = self.transform_fun(pil_numpy)
        #data_info.write(",".join([str(i) for i in pil_numpy.numpy().flatten() if i != 0])+"\n")
        return pil_numpy

    def __getitem__(self,index):
        #print(len(self.attributes),"hair",len(self.hair_imgs),"ind:",index)
        hair_img,sketch_img,hair_neg_img,fg_label,label,attribute = self.hair_imgs[index], self.sketch_imgs[index], self.hair_neg_imgs[index], self.fg_labels[index], self.labels[index], self.attributes[index]
        open_type = "L" if self.edge_map else "RGB"

        if self.levels == "stack":
            sketch_s_pil, sketch_c_pil = self.transform(Image.open(sketch_img[0])), self.transform(Image.open(sketch_img[1]))
            sketch_s_pil[:,:,1] = sketch_c_pil[:,:,0]
            sketch_pil = sketch_s_pil
        else:
            sketch_pil = Image.open(sketch_img)
            #print("sketch", np.array(sketch_pil).shape)
            sketch_pil = self.transform(sketch_pil)

        hair_pil, hair_neg_pil = Image.open(hair_img).convert(open_type),  Image.open(hair_neg_img).convert(open_type)
        #print("hair", np.array(hair_pil).shape)
        #if self.transform is not None:
        hair_pil = self.transform(hair_pil, "image")
        hair_neg_pil = self.transform(hair_neg_pil, "image")

        return sketch_pil,hair_pil,hair_neg_pil,attribute, fg_label,label

    def generate_triplet(self, pair_inclass_num, pair_outclass_num=0):
        sketch_imgs, hair_neg_imgs, hair_imgs,attributes, fg_labels, labels = [],[],[],[],[],[]

        labels_dict = [[] for i in range(self.n_labels)]
        for i, label in enumerate(self.labels):
            labels_dict[label].append(i)
        fg_labels_dict = [[] for i in range(self.n_fg_labels)]
        for i, fg_label in enumerate(self.fg_labels):
            fg_labels_dict[fg_label].append(i)

        for i, (sketch_img, hair_img, fg_label, label,attribute) in enumerate(zip(self.sketch_imgs, self.hair_imgs, self.fg_labels, self.labels, self.attributes)):
            num = len(labels_dict[label])
            inds = [labels_dict[label].index(i)]
            for j in range(pair_inclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                sketch_imgs.append(sketch_img)
                hair_neg_imgs.append(self.hair_imgs[labels_dict[label][ind]])
                hair_imgs.append(hair_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)
        num = len(self.hair_imgs)
        for i, (sketch_img, hair_img, fg_label, label,attribute) in enumerate(zip(self.sketch_imgs, self.hair_imgs, self.fg_labels, self.labels,self.attributes)):
            inds = [labels_dict[label].index(i)]
            for j in range(pair_outclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label] or ind in labels_dict[label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                sketch_imgs.append(sketch_img)
                hair_neg_imgs.append(self.hair_imgs[ind])
                hair_imgs.append(hair_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)


        self.sketch_imgs, self.hair_neg_imgs, self.hair_imgs, self.fg_labels, self.labels, self.attributes = sketch_imgs, hair_neg_imgs, hair_imgs, fg_labels, labels, attributes

def load_attribute(path):
    with open(path) as reader:
        categories = reader.readline()
        categories = categories.strip().split()
        attributes = {category:[] for category in categories}
        length = 0
        for line in reader:
            length += 1
            terms = line.strip().split()
            terms = np.array([float(term) for term in terms])
            if np.max(terms)-np.min(terms) != 0:
                terms =  (terms - np.min(terms))/ (np.max(terms) - np.min(terms))
            for i, term in enumerate(terms):
                attributes[categories[i]].append(term)
        attributes = {key:np.array(val) for key,val in attributes.items() }
        return attributes, length

