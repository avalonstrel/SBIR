from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb
import os, re
from PIL import Image
import json
import cv2 
"""Hairstyle Dataset"""
class HairDataset(data.Dataset):
    def name(self):
        return "hairstyle"
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
        self.photo_imgs = []
        self.sketch_imgs = []
        self.photo_neg_imgs = []
        self.fg_labels = []
        self.labels = []

        label = 0
        fg_label = 0
        if mode == "train":
            start, end = 0, 24
        elif mode == 'test':
            start, end = 24, 30
        if self.levels == "stack":
            self.levels = "s"

        # load pictures
        for root,subFolders,files in os.walk(self.root):
            photo_pat = re.compile("\w+.*\d+.*\.jpg")
            photo_imgs = list(filter(lambda fname:photo_pat.match(fname),files))
            if len(photo_imgs) == 0:
                #print(root)
                continue
            sketch_imgs=[]
            cls_name = root[root.rfind('/')+1:]
            #print(len(photo_imgs))
            for i, photo_img in enumerate(photo_imgs):
                digit = re.findall("\d+", photo_img)[0]
                if i >= start and i < end :
                    try:
                        for level in self.levels:

                            for augment_type in augment_types:

                                flag = "_" if mode == "train" and augment_type != "" else ""
                                sketch_pat = re.compile(augment_type+flag+str(digit)+level+".*\.png")
                                sketch_imgs = list(filter(lambda fname:sketch_pat.match(fname),files))
                                
                                for sketch_img in sketch_imgs:
                                
                                    Image.open(os.path.join(root,photo_img)).convert('L')
                                    Image.open(os.path.join(root,photo_img)).convert('RGB')
                                    Image.open(os.path.join(root,sketch_img)).convert('L')
                                    Image.open(os.path.join(root,sketch_img)).convert('RGB')
                                    self.photo_imgs.append(os.path.join(root,photo_img))
                                    if self.levels == "stack":
                                        sketch_other_img = sketch_img.replace("s.","c.")
                                        sketch_ohter_img = sketch_other_img.replace("s_","c_")
                                        self.sketch_imgs.append([os.path.join(root,sketch_img),os.path.join(root,sketch_other_img)])
                                    else:

                                        self.sketch_imgs.append(os.path.join(root,sketch_img))
                                    self.photo_neg_imgs.append(os.path.join(root,photo_img))
                                    
                                    self.attributes.append(self.attributes_dict[cls_name])
                                    self.fg_labels.append(fg_label)
                                    self.labels.append(label)
                        fg_label += 1
                    except:
                        print(photo_img,'is truncated in loading')
            label += 1
        print(self.photo_imgs[0])
        self.ori_photo_imgs  = self.photo_imgs.copy()
        self.ori_sketch_imgs = self.sketch_imgs.copy()
        self.ori_labels = self.labels.copy()
        self.ori_fg_labels = self.fg_labels.copy()
        self.ori_attributes = self.attributes.copy()
        self.n_labels = label
        self.n_fg_labels = fg_label
        self.query_what = self.opt.query_what
        print("Total Sketchy Class:{}, fg class: {}".format(self.n_labels, self.n_fg_labels))
        print("FG TOTAL:",fg_label,len(self.photo_imgs))
        print("{} images loaded.".format(len(self.photo_imgs)))
        self.labels_dict = {i:[] for i in range(self.n_labels)}
        for i, label in enumerate(self.labels):
            self.labels_dict[label].append(i)
        self.fg_labels_dict = {i:[] for i in range(self.n_fg_labels)}
        for i, fg_label in enumerate(self.fg_labels):
            self.fg_labels_dict[fg_label].append(i)

        if self.query_what == "image":
            self.query_image()
        elif self.query_what == "sketch":
            self.query_sketch()  
        self.generate_triplet_all()
        print("FG TOTAL:",fg_label,len(self.query_imgs))
        print("{} images loaded.".format(len(self.query_imgs))) 

    def __len__(self):
        return len(self.query_imgs)

    def generate_triplet_all(self):
        pair_inclass_num, pair_outclass_num = self.opt.pair_num        
        if self.opt.phase == "train" and not self.opt.neg_flag == "moderate":
            if self.opt.task == 'fg_sbir':
                self.generate_triplet(pair_inclass_num,pair_outclass_num)

            elif self.opt.task == 'cate_sbir':
                self.generate_cate_triplet(pair_inclass_num,pair_inclass_num)  
            elif self.opt.task == 'fg_sbir_cate_pos':
                self.generate_triplet_cate_pos(pair_inclass_num,pair_inclass_num)
    def query_image(self):
        self.query_imgs = self.ori_sketch_imgs.copy()
        self.search_imgs = self.ori_photo_imgs.copy()
        self.search_neg_imgs = self.ori_photo_imgs.copy()
        self.labels = self.ori_labels.copy()
        self.fg_labels = self.ori_fg_labels.copy()
        self.attributes = self.ori_attributes.copy()
        print("Query is Sketch Search Image")
    def query_sketch(self):
        self.query_imgs = self.ori_photo_imgs.copy()
        self.search_imgs = self.ori_sketch_imgs.copy()
        self.search_neg_imgs = self.ori_sketch_imgs.copy()
        self.labels = self.ori_labels.copy()
        self.fg_labels = self.ori_fg_labels.copy()
        self.attributes = self.ori_attributes.copy()
        print("Query is Image Search Sketch")
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
        #print(len(self.attributes),"photo",len(self.photo_imgs),"ind:",index)
        search_img,query_img,search_neg_img,fg_label,label,attribute = self.search_imgs[index], self.query_imgs[index], self.search_neg_imgs[index], self.fg_labels[index], self.labels[index], self.attributes[index]
        open_type = "L" if self.edge_map else "RGB"

        if self.levels == "stack":
            query_s_pil, query_c_pil = self.transform(Image.open(query_img[0])), self.transform(Image.open(query_img[1]))
            query_s_pil[:,:,1] = query_c_pil[:,:,0]
            query_pil = query_s_pil
        else:
            query_pil = Image.open(query_img)
            #print("query", np.array(query_pil).shape)
            query_pil = self.transform(query_pil)

        search_pil, search_neg_pil = Image.open(search_img).convert(open_type),  Image.open(search_neg_img).convert(open_type)
        #print("search", np.array(search_pil).shape)
        #if self.transform is not None:
        search_pil = self.transform(search_pil, "image")
        search_neg_pil = self.transform(search_neg_pil, "image")

        return query_pil, search_pil,search_neg_pil,attribute, fg_label,label
    def generate_cate_triplet(self, pair_inclass_num, pair_outclass_num):
        query_imgs, search_imgs, search_neg_imgs, attributes, fg_labels, labels = [],[],[],[],[]

        labels_dict = [[] for i in range(self.n_labels)]
        for i, label in enumerate(self.labels):
            labels_dict[label].append(i)

        for i, (query_img, search_img, attribute, fg_label, label) in enumerate(zip(self.query_imgs, self.search_imgs, self.attributes, self.fg_labels, self.labels)):

            
            for t, l in enumerate(labels_dict[label]):
                if l != i and t < pair_inclass_num:
                    for j in range(pair_outclass_num):
                        ind_label = np.random.randint(self.n_labels)
                        while ind_label == label:

                            ind_label = np.random.randint(self.n_labels)
                        #print(ind_label)
                        ind = np.random.randint(len(labels_dict[ind_label]))
                        
                        query_imgs.append(query_img)
                        search_imgs.append(self.search_imgs[l])
                        attributes.append(attribute)
                        search_neg_imgs.append(self.search_imgs[labels_dict[ind_label][ind]])
                        fg_labels.append(fg_label)
                        labels.append(label)

        self.query_imgs, self.search_neg_imgs, self.search_imgs, self.attributes, self.fg_labels, self.labels = query_imgs, search_neg_imgs, search_imgs, attributes, fg_labels, labels

    def generate_triplet_cate_pos(self, pair_inclass_num, pair_outclass_num):
        query_imgs, search_neg_imgs, search_imgs,attributes, fg_labels, labels = [],[],[],[],[],[]

        labels_dict = [[] for i in range(self.n_labels)]
        for i, label in enumerate(self.labels):
            labels_dict[label].append(i)
        fg_labels_dict = [[] for i in range(self.n_fg_labels)]
        for i, fg_label in enumerate(self.fg_labels):
            fg_labels_dict[fg_label].append(i)

        for i, (query_img, search_img, fg_label, label,attribute) in enumerate(zip(self.query_imgs, self.search_imgs, self.fg_labels, self.labels, self.attributes)):
            num = len(labels_dict[label])
            inds = [labels_dict[label].index(i)]
            for j in range(pair_inclass_num):
                num = len(labels_dict[label])
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                query_imgs.append(query_img)
                search_neg_imgs.append(self.search_imgs[labels_dict[label][ind]])
                search_imgs.append(search_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)

                cate_ind = np.random.randint(num)
                while cate_ind in inds or cate_ind in fg_labels_dict[fg_label]:
                    cate_ind = np.random.randint(num)
                num = len(self.search_imgs)
                fg_inds = [labels_dict[label].index(i)]
                for k in range(pair_outclass_num):
                    fg_ind = np.random.randint(num)
                    while fg_ind in fg_inds or fg_ind in fg_labels_dict[fg_label] or fg_ind in labels_dict[label]:
                        fg_ind = np.random.randint(num)
                    fg_inds.append(fg_ind)
                    query_imgs.append(query_img)
                    search_neg_imgs.append(self.search_imgs[fg_ind])
                    search_imgs.append(self.search_imgs[cate_ind])
                    fg_labels.append(fg_label)
                    attributes.append(attribute)
                    labels.append(label)
        num = len(self.search_imgs)
        for i, (query_img, search_img, fg_label, label,attribute) in enumerate(zip(self.query_imgs, self.search_imgs, self.fg_labels, self.labels,self.attributes)):
            inds = [labels_dict[label].index(i)]
            for j in range(pair_outclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label] or ind in labels_dict[label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                query_imgs.append(query_img)
                search_neg_imgs.append(self.search_imgs[ind])
                search_imgs.append(search_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)
        self.query_imgs, self.search_neg_imgs, self.search_imgs, self.fg_labels, self.labels, self.attributes = query_imgs, search_neg_imgs, search_imgs, fg_labels, labels, attributes
           
    def generate_triplet(self, pair_inclass_num, pair_outclass_num=0):
        query_imgs, search_neg_imgs, search_imgs,attributes, fg_labels, labels = [],[],[],[],[],[]

        labels_dict = [[] for i in range(self.n_labels)]
        for i, label in enumerate(self.labels):
            labels_dict[label].append(i)
        fg_labels_dict = [[] for i in range(self.n_fg_labels)]
        for i, fg_label in enumerate(self.fg_labels):
            fg_labels_dict[fg_label].append(i)

        for i, (query_img, search_img, fg_label, label,attribute) in enumerate(zip(self.query_imgs, self.search_imgs, self.fg_labels, self.labels, self.attributes)):
            num = len(labels_dict[label])
            inds = [labels_dict[label].index(i)]
            for j in range(pair_inclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                query_imgs.append(query_img)
                search_neg_imgs.append(self.search_imgs[labels_dict[label][ind]])
                search_imgs.append(search_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)
        num = len(self.search_imgs)
        for i, (query_img, search_img, fg_label, label,attribute) in enumerate(zip(self.query_imgs, self.search_imgs, self.fg_labels, self.labels,self.attributes)):
            inds = [labels_dict[label].index(i)]
            for j in range(pair_outclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label] or ind in labels_dict[label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                query_imgs.append(query_img)
                search_neg_imgs.append(self.search_imgs[ind])
                search_imgs.append(search_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)


        self.query_imgs, self.search_neg_imgs, self.search_imgs, self.fg_labels, self.labels, self.attributes = query_imgs, search_neg_imgs, search_imgs, fg_labels, labels, attributes

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

