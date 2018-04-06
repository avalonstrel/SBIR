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
        sketch_root = os.path.join(root, mode, "sketches")
        image_root = os.path.join(root, mode, "images")
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
        if 'chairs' in root:
            thing_type = 'chairs'
        else:
            thing_type = 'shoes'
        annotation_fn = os.path.join(root, "annotation/{}_annotation.json".format(thing_type))
        
        self.annotation_data = json.load(open(annotation_fn,"r"))
        self.num = len(self.annotation_data[mode]["sketches"])
 
        if thing_type == 'chairs':
            label_key = 'labels'
            self.offset = 1 if mode == "train" else 201

        
        elif thing_type == 'shoes':
            label_key = 'label'
            self.offset = 1 if mode == "train" else 305

        label_dat = sio.loadmat(os.path.join(root,"annotation/sketch_label.mat"))
        self.labels = np.array(label_dat[label_key])
        self.n_labels = self.labels.shape[1]
        self.attribute_size = self.n_labels
        self.attributes = np.array(label_dat[label_key])
        

        self.image_imgs = {}
        self.sketch_imgs = {}
        self.image_list = []
        self.sketch_list = []
        self.image_neg_imgs = []
        self.fg_labels = []
        j=0
        for root, subFolders, files in os.walk(sketch_root):
            sketch_pat = re.compile("\d+.png")
            sketch_imgs = list(filter(lambda fname:sketch_pat.match(fname), files))

            if len(sketch_imgs) == 0:
                continue
            for i, sketch_img in enumerate(sketch_imgs):
                digit = re.findall("\d+", sketch_img)[0]
                image_img = os.path.join(image_root, digit+".jpg")
                ind = int(digit)
                self.sketch_imgs[ind] = os.path.join(sketch_root, sketch_img)
                self.image_imgs[ind] = image_img #os.path.join(image_root, image_img)
                self.image_list.append(image_img)
                self.sketch_list.append(os.path.join(sketch_root, sketch_img))
                self.fg_labels.append(j)
            j+=1
        self.n_fg_labels = j
        self.ori_sketch_imgs = self.sketch_imgs.copy()
        self.ori_photo_imgs = self.image_imgs.copy()
        self.ori_labels = self.labels.copy()
        self.ori_fg_labels = self.fg_labels.copy()
        self.ori_attributes = self.attributes.copy()
        self.ori_sketch_list = self.sketch_list.copy()
        self.ori_image_list = self.image_list.copy()
        print("{} images loaded.".format(len(self.image_imgs)))
        
        # For generate triplet
        if self.query_what == "image":
            self.query_image()
        elif self.query_what == "sketch":
            self.query_sketch()
        if self.opt.phase == 'train':
            self.generate_triplet_all()
        print("Total Sketchy Class:{}, fg class: {}".format(self.n_labels, self.n_fg_labels))       
        print("{} images loaded. After generate triplet".format(len(self.query_imgs)))

    def generate_triplet_all(self):
        pair_inclass_num, pair_outclass_num = self.opt.pair_num
        if self.opt.triplet_type == 'annotation':
            self.generate_annotation_triplet()
        elif self.opt.triplet_type == 'random':
            if self.opt.task == 'fg_sbir':
                self.generate_triplet(pair_inclass_num, pair_outclass_num)
        

           
    def generate_triplet(self, pair_inclass_num, pair_outclass_num=0):
        query_imgs, search_neg_imgs, search_imgs,attributes, fg_labels, labels = [],[],[],[],[],[]

        

        num = len(self.search_list)
        for i, (query_img, search_img, fg_label, label,attribute) in enumerate(zip(self.query_list, self.search_list, self.fg_labels, self.labels,self.attributes)):
            inds = [i]
            for j in range(pair_outclass_num):
                ind = np.random.randint(num)
                while ind in inds:
                    ind = np.random.randint(num)
                inds.append(ind)
                query_imgs.append(query_img)
                search_neg_imgs.append(self.search_list[ind])
                search_imgs.append(search_img)
                fg_labels.append(fg_label)
                attributes.append(attribute)
                labels.append(label)


        self.query_imgs, self.search_neg_imgs, self.search_imgs, self.fg_labels, self.labels, self.attributes = query_imgs, search_neg_imgs, search_imgs, fg_labels, labels, attributes

    def generate_annotation_triplet(self):
        mode = self.opt.phase
        offset = self.offset
        query_imgs = []
        search_imgs = []
        search_neg_imgs = []
        labels = []
        fg_labels = []
        attributes = []

        for i, triplets in enumerate(self.annotation_data[mode]["triplets"]):
            triplets = triplets if self.opt.phase == "train" else [[i+offset-1,i+offset-1]]
            for j, triplet in enumerate(triplets):
                #if not self.opt.model == 'sphere_model' or j == 0:
                query_imgs.append(self.query_imgs[i+offset])
                search_imgs.append(self.search_imgs[triplet[0]+1])
                search_neg_imgs.append(self.search_imgs[triplet[1]+1])
                labels.append(self.labels[i].argmax())
                fg_labels.append(i)
                attributes.append(self.attributes[i+offset-1])
        
        self.query_imgs, self.search_imgs, self.search_neg_imgs, self.labels, self.fg_labels, self.attributes = query_imgs, search_imgs, search_neg_imgs, labels, fg_labels, attributes
        self.n_fg_labels = i + 1
        self.n_labels = 15
    def query_image(self):
        self.query_imgs = self.ori_sketch_imgs.copy()
        self.search_imgs = self.ori_photo_imgs.copy()
        self.search_neg_imgs = self.ori_photo_imgs.copy()
        self.labels = self.ori_labels.copy()
        self.fg_labels = self.ori_fg_labels.copy()
        self.query_list = self.ori_sketch_list.copy()
        self.search_list = self.ori_image_list.copy()
        self.load_search = self.load_image
        self.load_query = self.load_sketch
        print("Query is Sketch Search Image")
    def query_sketch(self):
        self.query_imgs = self.ori_photo_imgs.copy()
        self.search_imgs = self.ori_sketch_imgs.copy()
        self.search_neg_imgs = self.ori_sketch_imgs.copy()
        self.labels = self.ori_labels.copy()
        self.fg_labels = self.ori_fg_labels.copy()
        self.query_list = self.ori_image_list.copy()
        self.search_list = self.ori_sketch_list.copy()        
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
            pil_numpy = pil_numpy
        #print('image{}'.format(pil_numpy.shape))
        #if self.opt.image_type == 'GRAY' or self.opt.image_type == 'EDGE':
        #    pil_numpy = pil_numpy.reshape(pil_numpy.shape + (1,))
        #pil_numpy = cv2.resize(pil_numpy, (self.opt.scale_size, self.opt.scale_size))
        #if self.opt.sketch_type == 'GRAY' or self.opt.image_type == 'EDGE':
        #    pil_numpy = pil_numpy.reshape(pil_numpy.shape[:2])
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

        if self.opt.sketch_type == 'RGB':
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
            #pil_numpy = np.tile(pil_numpy,3).reshape(pil_numpy.shape+(-1,))
        elif pil_numpy.shape[2] == 4:
            pil_numpy = to_rgb(pil_numpy[:,:,3])
            #pil_numpy = np.tile(pil_numpy[:,:,3],3).reshape(pil_numpy.shape[0:2]+(-1,))
            #pil_numpy[:,:,2] = 0
        pil_numpy = cv2.resize(pil_numpy,(self.opt.scale_size, self.opt.scale_size))

        if self.transform_fun is not None:
            pil = Image.fromarray(pil_numpy)
            pil_numpy = self.transform_fun(pil)
        #data_info.write(",".join([str(i) for i in pil_numpy.numpy().flatten() if i != 0])+"\n")
        return pil_numpy
    def __len__(self):
        return len(self.query_imgs)

    def __getitem__(self,index):
        #print(len(self.attributes),"image",len(self.image_imgs),"ind:",index)
        search_img,query_img,search_neg_img,fg_label,label, attribute = self.search_imgs[index], self.query_imgs[index], self.search_neg_imgs[index], self.fg_labels[index], self.labels[index], self.attributes[index]
        if self.levels == "stack":
            query_s_pil, query_c_pil = self.load_query(Image.open(query_img[0])), self.load_query(Image.open(query_img[1]))
            query_s_pil[:,:,1] = query_c_pil[:,:,0]
            query_pil = query_s_pil
        else:
            query_pil = Image.open(query_img)
            query_pil = self.load_query(query_pil)
        search_pil, search_neg_pil = Image.open(search_img), Image.open(search_neg_img)

        #if self.transform is not None:
        search_pil = self.load_search(search_pil)
        search_neg_pil = self.load_search(search_neg_pil)
        #print(search_pil)
        #print(query_pil.size())
        return query_pil, search_pil, search_neg_pil, attribute, fg_label, label
        
