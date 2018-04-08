import os
import time
from torchvision import transforms
from options.train_options import TrainOptions
from options.test_options import TestOptions
from models.denseloss_model import *
from util.util import *
from datasets.base_dataset import CustomDatasetDataLoader
from models.base_model import create_model
import matplotlib.pyplot as plt
import re
# parameters setting

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


opt = TrainOptions().parse()
class Dataset():
    def read(opt):
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
            start, end = 0, 20
        elif mode == 'test':
            start, end = 20, 30
        if self.levels == "stack":
            self.levels = "s"

        # load pictures
        for root,subFolders,files in os.walk(self.root):
            photo_pat = re.compile("cropped_\w+.*\d+.*\.jpg")
            photo_imgs = list(filter(lambda fname:photo_pat.match(fname),files))
            if len(photo_imgs) == 0:
                #print(root)
                continue
            sketch_imgs=[]
            cls_name = root[root.rfind('/')+1:]
            for i, photo_img in enumerate(photo_imgs):
                digit = re.findall("\d+",photo_img)[0]
                if i >= start and i < end :
                    for level in self.levels:
                        for augment_type in augment_types:

                            flag = "_" if mode == "train" and augment_type != "" else ""
                            sketch_pat = re.compile("cropped_"+augment_type+flag+str(digit)+level+".*\.png")
                            sketch_imgs = list(filter(lambda fname:sketch_pat.match(fname),files))
                            #print(sketch_imgs)
                            for sketch_img in sketch_imgs:
                                
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
            label += 1


dataset = Dataset().read(opt)

retreival_result_file = 'retreival_result.txt'
retrieval_result = read_result(retreival_result_file)
path_result = []
plt.tight_layout()
for i, fig, result in enumerate(retrieval_result):
    plt.subplot(5, 11, 11*i)
    #tmp = (sketch_imgs[fig], [photo_imgs[photo_ind] for photo_ind in result])
    plt.imshow(dataset.sketch_imgs[fig])
    plt.axis('off') 
    for j, photo_ind in enumerate(result):
        plt.subplot(5,11, 11*i+j)
        plt.imshow(dataset.photo_imgs[photo_ind])
        plt.axis('off') 
plt.savefig('result.png')

def read_result(path):
    retrieval_result = {}
    with open(path, 'r') as reader:
        for line in reader:
            fig, result = line.strip().split(',')
            result = result[17:]
            result = result[1:len(result)-1].split(' ')
            result = [int(i) for i in result if i != '']
            retrieval_result[int(fig[3:])] = result




