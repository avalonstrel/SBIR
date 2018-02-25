from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb
import os,re
class SketchyDataset(data.Dataset):
    def __init__(self, opt)# root,photo_types,sketch_types,batch_size, mode="train", train_split=2000,  pair_inclass_num=2,pair_outclass_num=3):
        root = opt.data_root
        photo_types = opt.photo_types
        sketch_types = opt.sketch_types
        mode = opt.phase
        
        self.photo_imgs = []
        self.sketch_imgs = []
        self.photo_neg_imgs = []
        self.transform_fun = transforms.Compose([transforms.ToTensor()])
        self.fg_labels = []
        self.labels = []
        if mode == "train":
            start, end = 0, 90
        elif mode == 'test':
            start, end = 90, 95
        photo_roots = [root+photo_type for photo_type in photo_types]
        print(photo_roots)
        for photo_root in photo_roots:
            print(photo_root)
            fg_label = 0
            label = 0
            for photo_root, subFolders, files in os.walk(photo_root):
                photo_pat = re.compile("n.+\.jpg")
                photo_imgs = list(filter(lambda fname:photo_pat.fullmatch(fname),files))
                if len(photo_imgs) == 0:
                    print(photo_root)
                    continue
                sketch_imgs = []
                for i, photo_img in enumerate(photo_imgs):
                    if i < start or i >= end:
                        continue
                    for sketch_type in sketch_types:
                        sketch_root = photo_root[:photo_root.find("photo")] + "sketch/"+sketch_type+"/"+photo_root[photo_root.rfind("/")+1:]
                        for i in range(1,20):
                            sketch_img = photo_img[:photo_img.find(".jpg")] + "-" + str(i) + ".png"
                            sketch_path = os.path.join(sketch_root, sketch_img)
                            if os.path.exists(sketch_path):
                                self.photo_imgs.append(os.path.join(photo_root,photo_img))
                                self.sketch_imgs.append(sketch_path)
                                self.photo_neg_imgs.append(os.path.join(photo_root,photo_img))
                                self.fg_labels.append(fg_label)
                                self.labels.append(label)
                            else:
                                break
                    fg_label += 1
                label += 1
        print("Total Sketchy:",label)
        self.n_labels = label
        self.n_fg_labels = fg_label
        print("{} pairs loaded.".format(len(self.photo_imgs)))
        pair_inclass_num, pair_outclass_num = self.opt.pair_num
        if mode == "train" :
            self.generate_triplet(pair_inclass_num,pair_outclass_num)
        if mode == "test":
            self.fg_labels = []
            for i in range(len(self.photo_imgs)):
                self.fg_labels.append(i % batch_size)


        print("{} pairs loaded. After generate triplet".format(len(self.photo_imgs)))

    def transform(self, pil):
        pil_numpy = np.array(pil)
        if len(pil_numpy.shape) == 2:
            pil_numpy = to_rgb(pil_numpy)
            #pil_numpy = np.tile(pil_numpy,3).reshape(pil_numpy.shape+(-1,))
        elif pil_numpy.shape[2] == 4:
            pil_numpy = to_rgb(pil_numpy[:,:,3])
            #pil_numpy = np.tile(pil_numpy[:,:,3],3).reshape(pil_numpy.shape[0:2]+(-1,))

        pil_numpy = cv2.resize(pil_numpy,(resize_size,resize_size))
        if self.transform_fun is not None:
            pil_numpy = self.transform_fun(pil_numpy)
        return pil_numpy
    def __len__(self):
        return len(self.photo_imgs)

    def __getitem__(self,index):
        photo_img,sketch_img,photo_neg_img,fg_label,label = self.photo_imgs[index], self.sketch_imgs[index], self.photo_neg_imgs[index], self.fg_labels[index], self.labels[index]
        photo_pil,sketch_pil,photo_neg_pil = Image.open(photo_img), Image.open(sketch_img), Image.open(photo_neg_img)
        #if self.transform is not None:
        photo_pil = self.transform(photo_pil)
        sketch_pil = self.transform(sketch_pil)
        photo_neg_pil = self.transform(photo_neg_pil)
        return sketch_pil,photo_pil,photo_neg_pil, label, fg_label,label


    def generate_triplet(self, pair_inclass_num,pair_outclass_num=0):
        sketch_imgs, photo_neg_imgs, photo_imgs, fg_labels, labels = [],[],[],[],[]

        labels_dict = [[] for i in range(self.n_labels)]
        for i, label in enumerate(self.labels):
           # print(label)
            labels_dict[label].append(i)
        fg_labels_dict = [[] for i in range(self.n_fg_labels)]
        for i, fg_label in enumerate(self.fg_labels):
           # print(fg_label)
            fg_labels_dict[fg_label].append(i)

        for i, (sketch_img, photo_img, fg_label, label) in enumerate(zip(self.sketch_imgs, self.photo_imgs, self.fg_labels, self.labels)):
            num = len(labels_dict[label])
            inds = [labels_dict[label].index(i)]
            for j in range(pair_inclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                sketch_imgs.append(sketch_img)
                photo_neg_imgs.append(self.photo_imgs[labels_dict[label][ind]])
                photo_imgs.append(photo_img)
                fg_labels.append(fg_label)
                labels.append(label)

        num = len(self.photo_imgs)
        for i, (sketch_img, photo_img, fg_label, label) in enumerate(zip(self.sketch_imgs, self.photo_imgs, self.fg_labels, self.labels)):
            inds = [i]
            for j in range(pair_outclass_num):
                ind = np.random.randint(num)
                while ind in inds or ind in fg_labels_dict[fg_label] or ind in labels_dict[label]:
                    ind = np.random.randint(num)
                inds.append(ind)
                sketch_imgs.append(sketch_img)
                photo_neg_imgs.append(self.photo_imgs[ind])
                photo_imgs.append(photo_img)
                fg_labels.append(fg_label)
                labels.append(label)

        self.sketch_imgs, self.photo_neg_imgs, self.photo_imgs, self.fg_labels, self.labels = sketch_imgs, photo_neg_imgs, photo_imgs, fg_labels, labels
