from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb
import os,re,json
import cv2
from PIL import Image
class TUBerlinDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        root = opt.data_root

        mode = opt.phase
        self.mode = mode

        transforms_list = []
        if self.opt.random_crop:
            transforms_list.append(transforms.Resize((300,300)))
            transforms_list.append(transforms.RandomCrop((self.opt.scale_size, self.opt.scale_size)))
        if self.opt.flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        self.transform_fun = transforms.Compose(transforms_list)
        self.test_transform_fun = transforms.Compose([transforms.Resize((self.opt.scale_size, self.opt.scale_size)), transforms.ToTensor()])

        
        self.sketch_imgs = []
        self.fg_labels = []
        self.labels = []

        if mode == "train":
            start, end = 0, 95
        elif mode == 'test':
            start, end = 95, 100

        for cls_root, subFolders, files in os.walk(root):
            photo_pat = re.compile("n.+\.JPEG")
            photo_imgs = list(filter(lambda fname:photo_pat.match(fname), files))
            if len(photo_imgs) == 0:
                print(cls_root)
                continue

            for i, photo_img in enumerate(photo_imgs, start=0):
                if i < start or i >= end:
                    continue
                img_path = os.path.join(root, cls_root, photo_img)
                self.sketch_imgs.append(img_path)
                
                self.fg_labels.append(fg_label)
                self.labels.append(label)
                fg_label += 1
            label += 1
        print("Total Sketchy:",label)
        self.n_labels = label
        self.n_fg_labels = fg_label
        print("{} pairs loaded.".format(len(self.photo_imgs)))

        print("{} pairs loaded. After generate triplet".format(len(self.photo_imgs)))

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


    def __len__(self):
        return len(self.photo_imgs)

    def __getitem__(self,index):
        sketch_img, fg_label, label =  self.sketch_imgs[index], self.fg_labels[index], self.labels[index]
        sketch_pil = Image.open(sketch_img)

        sketch_pil = self.load_sketch(sketch_pil)
        
        return sketch_pil, sketch_pil, sketch_pil, label, fg_label,label