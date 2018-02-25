from torch.utils import data
import numpy as np
from torchvision import transforms
from util.util import to_rgb
import os, re

#SketchX dataset
class SketchXDataset(data.Dataset):
    def __init__(self, opt)# root,thing_type="chairs",levels="cs", mode="train", flag="two_loss"):

        # Parameters Setting
        root = opt.data_root


        mode = opt.phase
        sketch_root = os.path.join(root, mode, "sketches")
        image_root = os.path.join(root, mode, "images")

        self.flag = opt.loss_flag
        self.levels = levels
        annotation_fn = os.path.join(root, "annotation/{}_annotation.json".format(thing_type))
        
        self.annotation_data = json.load(open(annotation_fn,"r"))
        self.num = len(self.annotation_data[mode]["sketches"])
        self.transform_fun = transforms.Compose([transforms.ToTensor()])
        
        if 'chairs' in self.root:
            thing_type = 'chairs'
        else:
            thing_type = 'shoes'
        if thing_type == 'chairs':
            label_key = 'labels'
            offset = 1 if mode == "train" else 201
        
        elif thing_type == 'shoes':
            label_key = 'label'
            offset = 1 if mode == "train" else 305

        label_dat = sio.loadmat(os.path.join(root,"annotation/sketch_label.mat"))
        self.labels = np.array(label_dat[label_key])
        self.n_labels = self.labels.shape[1]
        self.attribute_size = self.n_labels
        self.attributes = np.array(label_dat[label_key])
        

        self.image_imgs = {}
        self.sketch_imgs = {}
        self.image_neg_imgs = []
        self.fg_labels = []

        for root, subFolders, files in os.walk(sketch_root):
            sketch_pat = re.compile("\d+.png")
            sketch_imgs = list(filter(lambda fname:sketch_pat.fullmatch(fname), files))
            if len(sketch_imgs) == 0:
                continue
            for i, sketch_img in enumerate(sketch_imgs):
                digit = re.findall("\d+", sketch_img)[0]
                image_img = os.path.join(image_root, digit+".jpg")
                ind = int(digit)
                self.sketch_imgs[ind] = os.path.join(sketch_root, sketch_img)
                self.image_imgs[ind] = image_img #os.path.join(image_root, image_img)

        print("{} images loaded.".format(len(self.image_imgs)))
        
        # For generate triplet
        sketch_imgs = []
        image_imgs = []
        image_neg_imgs = []
        labels = []
        fg_labels = []
        attributes = []

        for i, triplets in enumerate(self.annotation_data[mode]["triplets"]):
            triplets = triplets if mode == "train" else [[i+offset-1,i+offset-1]]
            for triplet in triplets:
                sketch_imgs.append(self.sketch_imgs[i+offset])
                image_imgs.append(self.image_imgs[triplet[0]+1])
                image_neg_imgs.append(self.image_imgs[triplet[1]+1])
                labels.append(self.labels[i].argmax())
                fg_labels.append(i)
                attributes.append(self.attributes[i+offset-1])
        self.sketch_imgs, self.image_imgs, self.image_neg_imgs, self.labels, self.fg_labels, self.attributes = sketch_imgs, image_imgs, image_neg_imgs, labels, fg_labels, attributes
        self.n_fg_labels = len(sketch_imgs)
        print("{} images loaded. After generate triplet".format(len(self.image_imgs)))

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
        pil_numpy = cv2.resize(pil_numpy,(resize_size,resize_size))
        if self.transform_fun is not None:
            pil_numpy = self.transform_fun(pil_numpy)
        #data_info.write(",".join([str(i) for i in pil_numpy.numpy().flatten() if i != 0])+"\n")
        return pil_numpy

    def __len__(self):
        return len(self.image_imgs)

    def __getitem__(self,index):
        #print(len(self.attributes),"image",len(self.image_imgs),"ind:",index)
        image_img,sketch_img,image_neg_img,fg_label,label, attribute = self.image_imgs[index], self.sketch_imgs[index], self.image_neg_imgs[index], self.fg_labels[index], self.labels[index], self.attributes[index]
        if self.levels == "stack":
            sketch_s_pil, sketch_c_pil = self.transform(Image.open(sketch_img[0])), self.transform(Image.open(sketch_img[1]))
            sketch_s_pil[:,:,1] = sketch_c_pil[:,:,0]
            sketch_pil = sketch_s_pil
        else:
            sketch_pil = Image.open(sketch_img)
            
            sketch_pil = self.transform(sketch_pil)
        image_pil, image_neg_pil = Image.open(image_img), Image.open(image_neg_img)

        #if self.transform is not None:
        image_pil = self.transform(image_pil)
        image_neg_pil = self.transform(image_neg_pil)
        if self.flag == "two_loss":
            return sketch_pil, image_pil, image_neg_pil, fg_label,label
        elif self.flag == "three_loss":

            return sketch_pil, image_pil, image_neg_pil, attribute, fg_label,label
