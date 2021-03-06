import torch
import numpy as np
import os
from util.util import mkdir
def create_model(opt):
    name = opt.model
    if name == 'denselosssiamese':
        from .denseloss_model import DenseLossModel
        return DenseLossModel(opt)
    elif name in ['tripletsiamese', 'tripletheter']:
        from .triplet_model import TripletModel 
        return TripletModel(opt)
    elif name == 'cls_model':
        from .classification_model import ClassificationModel
        return ClassificationModel(opt)
    elif name == 'sphere_model':
        from .sphere_model import SphereModel
        return SphereModel(opt)

    return None


class BaseModel():
    
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.is_train
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.optimize_modules = []
        mkdir(self.save_dir)
        self.initialize()

    def initialize(self):
        raise NotImplementedError("not implement error")

    def optimize(self):
        raise NotImplementedError("not implement error")


    def test(self):
        raise NotImplementedError("not implement error")


    def save_model(self):
        raise NotImplementedError("not implement error")


    def load_model(self, model):
        raise NotImplementedError("not implement error")

    def train(self, mode=True):
        for module in self.optimize_modules:
            module.train(mode)

    def parallel(self):
        for i in range(len(self.optimize_modules)):
            pass#self.optimize_modules[i] = torch.nn.DataParallel(self.optimize_modules[i])

    def cuda(self):
        for i in range(len(self.optimize_modules)):
            self.optimize_modules[i].cuda()
    def cpu(self):
        for i in range(len(self.optimize_modules)):
            self.optimize_modules[i].cpu()
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth.tar' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()#if len(gpu_ids) and torch.cuda.is_available():
        #    network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, load_path=None):
        save_filename = '%s_net_%s.pth.tar' % (epoch_label, network_label)
        if not load_path is None:
            save_path = os.path.join(load_path, save_filename)
        else:
            save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            network.load_state_dict(torch.load(save_path))
        else:
            print("No loading path...")
