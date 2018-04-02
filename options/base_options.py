import argparse
import os 
import torch
from util import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Training Parameters Setting.')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_root', type=str,  help='path to sketch image pair dataset should have corresponding parser')
        self.parser.add_argument('--annotation_root', type=str, default='/home/lhy/datasets/coco2017/annotations',  help='path to annotation_root')
        self.parser.add_argument('--retrieval_now',action='store_true', help='Retrieval result when training?')
        self.parser.add_argument('--retrieval_once',action='store_true', help='Retrieval result one time?')
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--print_val_freq', type=int, default=10000, help='frequency of showing training results on console')
        self.parser.add_argument('--n_labels', type=int,required=True,  help='the number of classes')
        self.parser.add_argument('--n_fg_labels', type=int,required=True,  help='the number of classes')
        self.parser.add_argument('--n_attrs', type=int,required=True,  help='the number of attribute')
        self.parser.add_argument('--sketch_root', type=str, help='path to sketch dataset should be paired as image data by use number')
        self.parser.add_argument('--image_root', type=str, help='path to image dataset should be paried as sketch data by number')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu_ids e.g. 0 | 0,1 | 1,2,3')
        self.parser.add_argument('--phase', type=str, required=True, help='phase train, test, val')
        self.parser.add_argument('--batch_size', type=int, default=50, help='batch size for training')
        self.parser.add_argument('--topk', type=str, default='1,5,10', help='the option for retrieval result, shoe top k result')
        self.parser.add_argument('--pair_num', type=str, default='5,5', help='pair number for generating triplet training data')
        self.parser.add_argument('--name', type=str, default='experiment', help='Experiment name of this case')
        self.parser.add_argument('--feature_model', type=str, default='densenet169', help='The model for extracting feature of the data for retrieval')
        self.parser.add_argument('--model', type=str, default='denselosssiamese', help='The model for for retrieval')
        self.parser.add_argument('--n_threads', type=int, default=1, help='Threads for loading dataset')
        self.parser.add_argument('--scale_size', type=int, default=224, help='Scale Size for the image, resize')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Folder for saving models')
        self.parser.add_argument('--no_cuda', action='store_true', help='Choose whether use cuda')
        self.parser.add_argument('--edge_map', action='store_true', help='Choose Whether use the edge map of image data')
        self.parser.add_argument('--sketch_levels', type=str,default='c', help='The sketch level mode used')
        self.parser.add_argument('--feat_size', type=int,default=64, help='The size of embedding feature')
        self.parser.add_argument('--is_relu', action='store_true', help='Choose Whether the first Conv Layer use relu')
        self.parser.add_argument('--is_bn', action='store_true', help='Choose Whether the first Conv Layer use batchnorm')
        self.parser.add_argument('--attention_mode', action='store_true', help='Whether use attention in embedding')
        self.parser.add_argument('--fc_layer_mode',action='store_true', help='Whether use fc_layer in embedding mode')
        self.parser.add_argument('--fusion_mode', action='store_true', help='Whether use fusion mode')
        self.parser.add_argument('--save_mode', action='store_true', default=True, help='Whether save embedded feature')
        self.parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type')
        self.parser.add_argument('--image_type', type=str, default='EDGE', help='Image type')
        self.parser.add_argument('--sketch_type', type=str, default='GRAY', help='Sketch type')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--margin', type=float, default=5.0, help='margin for triplet loss parameter')
        self.parser.add_argument('--distance_type', type=str, default='euclidean', help='distance function in final retrieval ranking')
        self.parser.add_argument('--cnn_block', type=str, default='sketchanet', help='CNN Block')
        self.parser.add_argument('--task', type=str, default='fg_sbir', help='Task Type')
        self.parser.add_argument('--weight_decay', type=float , default=0.005, help='weight decay rate for regularization')
        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate for training')
        
        #self.initialized = True
        
        #self.update()

    def update(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        self.expression = ''
        self.expression += ('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            self.expression += ('%s: %s \n' % (str(k), str(v)))
        self.expression += ('-------------- End ----------------\n')
    
    def __str__(self):
        if not self.initialized:
            return 'Option not initialized'      
        return self.expression

    def save_to_file(self):
        args = vars(self.opt)

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    def parse_specific_args(self):
        self.opt.topk = tuple(int(k) for k in self.opt.topk.split(','))
        self.opt.pair_num = tuple(int(num) for num in self.opt.pair_num.split(','))

    def parse(self):
        if not self.initialized:
            self.initialize()
        if self.opt.phase == 'train':
            self.opt.is_train = True
        else:
            self.opt.is_train = False
        # set gpu ids
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
        if len(self.opt.gpu_ids) > 0:
            self.opt.cuda = True
        else:
            self.opt.cuda = False
        # other parameters parse
        self.parse_specific_args()
        
        # save to the disk
        self.save_to_file()
        return self.opt
