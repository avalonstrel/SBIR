import torch
import numpy as np
import os
from .base_model import BaseModel
from .networks import *
from .loss_utils import *
from util.evaluation import *
from util.util import *
from torch.autograd import Variable
from torchvision import models
class ClassificationModel(BaseModel):

    def name(self):
        return 'TripletModel'
    def get_loss(self, loss_type):
        if loss_type == 'triplet':
            return torch.nn.TripletMarginLoss(self.opt.margin)
        elif loss_type == 'holef':
            return HOLEFLoss(self.opt)
        return None

    def initialize(self):
        self.network = AttentionNetwork(self.opt)
        self.network = torch.nn.DataParallel(self.network)
        self.cls_network = ClassificationNetwork(self.opt.feat_size, self.opt.n_labels)
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.optimize_modules = [self.network, self.cls_network]
        #self.loss = torch.nn.DataParallel(self.loss)
        self.result_record = {'total':self.record_initialize(True)}
        #self.result_record['total'] = self.record_initialize(True)
        self.test_result_record = self.copy_initialize_record(self.result_record)
        
        self.optimizer = torch.optim.Adam([{"params":module.parameters()} for module in self.optimize_modules], lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        
        self.reset_features()
        self.reset_test_features()
        if len(self.opt.gpu_ids) > 1:
            self.parallel()
            print('Model parallel...')
            self.cuda()
            print('Model cuda ing...')
        elif self.opt.cuda:
            self.cuda()
            print('Model cuda ing...')
        if self.opt.continue_train:
            self.load_model(self.opt.start_epoch_label, self.opt.trained_model_path)


    def reset_records(self):
        self.result_record = self.copy_initialize_record(self.result_record)

    def reset_test_records(self):
        self.test_result_record = self.copy_initialize_record(self.test_result_record)
    
    def record_initialize(self, have_accs):
        if have_accs:
            return {'loss_value':AverageMeter(),'acc':{k:AverageMeter() for k in self.opt.topk}}
        else:
            return {'loss_value':AverageMeter()}

    def copy_initialize_record(self, record):
        result_record = {}
        for key, r in record.items():
            if len(r) == 2:
                result_record[key] = self.record_initialize(True)
            else:
                result_record[key] = self.record_initialize(False)
        return result_record

    # Record training data during training
    def update_record(self, result_record, key, loss, size, prediction=None, labels=None, accs=None ):
        if isinstance(loss, float):
            result_record[key]['loss_value'].update(loss, size)
        else:
            result_record[key]['loss_value'].update(loss.data[0], size)
        if accs != None:
            
            for i, topk in enumerate(self.opt.topk):
                if isinstance(accs, float):
                    result_record[key]['acc'][topk].update(accs, size)  
                else:
                    result_record[key]['acc'][topk].update(accs[topk], size)  
   
        elif not prediction is None and (not labels is None) :
            res = accuracy(prediction, labels, self.opt.topk)
            #print(res)
            for i, topk in enumerate(self.opt.topk):
                result_record[key]['acc'][topk].update(res[topk].data[0], size)  

    def generate_message(self, result_record):
        messages = []
        for key, record in result_record.items():
            if 'acc' in record:
                tmp_message = '{}:{:.3f}, {}'.format(key, record['loss_value'].avg, accs_message(record['acc']))
            else:
                tmp_message = '{}:{:.3f}'.format(key, record['loss_value'].avg)
            messages.append(tmp_message)
        message = " | ".join(messages)
        return message

    def reset_features(self):
        self.features = {'edge':[], 'labels':[]}

    def reset_test_features(self):
        self.test_features = {'edge':[], 'labels':[]}

    def append_features(self, features, output0, labels):
        features['edge'].append(output0.data.cpu())
        features['labels'].append(labels.data.cpu())
        
    def optimize(self, batch_data):

        if self.opt.cuda:
            for i,item in enumerate(batch_data):
                batch_data[i] = item.cuda()
        for i, item in enumerate(batch_data):
            batch_data[i] = Variable(item)
        
        x0, x1, x2, attrs, fg_labels, labels = batch_data

        #Feature Extractor (4 dim in each paramters)
        feature = self.network(x0)
        prediction = self.cls_network(feature)
        #Cls Loss
        cls_loss = self.cls_loss(prediction, labels)
        
        self.update_record(self.result_record, 'total', cls_loss, prediction.size(0), prediction, labels)

        self.optimizer.zero_grad()

        cls_loss.backward()

        self.optimizer.step()
        #self.append_features(self.features, output0, output1, output2, labels)
    def combine_features(self, features):
        combined_features = {}
        for key, feat_list in features.items():

            tmp = feat_list[0]
            for feat in feat_list[1:]:
                tmp = torch.cat([tmp, feat], 0)
            combined_features[key] = tmp
        return combined_features

    def test(self, test_data, retrieval_now=True):

        self.train(False)
        
        if self.opt.cuda:
            for i,item in enumerate(test_data):
                test_data[i] = item.cuda()
        for i, item in enumerate(test_data):
            test_data[i] = Variable(item)


        x0, x1, x2, attrs, fg_labels, labels = test_data

        #Feature Extractor (4 dim in each paramters)
        feature = self.network(x0)
        prediction = self.cls_network(feature)
        #Dense Loss
        
        cls_loss = self.cls_loss(prediction, labels)
        #print(loss.data[0])
        #Cls Loss
            
        self.update_record(self.test_result_record, 'total', cls_loss, prediction.size(0), prediction, labels )


        self.train(True)

    '''
    Save intermediate feature for further testing
    '''
    def save_feature(self, mode, epoch_label):
        feature_dir = os.path.join(self.save_dir, 'feature')
        mkdir(feature_dir)
        save_filename = 'AttentionClsNetwork_{}_{}.pth.tar'.format(mode, epoch_label)
        save_path = os.path.join(feature_dir, save_filename)
        if mode == 'train':
            torch.save(self.features, save_path)
            self.reset_features()
        else:
            torch.save(self.test_features, save_path)
            self.reset_test_features()
    '''
    Save the model
    '''
    def save_model(self,  epoch_label, is_save_feature=False):
        self.save_network(self.network, 'AttentionClsNetwork' , epoch_label)
        if self.opt.save_mode and is_save_feature:
            self.save_feature(self.opt.phase, epoch_label)

    '''
    Load the model
    '''
    def load_model(self,  epoch_label, load_path):
        self.load_network(self.network, 'AttentionClsNetwork' , epoch_label, load_path=load_path)