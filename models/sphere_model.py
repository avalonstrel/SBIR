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

class SphereModel(BaseModel):

    def name(self):
        return 'SphereModel'

    def initialize(self):
        self.network = SphereModel(self.opt)
        #self.network = torch.nn.DataParallel(self.network)
        self.network.feat_extractor = torch.nn.DataParallel(self.network.feat_extractor)
        self.cls_network = AngleClassificationNetwork(self.opt)
        self.loss = AngleLoss()
        self.optimize_modules = torch.nn.ModuleList([self.network, self.cls_network])
         
        self.result_record = {'total':self.record_initialize(False)}
        self.test_result_record = self.copy_initialize_record(self.result_record)
        
        self.optimizer = torch.optim.Adam([{"params":module.parameters()} for module in self.optimize_modules], lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        
        self.reset_features()
        self.reset_test_features()
        if self.opt.continue_train:
            if self.opt.load_only_feat_network:
                try:
                    self.load_CNN(self.opt.model_prefix, self.opt.start_epoch_label, self.opt.trained_model_path )
                except:
                    self.load_CNN(self.opt.model_prefix, self.opt.start_epoch_label, self.opt.trained_model_path )
            else:

                self.load_model(self.opt.start_epoch_label, self.opt.trained_model_path)
        self.parallel_flag = len(self.opt.gpu_ids) > 1
        if self.parallel_flag:
            self.parallel()
            print('Model parallel...')
            self.cuda()
            print('Model cuda ing...')
        elif self.opt.cuda:
            self.cuda()
            print('Modelcuda ing...')

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

    #def message(self):
    #   return self.message
    def reset_features(self):
        self.features = {'sketch':[], 'image':[], 'neg_image':[], 'labels':[]}
    def reset_test_features(self):
        self.test_features = {'sketch':[], 'image':[], 'neg_image':[], 'labels':[]}
    def append_features(self, features, output0, output1, labels):
        features['sketch'].append(output0.data.cpu())
        features['image'].append(output1.data.cpu())
        #features['neg_image'].append(output2.data.cpu())
        features['labels'].append(labels.data.cpu())

    def optimize(self, batch_data):

        if self.opt.cuda:
            for i,item in enumerate(batch_data):
                batch_data[i] = item.cuda()
        for i, item in enumerate(batch_data):
            batch_data[i] = Variable(item)
        
        x0, x1, x2, attrs, fg_labels, labels = batch_data

        #Feature Extractor (4 dim in each paramters)
        #output0, output1, output2 = self.network(x0, x1, x2)
        output0 = self.network(x0)
        output1 = self.network(x1)

        output = torch.cat([output0, output1], 0)
        fg_labels = torch.cat([fg_labels, fg_labels], 0)
        output = output.cuda()
        fg_labels = fg_labels.cuda()
        prediction = self.cls_network(output)
        loss = self.loss(prediction, fg_labels)
            
        self.update_record(self.result_record, 'total', loss, output.size(0))

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        #self.append_features(self.features, output0, output1, output2, labels)
    def combine_features(self, features):
        combined_features = {}
        for key, feat_list in features.items():

            tmp = feat_list[0]
            for feat in feat_list[1:]:
                tmp = torch.cat([tmp, feat], 0)
            combined_features[key] = Variable(tmp)
        return combined_features

    def retrieval_evaluation(self, data, loss,  labels):
        self.test_result_record['retrieval'] = self.record_initialize(True)
        cate_accs, cate_fg_accs = retrieval_evaluation(data['sketch'], data['image'], labels, self.opt.topk)
        self.update_record(self.test_result_record, 'retrieval', loss, labels.size(0), accs=cate_fg_accs)
        self.test_result_record['cate_retrieval'] = self.record_initialize(True)
        self.update_record(self.test_result_record, 'cate_retrieval', loss, labels.size(0), accs=cate_accs)

    def test(self, test_data, retrieval_now=True):

        self.train(False)

        for i, item in enumerate(test_data):
            test_data[i] = Variable(item)

        if self.opt.cuda:
            for i, item in enumerate(test_data):
                item.cuda()

        x0, x1, x2, attrs, fg_labels, labels = test_data

        output0 = self.network(x0)
        output1 = self.network(x1)

        output = torch.cat([output0, output1], 0)
        fg_labels = torch.cat([fg_labels, fg_labels], 0)
        output = output.cuda()
        fg_labels = fg_labels.cuda()

        #loss = self.loss(output, fg_labels)
        final_layer_data = {'sketch':output0, 
                            'image':output1, 
                            'combine':torch.cat([output0, output1], dim=1)}

        self.update_record(self.test_result_record, 'total', 0.0 , final_layer_data['sketch'].size(0))

        #if not (self.opt.dataset_type == 'sketchy' or self.opt.dataset_type == 'imagenet'):
        self.append_features(self.test_features, output0, output1, labels)
        if retrieval_now:
            self.retrieval_evaluation(final_layer_data, loss, labels)
        #self.cuda()
        self.train(True)

    '''
    Save intermediate feature for further testing
    '''
    def save_feature(self, mode, epoch_label):
        feature_dir = os.path.join(self.save_dir, 'feature')
        mkdir(feature_dir)
        save_filename = 'TripletSBIRNetwork_{}_{}.pth.tar'.format(mode, epoch_label)
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
        self.save_network(self.network, 'TripletSBIRNetwork' , epoch_label)
        for key, i in self.feat_map.items():
            self.save_network(self.cls_network[i], key + '_Cls', epoch_label)
        if 'attr' in self.opt.loss_type:
            self.save_network(self.attr_network, 'attr', epoch_label)
        if 'holef' in self.opt.distance_type:
            self.save_network(self.loss.base_loss.linear, epoch_label)
        if self.opt.save_mode and is_save_feature:
            self.save_feature(self.opt.phase, epoch_label)
    '''
    Only Load CNN Model
    '''
    def load_CNN(self, model_prefix, epoch_label, load_path ):

        self.load_network(self.network.feat_extractor, model_prefix , epoch_label, load_path=load_path)
    '''
    Load the model
    '''
    def load_model(self,  epoch_label, load_path):
        self.load_network(self.network, 'TripletSBIRNetwork' , epoch_label, load_path=load_path)
        for key, i in self.feat_map.items():
            self.load_network(self.cls_network[i], key + '_Cls', epoch_label)
        if 'attr' in self.opt.loss_type:
            self.load_network(self.attr_network, 'attr', epoch_label, load_path=load_path)
        if 'holef' in self.opt.distance_type:
            self.load_network(self.loss.base_loss.linear, epoch_label, load_path=load_path)

