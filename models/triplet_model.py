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

class TripletModel(BaseModel):

    def name(self):
        return 'TripletModel'
    def get_loss(self, loss_type):
        if loss_type == 'triplet' or loss_type == 'moderate_triplet':
            return TripletLoss(self.opt)#torch.nn.TripletMarginLoss(self.opt.margin)
        elif loss_type == 'holef':
            return HOLEFLoss(self.opt)
        elif loss_type == 'moderate_triplet':
            return ModerateTripletNegativeLoss(self.opt)
        return None

    def initialize(self):
        if self.opt.model == 'tripletsiamese':
            self.network = TripletSiameseNetwork(self.opt)
        elif self.opt.model == 'tripletheter':
            self.network = TripletHeterNetwork(self.opt)
        #self.network = torch.nn.DataParallel(self.network)
        if self.opt.stn:
            self.network.stn = torch.nn.DataParallel(self.network.stn)
        self.network.feat_extractor = torch.nn.DataParallel(self.network.feat_extractor)
        self.loss = self.get_loss(self.opt.loss_type[0])
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.angle_loss = AngleLoss()
        self.attr_loss = torch.nn.BCEWithLogitsLoss()
        self.optimize_modules = torch.nn.ModuleList([self.network])
        self.cls_network = torch.nn.ModuleList([])
        self.feat_map = {} 
        self.result_record = {'total':self.record_initialize(False)}
        self.features = []

        if 'sketch_cls' in self.opt.loss_type:
            self.cls_network.append(ClassificationNetwork(self.opt.feat_size, self.opt.n_labels))
            self.feat_map['sketch'] = len(self.feat_map) 
            self.optimize_modules.append(self.cls_network[self.feat_map['sketch']])
            self.result_record['sketch'] = self.record_initialize(True)

        if 'image_cls' in self.opt.loss_type:
            self.cls_network.append(ClassificationNetwork(self.opt.feat_size, self.opt.n_labels))
            self.feat_map['image'] = len(self.feat_map) 
            self.optimize_modules.append(self.cls_network[self.feat_map['image']])
            self.result_record['image'] = self.record_initialize(True)
        if 'combine_cls' in self.opt.loss_type:
            self.cls_network.append(ClassificationNetwork(self.opt.feat_size*2, self.opt.n_labels))
            self.feat_map['combine'] = len(self.feat_map)
            self.optimize_modules.append(self.cls_network[self.feat_map['combine']])
            self.result_record['combine'] = self.record_initialize(True)
        if 'fg_cls' in self.opt.loss_type:
            self.cls_network.append(ClassificationNetwork(self.opt.feat_size, self.opt.n_fg_labels))
            self.feat_map['fg'] = len(self.feat_map) 
            self.optimize_modules.append(self.cls_network[self.feat_map['fg']])
            self.result_record['fg'] = self.record_initialize(True)
        if 'sphere_cls' in self.opt.loss_type:
            self.cls_network.append(AngleClassificationNetwork(self.opt))
            self.feat_map['sphere'] = len(self.feat_map)
            self.optimize_modules.append(self.cls_network[self.feat_map['sphere']])
            self.result_record['sphere'] = self.record_initialize(False)
        if 'attr' in self.opt.loss_type:

            self.attr_network = AttributeNetwork(self.opt.feat_size*2, self.opt.n_attrs)
            self.attr_network = torch.nn.DataParallel(self.attr_network)
            self.optimize_modules.append(self.attr_network)
            self.result_record['attr'] = self.record_initialize(False)
        if 'holef' in self.opt.loss_type:
            self.optimize_modules.append(self.loss)
            #self.loss = torch.nn.DataParallel(self.loss)

        self.test_result_record = self.copy_initialize_record(self.result_record)
        
        self.optimizer = torch.optim.Adam([{"params":module.parameters()} for module in self.optimize_modules], lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        
        self.reset_features()
        self.reset_test_features()
        #self.network.module.feat_extractor = torch.nn.DataParallel(self.network.module.feat_extractor)
        if self.opt.continue_train:
            if self.opt.load_only_feat_network:
                try:
                    #self.network.feat_extractor = torch.nn.DataParallel(self.network.module.feat_extractor)
                    #print(self.network.module.feat_extractor.state_dict())
                    self.load_CNN(self.opt.model_prefix, self.opt.start_epoch_label, self.opt.trained_model_path )
                except:
                    #self.network.module.feat_extractor = torch.nn.DataParallel(self.network.module.feat_extractor)
                    #print(self.network.module.feat_extractor.state_dict())
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

    def append_features(self, features, output0, output1, output2, labels):
        features['sketch'].append(output0.data.cpu())
        features['image'].append(output1.data.cpu())
        features['neg_image'].append(output2.data.cpu())
        features['labels'].append(labels.data.cpu())

    def optimize(self, batch_data):

        if self.opt.cuda:
            for i,item in enumerate(batch_data):
                batch_data[i] = item.cuda()
        for i, item in enumerate(batch_data):
            batch_data[i] = Variable(item)
        
        x0, x1, x2, attrs, fg_labels, labels = batch_data

        #Feature Extractor (4 dim in each paramters)
        output0, output1, output2 = self.network(x0, x1, x2)
        #num_feat = len(output0)
        #self.features =  {'sketch':output0, 'image':output1, 'neg_image':output2}#output0, output1, output2]
        
        #Dense Loss
        #print(num_feat)
        loss = self.loss(output0, output1, output2)

        #Cls Loss
        final_layer_data = {'sketch':output0, 
                            'image':output1, 
                            'fg':output1,
                            'sphere':torch.cat([output0, output1], dim=0),
                            'combine':torch.cat([output0, output1], dim=1)}
        cls_loss = {}
        for key, i in self.feat_map.items():
            prediction = self.cls_network[i](final_layer_data[key])
            if key == 'sphere':
                cls_loss[key] = self.angle_loss(prediction, torch.cat([fg_labels, fg_labels], dim=0))
                self.update_record(self.result_record, key, cls_loss[key], labels.size(0))
            elif key == 'fg':
                cls_loss[key] = self.cls_loss(prediction, fg_labels)
                self.update_record(self.result_record, key, cls_loss[key], fg_labels.size(0), prediction, fg_labels)
            else:
                cls_loss[key] = self.cls_loss(prediction, labels)
                self.update_record(self.result_record, key, cls_loss[key], labels.size(0), prediction, labels)
            loss += cls_loss[key] * self.opt.loss_rate[2]
            #Update result
            

        #Attr Loss
        #
        if 'attr' in self.opt.loss_type:
            predicted_attrs = self.attr_network(final_layer_data['combine'])
            attrs = attrs.float()
            attr_loss = self.attr_loss(predicted_attrs, attrs)
            loss += attr_loss * self.opt.loss_rate[1]
            self.update_record(self.result_record, 'attr', attr_loss, predicted_attrs.size(0))
            
        self.update_record(self.result_record, 'total', loss, final_layer_data['sketch'].size(0))

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
        if self.opt.test_metric == 'euclidean': 
            cate_accs, cate_fg_accs = retrieval_evaluation(data['sketch'], data['image'], labels, self.opt.topk)
        elif self.opt.test_metric == 'cosine':
            cate_accs, cate_fg_accs = retrieval_cosine_evaluation(data['sketch'], data['image'], labels, self.opt.topk)
        self.update_record(self.test_result_record, 'retrieval', loss, labels.size(0), accs=cate_fg_accs)
        self.test_result_record['cate_retrieval'] = self.record_initialize(True)
        self.update_record(self.test_result_record, 'cate_retrieval', loss, labels.size(0), accs=cate_accs)

    def test(self, test_data, retrieval_now=True):

        self.train(False)
        #self.cpu()
        #self.network.train(True)
        if self.opt.cuda:
            for i, item in enumerate(test_data):
                test_data[i] = item.cuda()
        for i, item in enumerate(test_data):
            test_data[i] = Variable(item)

        x0, x1, x2, attrs, fg_labels, labels = test_data
        #print(x0, x1)
        #Feature Extractor (4 dim in each paramters)
        output0, output1, output2 = self.network(x0, x1, x2)
        #print(output0.data[0], output1.data[0])

        #num_feat = len(output0)
        #self.features =  {'sketch':output0, 'image':output1, 'neg_image':output2}#output0, output1, output2]
        
        #Dense Loss
        #print(num_feat)
        loss = self.loss(output0, output1, output2)
        #print(loss.data[0])
        #Cls Loss
        final_layer_data = {'sketch':output0, 
                            'image':output1, 
                            'fg':output1,
                            'sphere':torch.cat([output0, output1], dim=0),
                            'combine':torch.cat([output0, output1], dim=1)}

        cls_loss = {}
        for key, i in self.feat_map.items():
            final_layer_data[key] = final_layer_data[key].cuda()
            prediction = self.cls_network[i](final_layer_data[key])
            #prediction = prediction.cuda()
            #labels = labels.cuda()

            if key == 'sphere':
                prediction = (prediction[0].cuda(), prediction[1].cuda())
                s_fg_labels = torch.cat([fg_labels, fg_labels], dim=0)
                s_fg_labels = s_fg_labels.cuda()                
                cls_loss[key] = self.angle_loss(prediction, s_fg_labels)
                self.update_record(self.test_result_record, key, cls_loss[key], labels.size(0))
            elif key == 'fg':
                prediction = prediction.cuda()
                fg_labels = fg_labels.cuda()  
                cls_loss[key] = self.cls_loss(prediction, fg_labels)
                self.update_record(self.test_result_record, key, cls_loss[key], fg_labels.size(0), prediction, fg_labels)
              
            else:
                prediction = prediction.cuda()
                labels = labels.cuda()
                cls_loss[key] = self.cls_loss(prediction, labels)
                self.update_record(self.test_result_record, key, cls_loss[key], labels.size(0), prediction, labels)
            loss += cls_loss[key] * self.opt.loss_rate[2]
            
            #Update result
            
            

        #Attr Loss
        if 'attr' in self.opt.loss_type:
            predicted_attrs = self.attr_network(final_layer_data['combine'])
            attrs = attrs.float()
            predicted_attrs = predicted_attrs.cuda()
            attrs = attrs.cuda()
            attr_loss = self.attr_loss(predicted_attrs, attrs)
            loss += attr_loss * self.opt.loss_rate[1]
            self.update_record(self.test_result_record, 'attr', attr_loss, predicted_attrs.size(0))
            
        self.update_record(self.test_result_record, 'total', loss, final_layer_data['sketch'].size(0))

        #if not (self.opt.dataset_type == 'sketchy' or self.opt.dataset_type == 'imagenet'):
        self.append_features(self.test_features, output0, output1, output2, labels)
        print(self.test_features)
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

