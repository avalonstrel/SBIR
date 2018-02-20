import torch
import numpy as np
import os
from .base_model import BaseModel
from .networks import *
from .loss_utils import *
from util.evaluation import *
from util.util import *

class TripletModel(BaseModel):

    def name(self):
        return 'DenseLossModel'

    def initialize(self):
        self.network = SBIRSiameseNetwork(self.opt)
        if distance_type == 'holef':
            self.loss = HOLEF(self.opt)
        elif distance_type == 'euclidean':
            self.loss = TripletLoss(self.opt)
        self.cls_loss = nn.CrossEntropyLoss()
        self.attr_loss = nn.BCEWithLogitsLoss()
        self.optimize_modules = [self.network]
        self.cls_network = {}
        self.result_record = {'total':self.record_initialize(False)}
        self.features = []
        if 'sketch_cls' in self.opt.loss_type:
            self.cls_network['sketch'] = ClassificationNetwork(self.opt.feat_size, self.opt.n_labels)
            optimize_modules.append(self.cls_network['sketch'])
            self.loss_record['sketch'] = self.record_initialize(True)
        if 'image_cls' in self.opt.loss_type:
            self.cls_network['image']= ClassificationNetwork(self.opt.feat_size, self.opt.n_labels)
            optimize_modules.append(self.cls_network['image'])
            self.loss_record['image'] = self.record_initialize(True)
        if 'combine_cls' in self.opt.loss_type:
            self.cls_network['combine'] = ClassificationNetwork(self.opt.feat_size*2, self.opt.n_labels)
            optimize_modules.append(self.cls_network['combine'])
            self.loss_record['combine'] = self.record_initialize(True)
        if 'attr' in self.opt.loss_type:
            self.attr_network = AttributeNetwork(self.opt.feat_size*2, self.opt.n_attrs)
            optimize_modules.append(self.attr_network)
            self.loss_record['attr'] = self.record_initialize(False)
        if 'holef' == self.opt.distance_type:
            optimize_modules.append(self.loss.base_loss.linear)

        self.test_result_record = self.copy_initialize_record(self.result_record)

        self.optimizer = torch.optim.Adam([{"params":module.parameters for module in optimize_modules}], lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        
        if len(self.opt.gpu_ids) > 1:
            self.parallel()
            self.cuda()
        else:
            self.cuda()
        if self.opt.continue_train:
            self.load_model(self.opt.start_epoch)


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
        result_record[key]['loss_value'].update(loss.data[0], size)
        if accs != None:
            for i, topk in enumerate(self.opt.topk):
                result_record[key]['acc'][topk].update(accs[topk].data[0], size)  
   
        elif prediction != None and labels != None:
            res = accuracy(prediction, labels, self.opt.topk)
            for i, topk in enumerate(self.opt.topk):
                result_record[key]['acc'][topk].update(res[topk].data[0], size)  

    def generate_message(self, result_record):
        messages = []
        for key, record in result_record.items():

                if 'acc' in record:
                    tmp_message = '{}:{%6f}, {}'.format(key, record['loss_value'].avg, accs_message(record['acc']))
                else:
                    tmp_message = '{}:{%6f}'.format(key, record['loss_value'].avg)
            messages.append(tmp_message)
        message = " | ".join(messages)
        return message

    #def message(self):
    #   return self.message

    def optimize(self, batch_data):

        if self.opt.cuda:
            for i,item in enumerate(batch_data):
                batch_data[i] = item.cuda()
        for i, item in enumerate(batch_data):
            batch_data[i] = Variable(item)
        
        x0, x1, x2, attrs, fg_labels, labels = batch_data

        #Feature Extractor (4 dim in each paramters)
        output0, output1, output2 = self.network(x0, x1, x2)
        num_feat = len(output0)
        self.features =  {'sketch':output0, 'image':output1, 'neg_image':output2}#output0, output1, output2]
        #Dense Loss
        loss = self.loss(output0, output1, output2)

        #Cls Loss
        final_layer_data = {'sketch':output0, 
                            'image':output1, 
                            'combine':torch.cat([output0,output1], dim=1)}
        cls_loss = {}
        for key, cls_network in self.cls_network.items():
            prediction = cls_network(final_layer_data[key])
            cls_loss[key] = self.cls_loss(prediction, labels)
            loss += cls_loss[key] * self.opt.loss_rate[2]
            #Update result
            self.update_record(self.result_record, key, cls_loss[key], prediction, labels)

        #Attr Loss
        if 'attr' in self.opt.loss_type:
            predicted_attrs = self.attr_network(final_layer_data['combine'])
            attrs = attrs.float()
            attr_loss = self.attr_loss(predicted_attrs, attrs)
            loss += attr_loss * self.opt.loss_rate[1]
            self.update_record(self.result_record, key, attr_loss)
            
        self.update_record(self.result_record, 'total', loss)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()



    def test(self, test_data):

        self.train(False)
        if self.opt.cuda:
            for i,item in enumerate(batch_data):
                batch_data[i] = item.cuda()
        for i, item in enumerate(batch_data):
            batch_data[i] = Variable(item)

        x0, x1, x2, attrs, fg_labels, labels = batch_data

        #Feature Extractor (4 dim in each paramters)
        output0, output1, output2 = self.network(x0, x1, x2)
        num_feat = len(output0)
        self.features =  {'sketch':output0, 'image':output1, 'neg_image':output2}#output0, output1, output2]
        #Dense Loss
        loss = self.loss(output0, output1, output2)

        #Cls Loss
        final_layer_data = {'sketch':output0, 
                            'image':output1, 
                            'combine':torch.cat([output0,output1], dim=1)}

        cls_loss = {}
        for key, cls_network in self.cls_network.items():
            prediction = cls_network(final_layer_data[key])
            cls_loss[key] = self.cls_loss(prediction, labels)
            loss += cls_loss[key] * self.opt.loss_rate[2]
            
            #Update result
            self.update_record(self.test_result_record, key, cls_loss[key], prediction.size(0), prediction, labels)

        #Attr Loss
        if 'attr' in self.opt.loss_type:
            predicted_attrs = self.attr_network(final_layer_data['combine'])
            attrs = attrs.float()
            attr_loss = self.attr_loss(predicted_attrs, attrs)
            loss += attr_loss * self.opt.loss_rate[1]
            self.update_record(self.test_result_record, key, attr_loss, predicted_attrs.size(0))
            
        self.test_result_record['retrieval'] = self.record_initialize(True)

        cate_accs, cate_fg_accs = retrieval_evaluation(final_layer_data['sketch'], final_layer_data['image'], labels, self.opt.topk)

        self.update_record(self.test_result_record, 'retrieval', loss, prediction.size(0), accs=cate_fg_accs)
        self.test_result_record['cate_retrieval'] = self.record_initialize(True)
        self.update_record(self.test_result_record, 'cate_retrieval', loss, prediction.size(0), accs=cate_accs)

        self.train(True)

    '''
    Save intermediate feature for further testing
    '''
    def save_feature(self, mode, epoch_label):
        feature_dir = os.path.join(self.save_dir, 'feature')
        mkdir(feature_dir)
        feature_name = 'TripletSBIR_{}_{}.pth.tar'.format(mode, epoch_label)
        save_path = os.path.join(feature_dir, save_filename)
        torch.save(self.features, save_path)

    '''
    Save the model
    '''
    def save_model(self,  epoch_label):
        self.save_network(self.network.feature_extractor, '{}_feat_{}'.format(self.network.embedding_type, self.network.feat_size) , epoch_label)
        for key, network in self.cls_network.items():
            self.save_network(network, key + '_Cls', epoch_label)
        if 'attr' in self.opt.loss_type:
            self.save_network(self.attr_network, 'attr', epoch_label)
        if 'holef' in self.opt.distance_type:
            self.save_network(self.loss.base_loss.linear, epoch_label)
        if self.opt.save_mode:
            self.save_feature(self.opt.mode, epoch_label)

    '''
    Load the model
    '''
    def load_model(self,  epoch_label):
        self.load_network(self.network, '{}_feat_{}'.format(self.network.embedding_type, self.network.feat_size) , epoch_label)
        for key, network in self.cls_network.items():
            self.load_network(network, key + '_Cls', epoch_label)
        if 'attr' in self.opt.loss_type:
            self.load_network(self.attr_network, 'attr', epoch_label)
        if 'holef' in self.opt.distance_type:
            self.load_network(self.loss.base_loss.linear, epoch_label)

