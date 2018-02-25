import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .mydensenet import MultiDenseNet
import torch.nn as nn
def save_feature(state, filename):
    torch.save(state, filename)

class AttentionNetwork(torch.nn.Module):
    def __init__(self, out_features):
        super(AttentionNetwork, self).__init__()
        self.attention = nn.Linear(out_features, out_features, False)
        self.out_features = out_features
    def forward(self, x):
        attention = self.attention(x)
        attention = F.softmax(attention,dim=1)

        return x + attention * x


class DenseSBIRNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(DenseSBIRNetwork, self).__init__()
        input_shape = (3, opt.scale_size, opt.scale_size)
        self.feat_extractor = MultiDenseNet(input_shape, feat_size=opt.feat_size)
        self.save_mode = opt.save_mode
        self.attention_mode = opt.attention_mode
        self.fc_layer_mode = opt.fc_layer_mode
        self.fusion_mode = opt.fusion_mode

    def forward_once(self, x):
        output = self.feat_extractor(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)


        return output1, output2, output3

class SBIRSiameseNetwork(torch.nn.Module):

    def __init__(self, opt):#output_size,embedding_type, pretrain, concatation=[0,1,2,3], mode = False, attention=True,fc_layer_mode=True, fusion_mode=True):
        super(SBIRSiameseNetwork, self).__init__()
        self.save_mode = opt.save_mode
        self.embedding_type = opt.feature_model
        self.attention_mode = opt.attention_mode
        self.fc_layer_mode = opt.fc_layer_mode
        self.fusion_mode = opt.fusion_mode
        self.feat_size = opt.feat_size
        output_size = opt.feat_size
        hidden_size = output_size
        concatation = [0,1,2,3]
        pretrain=True
        self.feature_extractor_image = self.get_extractor(self.embedding_type, pretrain, hidden_size, concatation)
        #self.feature_extractor_sketch = self.get_extractor(embedding_type, pretrain, output_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.feature_extractor = self.feature_extractor_image
        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.linear = nn.Linear(hidden_size,output_size)
        self.attention = AttentionNetwork(output_size)

    def get_extractor(self, embedding_type, pretrain, output_size, concatation=[0,1,2,3]):
        if embedding_type == "inception":
            feature_extractor = models.inception_v3(pretrained=pretrain)
            feature_extractor.fc = nn.Linear(2048,output_size)
            feature_extractor.aux_logits = False
            self.dropout = nn.Dropout(0.5)
        elif embedding_type == "resnet50":
            feature_extractor = models.resnet50(pretrained=pretrain)
            #self.output_sizes = np.array([1218816,524288,50176,32768])
            #final_input_size = np.sum(self.output_sizes[concatation])
            #print('final',final_input_size)
            print(feature_extractor.fc.in_features)
            feature_extractor.fc = nn.Linear(feature_extractor.fc.in_features, output_size)
        elif embedding_type == 'resnet101':
            feature_extractor = models.resnet101(pretrained=pretrain)
            #self.output_sizes = np.array([1218816,524288,50176,32768])
            #final_input_size = np.sum(self.output_sizes[concatation])
            #print('final',final_input_size)
            print(feature_extractor.fc.in_features)
            feature_extractor.fc = nn.Linear(feature_extractor.fc.in_features, output_size)
        
        elif embedding_type == 'densenet169':
            feature_extractor = models.densenet169(pretrained=pretrain)
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, output_size)
        elif embedding_type == 'densenet201':
            feature_extractor = models.densenet201(pretrained=pretrain)
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, output_size) 
        return feature_extractor
    def forward_once(self, x, feature_extractor):
        output = feature_extractor(x)
        return output

    def forward(self, input1, input2, input3):
        output1_ori = self.forward_once(input1, self.feature_extractor_image)
        output2_ori = self.forward_once(input2, self.feature_extractor_image)
        output3_ori = self.forward_once(input3, self.feature_extractor_image)
        output1, output2, output3 = output1_ori , output2_ori, output3_ori
        #output1, output2, output3 = output1_ori[0] , output2_ori[0], output3_ori[0]
         
        #output1, output2, output3 = self.batch_norm(output1), self.batch_norm(output2), self.batch_norm(output3)
        #output1, output2, output3 = self.dropout(output1), self.dropout(output2), self.dropout(output3)
        #output1, output2, output3 = F.relu(output1), F.relu(output2), F.relu(output3)
        if self.fc_layer_mode:
            output1, output2, output3 = self.linear(output1) , self.linear(output2), self.linear(output3)
            output1, output2, output3 = self.batch_norm2(output1), self.batch_norm2(output2), self.batch_norm2(output3)

        #output1, output2, output3 = F.relu(output1), F.relu(output2), F.relu(output3)
        #output1, output2, output3 = F.normalize(output1),F.normalize(output2),F.normalize(output3)
        if self.attention_mode:
            output1, output2, output3 = self.attention(output1), self.attention(output2), self.attention(output3)
        if self.fusion_mode:
            output1, output2, output3 = output1 + output1_ori, output2 + output2_ori, output3 + output3_ori
        return output1, output2, output3, output1_ori, output2_ori, output3_ori


class ClassificationNetwork(torch.nn.Module):
    def __init__ (self, input_size, n_labels):
        super(ClassificationNetwork, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        #self.dropout = nn.Dropout(0.5)
        #self.batch_norm2 = nn.BatchNorm1d(n_labels)
        self.classifier = nn.Linear(input_size, n_labels)
    def forward(self, x):
        output = self.batch_norm(x)
 #       output = F.relu(output)
        output = self.classifier(output)
        #output = self.batch_norm2(output)
        return output

class AttributeNetwork(torch.nn.Module):
    def __init__ (self, input_size, n_labels):
        super(AttributeNetwork, self).__init__()
        hidden_size = 128
        self.linear1 = nn.Linear(input_size, n_labels)
        self.batch_norm = nn.BatchNorm1d(input_size)
        #self.linear2 = nn.Linear(hidden_size, n_labels)

    def forward(self, x):
        output = self.batch_norm(x)
        output = self.linear1(output)
        #output = F.relu(output)
        #output = self.linear2(output)
        return output
