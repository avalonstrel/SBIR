import torch
import torch.nn.functional as F
#import torchvision.models as models
import numpy as np
from .mydensenet import MultiDenseNet
import torch.nn as nn
def save_feature(state, filename):
    torch.save(state, filename)

class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.attention1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.attention2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        
        #self.out_features = out_features
    def forward(self, x):
        
        out = self.attention1(x)
        attention = self.attention2(out)
        out = (x + attention * x)

        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, kernel_size, stride, bias=False, padding=0, is_relu=True, is_bn=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(num_input_features, num_output_features 
                        , kernel_size=kernel_size, stride=stride, padding=padding,  bias=bias)
        self.bn = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.is_relu = is_relu
        self.is_bn = is_bn
    def forward(self, x):
        out = self.conv(x)
        if self.is_bn:
            out = self.bn(out)
        if self.is_relu:
            out = self.relu(out)
        return out

class ConvBlock(torch.nn.Module):
    def __init__(self, opt):
        super(ConvBlock, self).__init__()
        if opt.sketch_type == 'GRAY':
            num_input_features = 1
        else:
            num_input_features = 3
        self.conv1 = ConvLayer(num_input_features, 64, kernel_size=15, stride=3, bias=False, is_relu=opt.is_relu, is_bn=opt.is_bn)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvLayer(64, 128, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1, is_relu=False)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.pool3(out)
        return out


class AttentionNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(AttentionNetwork, self).__init__()
        self.conv_block = ConvBlock(opt)
        self.attention_layer = AttentionLayer()
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc6 = nn.Linear(256*7*7, 512)
        self.fc7 = nn.Linear(512, 256)
        #self.bn7 = nn.BatchNorm2d(512)
    def forward(self, x):
        conv_feature = self.conv_block(x)
        attention_feature = self.attention_layer(conv_feature)
        linear_input_feature = attention_feature.view(attention_feature.size(0), -1)
        out = self.fc6(linear_input_feature)
        out = self.fc7(out)

        gap_feature = self.gap(attention_feature).view(attention_feature.size(0), -1)
        out = torch.cat([out, gap_feature], 1)
        #out = self.bn7(out)
        return out

'''
Triplet Siamese Network, For SBIR
'''
class TripletSiameseNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(TripletSiameseNetwork, self).__init__()
        self.opt = opt
        self.feat_extractor = self.get_extractor(opt.feature_model)

    def forward_once(self, x):
        out = self.feat_extractor(x)
        return out  
    def get_extractor(self, feature_model):
        feature_extractor = None
        if feature_model == 'attention':
            feature_extractor = AttentionNetwork(self.opt)
        elif feature_model == 'densenet169':
            feature_extractor = models.densenet169(pretrained=pretrain)
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, self.opt.feat_size)
        return feature_extractor

    def forward(self, x0, x1, x2):
        out0 = self.forward_once(x0)
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out0, out1, out2

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
