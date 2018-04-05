import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .mydensenet import MultiDenseNet, DenseNet
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
    def __init__(self, num_input_features, num_output_features, kernel_size, stride, bias=False, padding=0, is_relu=True, is_bn=True, dilation=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(num_input_features, num_output_features 
                        , kernel_size=kernel_size, stride=stride, padding=padding,  bias=bias, dilation=dilation)
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
   
class DilatedConvBlock(torch.nn.Module):
    def __init__(self, opt, num_input_features):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = ConvLayer(num_input_features, 64, kernel_size=7, stride=3, bias=False, is_relu=opt.is_relu, is_bn=opt.is_bn, dilation=1)
        #self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvLayer(64, 128, kernel_size=5, stride=1, padding=0, dilation=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=1, padding=1, dilation=5)
        self.conv4 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1, is_relu=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.pool(out)
        print(out.size())
        return out
                

class SpatialTransformerNetwork(torch.nn.Module):
    def __init__(self,opt, num_input_features):
        super(SpatialTransformerNetwork, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(num_input_features, 8, kernel_size=7, stride=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            ConvLayer(8, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 9 * 9, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        print(xs.size())
        xs = xs.view(-1, 16 * 9 * 9)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

"""
class HybridDilatedConv(torch.nn.Module):
    def __init__(self, opt, num_input_features):
        super(HybridDilatedConv, self).__init__()
"""
class ConvBlock(torch.nn.Module):
    def __init__(self, opt, num_init_features):
        super(ConvBlock, self).__init__()
        """      
        if opt.sketch_type == 'GRAY':
            num_input_features = 1
        else:
            num_input_features = 3
        """
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

class SketchANet(torch.nn.Module):
    def __init__(self, opt, num_input_features):
        super(SketchANet, self).__init__()
        """        
        if opt.sketch_type == 'GRAY':
            num_input_features = 1
        else:
            num_input_features = 3
        """
        self.conv1 = ConvLayer(num_input_features, 64, kernel_size=15, stride=3, bias=False, is_bn=False )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvLayer(64, 128, kernel_size=5, stride=1, padding=0, is_bn=False)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=1, padding=1, is_bn=False)
        self.conv4 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1, is_bn=False)
        self.conv5 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1, is_bn=False)
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
    def __init__(self, opt, num_input_features):
        super(AttentionNetwork, self).__init__()
        if opt.cnn_block == 'sketchanet':
            cnn_block = SketchANet
        elif opt.cnn_block == 'bn_cnn_block':
            cnn_block = ConvBlock
        elif opt.cnn_block == 'dilated_cnn_block':
            cnn_block = DilatedConvBlock
        self.conv_block = cnn_block(opt, num_input_features)
        self.attention_layer = AttentionLayer()
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.bn_attention = nn.BatchNorm2d(512)
        self.fc6 = nn.Linear(256*7*7, 512)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc7 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        conv_feature = self.conv_block(x)
        attention_feature = self.attention_layer(conv_feature)
        linear_input_feature = attention_feature.view(attention_feature.size(0), -1)

        #print(linear_input_feature.size())
        
        out = self.fc6(linear_input_feature)
        out = self.fc7(out)
        gap_feature = self.gap(attention_feature).view(attention_feature.size(0), -1)
        out = torch.cat([out, gap_feature], 1)
        #out = self.bn7(out)
        return out

class TripletHeterNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(TripletHeterNetwork, self).__init__()
        self.opt = opt
        self.sketch_feat_model = self.get_feat_model(opt.feature_model, opt.sketch_type)
        self.image_feat_model = self.get_feat_model(opt.feature_model, opt.image_type)



    def get_feat_model(self, feature_model, read_type):
        if read_type == 'RGB':
            num_input_features = 3
        else:
            num_input_features = 1
        #copy_opt = Namespace(**vars(self.opt))
        feat_model = None
        if feature_model == 'attention':
            feat_model = AttentionNetwork(self.opt, num_input_features)
        elif feature_model == 'densenet169':
            feat_model = models.densenet169(pretrained=not self.opt.no_densenet_pretrain)
            feat_model.classifier = nn.Linear(feat_model.classifier.in_features, self.opt.feat_size)
        return feat_model
    def forward_once(self, x, feat_model):
        out = feat_model(x)
        return out

    def forward(self, x0, x1, x2):
        out0 = self.forward_once(x0, self.sketch_feat_model)
        out1 = self.forward_once(x1, self.image_feat_model)
        out2 = self.forward_once(x2, self.image_feat_model)
        return out0, out1, out2

'''
Triplet Siamese Network, For SBIR
'''
class TripletSiameseNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(TripletSiameseNetwork, self).__init__()
        self.opt = opt
        if self.opt.sketch_type == 'GRAY':
            self.num_input_features = 1
        else:
            self.num_input_features = 3
        
        self.feat_extractor = self.get_extractor(opt.feature_model)
        self.bn = nn.BatchNorm1d(opt.feat_size)
        if self.opt.stn:
            self.stn = SpatialTransformerNetwork(opt, self.num_input_features)
    def forward_once(self, x):
        if self.opt.stn:
            x = self.stn(x)
        out = self.feat_extractor(x)

        #out = self.bn(out)
        return out  
    def get_extractor(self, feature_model):
        feature_extractor = None

        

        if feature_model == 'attention':
            feature_extractor = AttentionNetwork(self.opt, self.num_input_features)
        elif feature_model == 'densenet169':
            feature_extractor = models.densenet169(pretrained=not self.opt.no_densenet_pretrain)
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, self.opt.feat_size)
        elif feature_model == 'densenet121':
            feature_extractor = models.densenet121(pretrained=not self.opt.no_densenet_pretrain)
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, self.opt.feat_size)
        elif feature_model == 'denseblock':
            feature_extractor = DenseNet(num_init_features=64, growth_rate=32,block_config=(6,6,12,12))
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, self.opt.feat_size)
            #rint(feature_extractor.classifier.in_features)
        return feature_extractor

    def forward(self, x0, x1, x2):
        out0 = self.forward_once(x0)
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out0, out1, out2

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)



class AngleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        nan_testor = np.isnan(x.data.cpu().numpy())
        if np.max(nan_testor) == 1:
            print('x is nan')
            print(x.data.cpu())
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = torch.autograd.Variable(cos_theta.data.acos())

            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)
        nan_testor = np.isnan(theta.data.cpu().numpy())
        if np.max(nan_testor) == 1:
            print('theta is nan')
            print(theta.data.cpu())
        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        nan_testor = np.isnan(cos_theta.data.cpu().numpy())
        if np.max(nan_testor) == 1:
            print('cos theta is nan')
            print(cos_theta.data.cpu())
        nan_testor = np.isnan(phi_theta.data.cpu().numpy())
        if np.max(nan_testor) == 1:
            print('phi_theta is nan')
            print(phi_theta.data.cpu())
        output = (cos_theta,phi_theta)

        return output # size=(B,Classnum,2)
class AngleClassificationNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(AngleClassificationNetwork, self).__init__()
        self.opt = opt
        self.angle_linear = AngleLinear(self.opt.feat_size, self.opt.n_fg_labels)
        
    def forward(self, x):
        return self.angle_linear(x)
class SphereNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(SphereNetwork, self).__init__()
        self.opt = opt
        self.feat_extractor = self.get_extractor(self.opt.feature_model)
        
        #self.angle_linear = AngleLinear(self.opt.feat_size, self.opt.n_fg_labels)

    def forward(self, x):
        out = self.feat_extractor(x)
        #al_out = self.angle_linear(out)
        return out

    def get_extractor(self, feature_model):
        feature_extractor = None
        if self.opt.sketch_type == 'GRAY':
            num_input_features = 1
        else:
            num_input_features = 3
        if feature_model == 'attention':
            feature_extractor = AttentionNetwork(self.opt, num_input_features)
        elif feature_model == 'densenet169':
            feature_extractor = models.densenet169(pretrained=not self.opt.no_densenet_pretrain)
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, self.opt.feat_size)
        elif feature_model == 'densenet121':
            feature_extractor = models.densenet121(pretrained=not self.opt.no_densenet_pretrain)
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, self.opt.feat_size)
        elif feature_model == 'denseblock':
            feature_extractor = DenseNet(num_init_features=64, growth_rate=32,block_config=(6,6,12,12))
            feature_extractor.classifier = nn.Linear(feature_extractor.classifier.in_features, self.opt.feat_size)
            print(feature_extractor.classifier.in_features)
        return feature_extractor

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
