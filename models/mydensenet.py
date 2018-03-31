import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.autograd import Variable
__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet161']))
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.size())
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        #print(out.size())
        out = self.classifier(out)
        return out

class MultiDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_shape, feat_size=128, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(MultiDenseNet, self).__init__()
        self.block_config = block_config
        self.multi_feat_sizes = []
        # First convolution
        self.features_before = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.add_module('features_before', self.features_before)
        #self.features = nn.Sequential(OrderedDict([
        #    ('feat0', self.features_before)]))
        # Each denseblock
        num_features = num_init_features
        self.blocks = nn.ModuleList([])
        self.transs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        #self.bottlenecks = nn.ModuleList([])
        self.gaps = []
        self.linears = nn.ModuleList([])

        block_input_shape = self._get_block_input_shape(input_shape)
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.blocks.append(block)
            #self.features.add_module('denseblock%d' % (i + 1), block)
            
            #self.bottlenecks.append(nn.Conv2d(num_features + num_layers * growth_rate, 1, kernel_size=1, stride=1, bias=False))
            block_input_shape, linear_num_features = self._get_linear_input_shape(num_features, block_input_shape, block,i)
            print(block_input_shape, num_features)
            num_features = num_features + num_layers * growth_rate
            self.bns.append(nn.BatchNorm2d(num_features))
            #print(i, block_input_shape, linear_num_features)
            self.gaps.append(nn.AvgPool2d(kernel_size=block_input_shape[0], stride=1))
            self.linears.append(nn.Linear(num_features, feat_size))
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                
                block_input_shape = self._get_trans_input_shape(num_features, block_input_shape, trans) 
                self.transs.append(trans)
            #    self.features.add_module('transition%d' % (i + 1), trans), 'bottlenecks':self.bottlenecks
                num_features = num_features // 2
        for pre_key, modules in {'block':self.blocks, 'trans':self.transs, 'bn':self.bns, 'linaer':self.linears}.items():
            for key, module in enumerate(modules):
                self.add_module('{}_{}'.format(pre_key, key), module)
        # Final batch norm
        #self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.final_bn = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, feat_size)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def _get_trans_input_shape(self, num_features, input_shape, trans):
        bs = 1
        input_var = Variable(torch.rand(bs,num_features,*input_shape))
        output_var = trans(input_var)
        return output_var.size()[2:] 
    def _get_block_input_shape(self, input_shape):
        bs = 1
        input_var = Variable(torch.rand(bs, *input_shape))
        output_var = self.features_before(input_var)
        return output_var.size()[2:]
    def _get_linear_input_shape(self, num_features, input_shape, block, i):
        bs = 1
        input_var = Variable(torch.rand(bs,num_features,*input_shape))
        output_var = block(input_var)
        #output_var_ = self.simpliy(output_var,i)#F.avg_pool2d(output_var, kernel_size=7, stride=1).view(output_var.size(0), -1)
        return output_var.size()[2:], output_var.size(1) 
    #def simpliy(self, features, i):
        #print(features.size())
        #features = self.bottlenecks[i](features)
        #features = F.avg_pool2d(features, kernel_size=7, stride=1)#.view(features.size(0), -1)
        #features = F.avg_pool2d(features, kernel_size=7, stride=1).view(features.size(0), -1)
        #return features
    def forward(self, x):
        x = self.features_before(x)
        inter_xs = []
        for i, num_layers in enumerate(self.block_config):
            x = self.blocks[i](x)
            if self.training:
                #print('before', x.size())
                xtmp = self.bns[i](x)
                #print('after', xtmp.size())

                xtmp = F.relu(xtmp, inplace=True)
                #xtmp = self.simpliy(xtmp,i)#F.avg_pool2d(xtmp, kernel_size=7, stride=1).view(xtmp.size(0), -1)
                xtmp = self.gaps[i](xtmp)
                xtmp = xtmp.view(xtmp.size(0), -1)
                #print('xtmp',xtmp.size())
                xtmp = self.linears[i](xtmp)
                inter_xs.append(xtmp)
            if i != len(self.block_config) - 1:
                x = self.transs[i](x)
        x = self.final_bn(x)
        out = F.relu(x, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(out.size(0), -1)
        
        out = self.classifier(out)
        inter_xs.append(out)
        return inter_xs
        #features = self.features(x)
        #out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        #out = self.classifier(out)
        #return out
