import torch
import torch.nn
import math
import numpy as np
import torch.nn.functional as F
#import ot


class DenseLoss(torch.nn.Module):
    '''
    Different Layer embedding Loss
    By optimial transport
    '''
    def __init__(self, opt):
        super(DenseLoss, self).__init__()
        self.num_layers = opt.num_layers
        self.base_loss = self.get_loss(opt)
    def get_loss(self, opt):
        loss_type = opt.loss_type[0]
        if loss_type == 'holef':
            loss = HOLEFLoss(opt)
        else:
            loss = TripletLoss(opt)
 
        return loss

    def forward(self, x0, x1, x2):
        loss_W = np.zeros((self.num_layers, self.num_layers))
        for i in range(self.num_layers):
            for j in range(self.num_layers):
                loss_W[i, j] = self.base_loss(x0[i], x1[j], x2[j])
        #weight = ot.emd([],[],loss_W)
        loss_W = torch.from_numpy(loss_W)
        weight = np.eye(self.num_layers)
        weight = torch.from_numpy(weight)
        return torch.sum(weight*loss_W)



class HOLEFLoss(torch.nn.Module):
    """
    HOLEF Loss
    """
    def __init__(self, opt): # k, alpha=0.0005,beta=0.0005, margin=2.0, cuda=True):
        super(HOLEFLoss, self).__init__()
        self.margin = opt.margin
        self.alpha = 0.005
        self.beta = 0.005
        cuda = opt.cuda
        k = opt.feat_size
        #self.weight = torch.nn.Parameter(torch.Tensor(k,k))
        self.linear = torch.nn.Linear(k,k,bias=False)
        
        self.I = torch.autograd.Variable(torch.eye(k), requires_grad=False)
        
        #self.reset_parameter()
        if cuda:
            self.linear = self.linear.cuda()
            self.I = self.I.cuda()
        #self.I = torch.eye(k)
    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def higher_energy_distance(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        outer_sub = torch.pow(x - y, 2)
        output = self.linear(outer_sub)
        output = output.view(output.size(0),-1)
       
        return (torch.sum(output,1)) 

    def forward(self, x0, x1, x2):
        dist_pos = self.higher_energy_distance(x0, x1)
        dist_neg = self.higher_energy_distance(x0, x2)
        mdist = self.margin + dist_pos - dist_neg
        loss = torch.clamp(mdist, min=0.0)
        norm1 = self.alpha * torch.norm(self.linear.weight -self.I,1)
        normF = self.beta * torch.sqrt(torch.sum(torch.pow(self.linear.weight - self.I, 2)))
        loss = torch.sum(loss) / 2.0 / x0.size(0) + norm1 + normF
        # print("loss:{},norm1:{},normF:{}".format(loss.data,norm1.data, normF.data))
        return loss
class TripletLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on: l2 distance
    """

    def __init__(self, opt):
        super(TripletLoss, self).__init__()
        self.margin = opt.margin

    def forward(self, x0, x1, x2 ):
        # euclidian distance

        diff_pos = x0 - x1
        diff_neg = x0 - x2

        dist_pos_sq = torch.sum(torch.pow(diff_pos, 2), 1)
        dist_pos = torch.sqrt(dist_pos_sq)

        dist_neg_sq = torch.sum(torch.pow(diff_neg, 2), 1)
        dist_neg = torch.sqrt(dist_neg_sq)
        mdist = self.margin + dist_pos - dist_neg
        loss = torch.clamp(mdist, min=0.0)

        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

class AttributeLoss(torch.nn.Module):
    """
    Loss function for attribute
    """
    def __init__(self):
        super(AttributeLoss, self).__init__()
    def forward(self, o, t):
        return torch.nn.BCEWithLogitsLoss(o,t)
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self, margin=100.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1,  y):
        # euclidian distance
        diff = x0 - x1

        #print(diff)
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        mdist = torch.clamp(mdist, min=0.0)
        y = y.float()
        loss = y * dist + (1 - y) * mdist
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
