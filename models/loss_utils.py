import torch
import torch.nn
import math
import numpy as np
import torch.nn.functional as F


class ModerateTripletNegativeLoss(torch.nn.Module):
    """
    Moderate Hardest Triplet Loss
    """
    def __init__(self,opt):
        super(ModerateTripletNegativeLoss, self).__init__()
        # margin=1.0, P=8, K=4
        self.opt = opt
        self.margin = self.opt.margin
        self.P = self.opt.P
        self.K = self.opt.K
        self.pset = []
        for p in range(self.P):
            pos_interval = torch.arange(p*self.K, (p+1)*self.K).type(torch.LongTensor)
            mask = torch.zeros(self.P * self.K).type(torch.LongTensor)

            mask[pos_interval] = 1
            self.pset.append((mask, 1-mask))

    def compute_dist(self, o0, o1):
        o0 = o0.unsqueeze(0)
        o1 = o1.unsqueeze(1)
        diff = o0 - o1
        dist_sq = torch.sum(torch.pow(diff, 2), -1)
        dist = torch.sqrt(dist_sq)
        return dist

    def forward(self, x0, x1, x2):
        loss = torch.autograd.Variable(torch.zeros(1).cuda())

        for p1, p2 in self.pset:
            p1, p2 = p1.cuda(), p2.cuda()
            o0, o1, o2 = x0[p1], x1[p1], x1[p2]
            dist_01 = self.compute_dist(o0, o1)
            dist_02 = self.compute_dist(o0, o2)
            mdist = self.margin + torch.max(dist_01, 1)[0] - torch.min(dist_02, 1)[0]
            loss_ = torch.sum(torch.clamp(mdist,min=0.0))
            loss = loss + loss_
        return loss / 2.0 / self.P /self.K


class AngleLoss(torch.nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = torch.autograd.Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output,dim=0)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        nan_testor = np.isnan(logpt.data.cpu().numpy())
        if np.max(nan_testor) == 1:
            print('logpt is nan')
        nan_testor = np.isnan(pt.data.cpu().numpy())
        if np.max(nan_testor) == 1:
            print('pt is nan')
        nan_testor = np.isnan(((1-pt)**self.gamma).data.cpu().numpy())
        if np.max(nan_testor) == 1:
            print('1-pt is nan')        
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

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
        self.alpha = 0.0005
        self.beta = 0.0005
        cuda = opt.cuda
        k = opt.feat_size
        self.k = k
        self.weight = torch.nn.Parameter(torch.eye(k))
        #self.linear = torch.nn.Linear(k,k,bias=False)
        #self.register_parameter('sweight',self.weight)
        self.I = torch.autograd.Variable(torch.eye(k), requires_grad=False)
        
        self.reset_parameter()
        if cuda:
            #self.linear = self.linear.cuda()
            self.weight.cuda()
            self.I = self.I.cuda()
        #self.I = torch.eye(k)
    def reset_parameter(self):
        torch.nn.init.kaiming_uniform(self.weight)
    def higher_energy_distance(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        outer_sub = torch.pow(x - y, 2)
        #output = self.linear(outer_sub)
        #output = output.view(output.size(0),-1)
        
        output = outer_sub * self.weight
        print(output.size(), outer_sub.size())
        return (torch.sum(output,1)) 

    def forward(self, x0, x1, x2):
        dist_pos = self.higher_energy_distance(x0, x1)
        dist_neg = self.higher_energy_distance(x0, x2)
        mdist = self.margin + dist_pos - dist_neg
        loss = torch.clamp(mdist, min=0.0)
        norm1 = self.alpha * torch.norm(self.weight -self.I,1)
        normF = self.beta * torch.sqrt(torch.sum(torch.pow(self.weight - self.I, 2)))
        loss = torch.sum(loss) / 2.0 / x0.size(0) + norm1 + normF
        #print(self.weight.data[0])
        #print("loss:{},norm1:{},normF:{}".format(loss.data,norm1.data, normF.data))
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
        #print(diff_pos.cpu().numpy())
        dist_pos_sq = torch.sum(torch.pow(diff_pos, 2), 1)
        dist_pos = torch.sqrt(dist_pos_sq)
        #print(dist_pos.data.cpu().numpy())
        dist_neg_sq = torch.sum(torch.pow(diff_neg, 2), 1)
        dist_neg = torch.sqrt(dist_neg_sq)
        #print(diff_.data.cpu().numpy())
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
