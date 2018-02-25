
import torch
import numpy as np
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def retrieval_evaluation(output0, output1, category_labels, topk=(1,5)):
    output0 = output0.data.cpu()
    output1 = output1.data.cpu()
    diff = torch.zeros(output0.size(0),output0.size(0))
    def distance(x,y):
        return -math.sqrt(torch.sum(torch.pow(x-y, 2)))
    for i in range(output0.size(0)):
        for j in range(output0.size(0)):
            diff[i,j] = distance(output0[i], output1[j])
    #diff = output0 - output1
    #dist_sq = torch.sum(torch.pow(diff, 2), -1)
    #dist = -torch.sqrt(dist_sq)
    #predictions = dist.cpu().numpy()
    predictions = diff.numpy()
    category_labels = category_labels.cpu().data.numpy()
    correct = 0.0
    correct_fg = {}
    for top in topk:
        correct_fg[top] = 0
    #category_labels = np.argmax(category_labels,1)
    total = predictions.shape[0]
    maxk = max(topk)
    for i, prediction in enumerate(predictions):
        #print("prediction:",np.argmax(prediction),"category:",category_labels[i],"predict category:",category_labels[np.argmax(prediction)])
        if category_labels[i] == category_labels[np.argmax(prediction)]:
            correct += 1
        maxk_indices = prediction.argsort()[-maxk:][::-1]
        for top in topk:
            if i in maxk_indices[:top]:
                correct_fg[top] += 1
    print(correct, total)
    for top in topk:
        correct_fg[top] /= total * 0.01
        #print("top",top,correct_fg[top], correct_fg[top])
    return correct / total * 100.0, correct_fg

   
def retrieval_evaluation_parallel(output0,output1,category_labels,topk=(1,5)):
    output0 = output0.data.unsqueeze(0)
    output1 = output1.data.unsqueeze(1)
    diff = output0 - output1
    dist_sq = torch.sum(torch.pow(diff, 2), -1)
    dist = -torch.sqrt(dist_sq)
    predictions = dist.cpu().numpy()
    category_labels = category_labels.cpu().data.numpy()
    correct = 0.0
    correct_fg = {}
    for top in topk:
        correct_fg[top] = 0
    #category_labels = np.argmax(category_labels,1)
    total = predictions.shape[0]
    maxk = max(topk)
    for i, prediction in enumerate(predictions):
        print("prediction:",np.argmax(prediction),"category:",category_labels[i],"predict category:",category_labels[np.argmax(prediction)])
        if category_labels[i] == category_labels[np.argmax(prediction)]:
            correct += 1
        maxk_indices = prediction.argsort()[-maxk:][::-1]
        for top in topk:
            if i in maxk_indices[:top]:
                correct_fg[top] += 1
    print(correct, total)
    for top in topk:
        correct_fg[top] /= total * 0.01
        print("top",top,correct_fg[top], correct_fg[top])
    return correct / total * 100.0, correct_fg

def retreival_accuracy(output0, output1, target, topk=(1,)):
    """Computes the precision@k for retreival for the specified values of k"""
    output0 = output0.unsqueeze(0)
    output1 = output1.unsqueeze(1)
    diff = output0 - output1
    dist_sq = torch.sum(torch.pow(diff, 2), -1)
    dist = -torch.sqrt(dist_sq)
    return accuracy(dist, target, topk)

#def retreival_accuracy_evaluation(output0, output1, target,  top=(1,)):


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res[k] = (correct_k.mul_(100.0 / batch_size))
    return res
