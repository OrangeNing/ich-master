import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from configs import config_cnn as cfg
from torch.autograd import Variable
import math


class FocalLoss(object):
    def __init__(self, alpha=0.7, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, input, target):
        input_prob = torch.sigmoid(input)
        hard_easy_weight = (1 - input_prob) * target + input_prob * (1 - target)
        posi_nega_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = (posi_nega_weight * torch.pow(hard_easy_weight, self.gamma)).detach()
        focal_loss = F.binary_cross_entropy_with_logits(input, target, weight=focal_weight)
        return focal_loss


# class mutilabel_softxmax(object):
#     def __init__(self):
#         super(mutilabel_softxmax, self).__init__()
#     def __call__(self, input, target):
#         # if

class WeightedFocalLoss(object):
    def __init__(self, pos=None, Epoch=20):
        super(WeightedFocalLoss, self).__init__()
        self.pos = pos
        self.neg = [1 - i for i in pos]
        self.EPOCH = Epoch

    def __call__(self, input, target):
        size = target.size()
        pos = [math.exp(-i) for i in self.pos]
        neg = [math.exp(-i) for i in self.neg]
        # neg = 1-pos
        pos = torch.Tensor(pos).cuda()
        neg = torch.Tensor(neg).cuda()
        # neg = 1-pos
        # gamma = torch.Tensor(self.neg).cuda()
        input_prob = torch.sigmoid(input)
        pos = pos.expand(size[0], 6).cuda()
        neg = neg.expand(size[0], 6).cuda()
        hard_easy_weight = (1 - input_prob) * target + input_prob * (1 - target)
        posi_nega_weight = torch.mul(pos, target) + torch.mul(neg, (1 - target))
        focal_weight = (posi_nega_weight * torch.pow(hard_easy_weight, 0.75)).detach()  # 0,0.5,0.75,1,2
        focal_loss = F.binary_cross_entropy_with_logits(input, target, weight=focal_weight)
        return focal_loss


class WeightedFocalLoss_v2(object):
    def __init__(self, distribution=None, Epoch=20):
        super(WeightedFocalLoss_v2, self).__init__()
        self.distribution = distribution
        self.w0_pos = 1 - self.distribution[1] / self.distribution[0]
        self.w0_neg = self.distribution[1] / self.distribution[0]
        self.w1_pos = 1 - self.distribution[2] / self.distribution[1]
        self.w1_neg = self.distribution[2] / self.distribution[0]
        self.w2_pos = 1 - self.distribution[3] / self.distribution[1]
        self.w2_neg = self.distribution[3] / self.distribution[0]
        self.w3_pos = 1 - self.distribution[4] / self.distribution[1]
        self.w3_neg = self.distribution[4] / self.distribution[0]
        self.w4_pos = 1 - self.distribution[5] / self.distribution[1]
        self.w4_neg = self.distribution[5] / self.distribution[0]
        self.w5_pos = 1 - self.distribution[6] / self.distribution[1]
        self.w5_neg = self.distribution[6] / self.distribution[0]
        # self.w0_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[1]) * 2)
        # self.w0_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[1]) * 2)
        # self.w0_pos = self.distribution[0] / (self.distribution[1] * 2)
        # self.w0_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[1]) * 2)
        # self.w1_pos = self.distribution[0] / (self.distribution[2] * 2)
        # self.w1_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[2]) * 2)
        # self.w2_pos = self.distribution[0] / (self.distribution[3] * 2)
        # self.w2_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[3]) * 2)
        # self.w3_pos = self.distribution[0] / (self.distribution[4] * 2)
        # self.w3_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[4]) * 2)
        # self.w4_pos = self.distribution[0] / (self.distribution[5] * 2)
        # self.w4_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[5]) * 2)
        # self.w5_pos = self.distribution[0] / (self.distribution[6] * 2)
        # self.w5_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[6]) * 2)
        self.pos = [self.w0_pos, self.w1_pos, self.w2_pos, self.w3_pos, self.w4_pos, self.w5_pos]
        self.neg = [self.w0_neg, self.w1_neg, self.w2_neg, self.w3_neg, self.w4_neg, self.w5_neg]
        self.EPOCH = Epoch

    def __call__(self, input, target, epoch):
        size = target.size()
        pos = torch.Tensor(self.pos).cuda()
        neg = torch.Tensor(self.neg).cuda()
        gamma = 1 / pos.cuda()
        # gamma[0] = 0
        input_prob = torch.sigmoid(input)
        pos = pos.expand(size[0], 6).cuda()
        neg = neg.expand(size[0], 6).cuda()

        hard_easy_weight = (1 - input_prob) * target + input_prob * (1 - target)
        posi_nega_weight = torch.mul(pos, target) + torch.mul(neg, (1 - target))
        focal_weight = (posi_nega_weight * torch.pow(hard_easy_weight, 2)).detach()
        focal_loss = F.binary_cross_entropy_with_logits(input, target, weight=focal_weight)
        return focal_loss


class FocalLoss2d(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        weight = Variable(self.weight)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


class WieghtBinaryEntropyLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(WieghtBinaryEntropyLoss, self).__init__()
        self.alpha = alpha
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """

        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = batch_size / cfg.batchsize

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
        # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


def multilabel_categorical_crossentropy(y_pred, y_true):
    '''多标签分类的交叉熵
       说明：y_true和y_pred的shape一致，y_true的元素非0即1，
        1表示对应的类为目标类，0表示对应的类为非目标类。
    '''
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


class WeightFocalLoss(nn.Module):
    def __init__(self, distribution=None, size_average=True):
        super(WeightFocalLoss, self).__init__()
        self.distribution = distribution
        self.size_average = size_average
        self.w0_pos = self.distribution[0] / (self.distribution[1] * 2)
        self.w0_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[1]) * 2)
        self.w1_pos = self.distribution[0] / (self.distribution[2] * 2)
        self.w1_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[2]) * 2)
        self.w2_pos = self.distribution[0] / (self.distribution[3] * 2)
        self.w2_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[3]) * 2)
        self.w3_pos = self.distribution[0] / (self.distribution[4] * 2)
        self.w3_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[4]) * 2)
        self.w4_pos = self.distribution[0] / (self.distribution[5] * 2)
        self.w4_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[5]) * 2)
        self.w5_pos = self.distribution[0] / (self.distribution[6] * 2)
        self.w5_neg = self.distribution[0] / ((self.distribution[0] - self.distribution[6]) * 2)
        self.pos = [self.w0_pos, self.w1_pos, self.w2_pos, self.w3_pos, self.w4_pos, self.w5_pos]
        self.neg = [self.w0_neg, self.w1_neg, self.w2_neg, self.w3_neg, self.w4_neg, self.w5_neg]

    def forward(self, input, target):
        size = target.size()
        # print(size)
        pos = torch.Tensor(self.pos).cuda()
        gama = 1 / pos
        weight = Variable(pos)

        # compute the negative likelyhood
        loss = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')
        logpt = loss(input, target)
        # # logpt = F.binary_cross_entropy_with_logits(input, target, pos_weight=weight).cuda()
        # pt = torch.sigmoid(input)
        #
        # # compute the loss
        # focal_loss = ((1 - pt)**2) * logpt
        # loss = focal_loss.mean()
        return logpt
        # size = target.size()
        # gamma = gamma
        # alpha
        # for
        # print(alpha)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         if target[i][j] == 1:
        #             alpha = self.pos[j]
        #             gamma = 1/self.pos[j]
        #         if target[i][j] == 0:
        #             alpha = self.neg[j]
        #             gamma = 1/self.neg[j]
        #         loss = - alpha * log_error[i][j]
        #         loss_all = loss+loss_all
        # loss_mean = loss_all/(size[0]*size[1])
        # loss = -1 * (1 - error) ** 2 * log_error
        # if self.size_average:
        #     return loss.mean()
        # else:
        #     return loss.sum()


def getdist(labels):
    size = labels.size()
    dist = [0, 0, 0, 0, 0, 0]
    labels = torch.transpose(labels, 0, 1)
    i = 0
    for label in labels:
        label = label.cpu().numpy().tolist()
        count = Counter(label)[1]
        dist[i] = count / size[0]
        i = i + 1
    return dist


if __name__ == '__main__':
    from collections import Counter

    # hard_easy_weight = torch.Tensor([2,2,2,2,2])
    # gamma = torch.Tensor([2,1,1,1,1])
    # a = torch.pow(hard_easy_weight, gamma)
    torch.cuda.set_device(0)
    labels = torch.randint(0, 2, (16, 6)).float().cuda()
    dist = getdist(labels)
    pred = torch.rand(16, 6).float().cuda()
    loss = WeightedFocalLoss(dist)
    # loss = multilabel_categorical_crossentropy(pred,labels)
    # loss = FocalLoss()
    output = loss(pred, labels, 4)
    print(output)
    # print(output)
    # loss.backward()
