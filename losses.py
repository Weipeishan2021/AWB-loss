import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opts import parser
import scipy.stats
from torch.autograd import Variable

args = parser.parse_args()


class AWB(nn.Module):
    def __init__(self, reduction='mean'):
        super(AWB, self).__init__()
        self.reduction = reduction

    def get_loss2(self, pt, target, Alpha):

        p_avg, p_std = self.p_avg_std(pt, target)
        Alpha = F.softmax(Alpha, 0)
        loss2 = p_std / p_avg * Alpha
        loss2 = loss2.gather(0, target.data.view(-1))
        return loss2

    def p_avg_std(self, pt, labels):
        p_avg = torch.zeros(args.num_classes).cuda()
        p_std = torch.zeros(args.num_classes).cuda()

        for i in range(0, args.num_classes):
            index = torch.nonzero(labels == torch.tensor(i))
            index = index[:, 0]
            p_i = pt.gather(0, index)
            p_avg[i] = self.cal_avg(p_i)
            p_std[i] = self.cal_std(p_i)

        return p_avg, p_std

    def cal_avg(self, pt):

        if pt.shape == (0,):
            return 1
        else:
            return torch.mean(pt)

    def cal_std(pself, pt):
        if pt.shape == (0,) or pt.shape == (1,):
            return 0
        else:
            return torch.std(pt)

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)
            logits = logits.transpose(1, 2)
            logits = logits.contiguous().view(-1, logits.size(2))
        target = target.view(-1, 1)

        pt1 = F.softmax(logits, 1)
        pt = pt1.gather(1, target).view(-1)

        c_sum = torch.zeros(args.num_classes).cuda()
        Alpha = torch.zeros(args.num_classes).cuda()

        for i in range(args.num_classes):
            c_sum[i] = torch.sum(target == i)
        c_sum_max = c_sum.max()
        for i in range(args.num_classes):
            if c_sum[i] != 0:
                Alpha[i] = torch.log(c_sum_max / c_sum[i]) + 1
            else:
                Alpha[i] = 0


        log_gt = torch.log(pt + 10e-7)
        loss1 = -1 * log_gt.cuda()
        loss1, _ = self.p_avg_std(loss1, target)
        loss1 = loss1 * Alpha
        loss2 = self.get_loss2(pt, target, Alpha)

        loss = loss1.mean() + loss2.mean()

        return loss


