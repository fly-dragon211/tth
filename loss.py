# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def l2norm(X, eps=1e-13, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps + 1e-14
    X = torch.div(X, norm)
    return X


def l1norm(X, eps=1e-13, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps + 1e-14
    X = torch.div(X, norm)
    return X


def normalization(X, dim=1):
    # 按行归一化
    _range = np.max(X) - np.min(X)
    return (X - np.min(X)) / _range


def cosine_sim(query, retrio):
    """Cosine similarity between all the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)
    return query.mm(retrio.t())

def vector_cosine_sim(query, retrio):
    """Cosine similarity between  the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)
    return torch.sum(torch.mul(query, retrio),dim=1).unsqueeze(0)


def hist_sim(im, s, eps=1e-14):
    bs = im.size(0)
    im = im.unsqueeze(1).expand(-1,bs,-1)
    s = s.unsqueeze(0).expand(bs,-1,-1)
    intersection = torch.min(im,s).sum(-1)
    union = torch.max(im,s).sum(-1) + eps
    score = intersection / union
    return score


class MarginRankingLoss(nn.Module):
    """
    Compute margin ranking loss
    arg input: (batchsize, subspace) and (batchsize, subspace)
    """
    def __init__(self, margin=0, measure='cosine', max_violation=False,
                 cost_style='sum', direction='bidir', device=torch.device('cpu')):
        """
        :param margin:
        :param measure: cosine 余弦相似度， hist_sim 扩展 jaccard 相似度
        :param max_violation:
        :param cost_style: 把所有误差相加 sum，还是取平均值 mean
        :param direction: compare every diagonal score to scores in its column and row
        """
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'hist':
            self.sim = hist_sim
        else:
            raise Exception('Not implemented.')

        self.max_violation = max_violation

    def forward(self, s, im):
        device = s.device
        # compute image-sentence score matrix
        scores = self.sim(im, s)  #
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)  # 扩展维度
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        I = I.to(device)

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in ['i2t', 'bidir']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)  # clamp 最大最小裁剪
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'bidir']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = torch.zeros(1).to(device)
        if cost_im is None:
            cost_im = torch.zeros(1).to(device)

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()
