import torch
import torch.nn as nn
import torch.nn.functional as func


class SimilarLoss(nn.Module):
    """
    用于相似度网络训练的损失值函数
    """
    def __init__(self, margin=1.0):
        """

        Args:
            margin: 预期将不同类样本分开的距离
        """
        super().__init__()
        self.margin = margin

    def forward(self, outputs, labels):
        x0 = outputs[0]
        x1 = outputs[1]
        y = labels[0] == labels[1]

        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = torch.sum(dist_sq[y]) + torch.sum(torch.pow(dist[~y], 2))
        loss = loss / 2.0 / len(labels)
        return loss
