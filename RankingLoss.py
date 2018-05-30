import torch
import torch.nn as nn
import torch.nn.functional as F
import utils2
from torch.autograd import Variable
import math


class RankingLoss(nn.Module):
    def __init__(self, target_list):
        self.distance_matrix = Variable(torch.FloatTensor(utils2.calculate_wordnet_distance(target_list)))
        super(RankingLoss, self).__init__()

    def forward(self, input, target):
        distance_vector = self.distance_matrix.narrow(0, target[0].data[0], 1)
        for i in range(1, len(input)):
            distance_vector = torch.cat([distance_vector, self.distance_matrix.narrow(0, target[i].data[0], 1)])
        ranking_loss = Variable(torch.FloatTensor([0]))
        for i in range(0, len(input)):
            distance_rank = distance_vector[i].sort()[1]
            for j in range(10):
                if j != target[i].data[0]:
                    ranking_loss += torch.max(Variable(torch.FloatTensor([0])),
                                              1 + distance_rank[j].float() - distance_rank[target[i].data[0]].float())
        return F.cross_entropy(input, target) + ranking_loss.cuda() / len(input)
        # return F.cross_entropy(input, target)


class MLL(nn.Module):
    def __init__(self, target_list):
        self.distance_matrix = Variable(torch.FloatTensor(utils2.calculate_wordnet_distance(target_list)))
        super(MLL, self).__init__()

    def forward(self, input, target):
        distance_vector = self.distance_matrix.narrow(0, target[0].data[0], 1)
        for i in range(1, len(input)):
            distance_vector = torch.cat([distance_vector, self.distance_matrix.narrow(0, target[i].data[0], 1)])
        ranking_loss = Variable(torch.FloatTensor([0])).cuda()
        for i in range(0, len(input)):
            distance_rank = distance_vector[i].sort()[1]
            for j in range(10):
                if j != target[i].data[0]:
                    ranking_loss += torch.exp(-input[i, target[i].data[0]] + input[i, j])
        return F.cross_entropy(input, target) + 0.5 * ranking_loss / (
            (len(self.distance_matrix) - 1) * len(input))
        # return F.cross_entropy(input, target)
