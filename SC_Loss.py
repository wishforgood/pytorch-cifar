import torch
import torch.nn as nn
import torch.nn.functional as F
import utils2
from torch.autograd import Variable
from random import randint
import math
import numpy as np


class SC_Loss(nn.Module):
    def __init__(self, target_list):
        distance_list, distance_rank = utils2.calculate_wordnet_distance(target_list)
        self.distance_rank = distance_rank
        self.class_num = len(distance_rank[0])
        self.distance_matrix = Variable(torch.FloatTensor(distance_list))
        super(SC_Loss, self).__init__()
        # np.save('cub_distance', self.distance_matrix.data.numpy())

    def forward(self, input, target):
        sc_regulirizer = self.calculate_sc(input, target)
        theata = 0.6
        return theata * F.cross_entropy(input, target) + (1 - theata) * sc_regulirizer

    def calculate_sc(self, input, target):
        alpha = len(input[0]) - 1
        # alpha = 1
        CtSTE = 0
        for batch_index in range(len(input)):
            input_p = input[batch_index]
            yi = input_p[self.distance_rank[target.data[batch_index]][1]]
            for class_index in range(2, self.class_num - 1):
                yj = input_p[self.distance_rank[target.data[batch_index]][class_index]]
                k = randint(class_index + 1, self.class_num - 1)
                yk = input_p[self.distance_rank[target.data[batch_index]][k]]
                midij = pow((1 + pow((yi - yj), 2) / alpha), (-1 * (alpha + 1) / 2))
                midik = pow((1 + pow((yi - yk), 2) / alpha), (-1 * (alpha + 1) / 2))
                pijk = midij / (midij + midik)
                logpijk = math.log(pijk, 2)
                CtSTE += logpijk
        return -1 * CtSTE / len(input) / (self.class_num - 1)

    def calculate_t_sne(self, input, target):
        for batch_index in range(len(input)):
            semantic_distance_vector = self.distance_matrix.narrow(0, target[0].data[0], 1)
            for i in range(1, len(input)):
                semantic_distance_vector = torch.cat(
                    [semantic_distance_vector, self.distance_matrix.narrow(0, target[i].data[0], 1)])

        return
