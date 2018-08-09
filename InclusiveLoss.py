from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils2
from torch.autograd import Variable
import math
import numpy as np


class InclusiveLoss(nn.Module):
    def __init__(self, target_list, resume_distance):
        super(InclusiveLoss, self).__init__()
        if resume_distance is not None:
            self.distance_matrix_initial = resume_distance.clone()
            self.distance_matrix = nn.Parameter(resume_distance.clone().cpu().data)
        else:
            self.distance_matrix_initial = Variable(
                torch.FloatTensor(utils2.calculate_wordnet_distance(target_list))).cuda()
            self.distance_matrix = nn.Parameter(
                torch.FloatTensor(utils2.calculate_wordnet_distance(target_list)))
        # self.distance_matrix.register_hook(print)
        self.parameter_list = nn.ParameterList()
        self.parameter_list.append(self.distance_matrix)
        np.save('cub_distance', self.distance_matrix.data.numpy())

    def forward(self, input, target):
        # ap_distance_matrix = c_c.distance_matrix()
        # print(self.distance_matrix)
        # input.register_hook(print)
        semantic_distance_vector = self.distance_matrix.narrow(0, target[0].data[0], 1)
        # ap_distance_vector = ap_distance_matrix.narrow(0, target[0].data[0], 1)
        for i in range(1, len(input)):
            semantic_distance_vector = torch.cat(
                [semantic_distance_vector, self.distance_matrix.narrow(0, target[i].data[0], 1)])
            #  ap_distance_vector = torch.cat(
            #     [ap_distance_vector, ap_distance_matrix.narrow(0, target[i].data[0], 1)])
        # inclusive_regulirizer = - torch.sum(semantic_distance_vector.cuda() * F.log_softmax(input, 1)) / (len(
        #     self.distance_matrix) * len(input))
        inclusive_regulirizer = - torch.sum(semantic_distance_vector.cuda() * F.log_softmax(input, 1)) / (
        len(input)) / 10
        # ap_regulirizer = -0.2 * torch.sum(ap_distance_vector.cuda() * F.log_softmax(input, 0)) / (len(
        #     self.distance_matrix) * len(input))
        # return F.cross_entropy(input, target) + 0.6 * (inclusive_regulirizer + ap_regulirizer)
        # print(inclusive_regulirizer)
        # print(math.pow(torch.dist(self.distance_matrix, self.distance_matrix_initial, 2), 2))
        return inclusive_regulirizer + 0.5 * torch.pow(
            torch.dist(self.distance_matrix,
                       self.distance_matrix_initial,
                       2), 2)
        # return F.cross_entropy(input, target) + 0.1 * inclusive_regulirizer
        # return F.cross_entropy(input, target)


class RecognitionAccuracy(object):
    def __init__(self, distance_matrix):
        self.added_acc = 0
        self.total_number = 0
        self.distance_matrix = distance_matrix.data

    def recognition_accuracy(self, output, target):
        output = F.softmax(output)
        output = output.cpu().data
        target = target.cpu().data
        upper = 0
        downer1 = 0
        downer2 = 0
        for p in output:
            p = abs(p - output[target[0]])
            for d in self.distance_matrix[target[0]]:
                upper += p * d
                downer1 += p * p
                downer2 += d * d
        return upper / math.sqrt(downer1 * downer2)

    def update(self, outputs, targets):
        for i in range(len(outputs)):
            self.added_acc += self.recognition_accuracy(outputs[i], targets[i])
            self.total_number += 1

    def output(self):
        return self.added_acc / self.total_number

    def re_init(self):
        self.added_acc = 0
        self.total_number = 0


class RankingCorrelation(object):
    def __init__(self, distance_matrix):
        self.added_acc = 0
        self.total_number = 0
        self.distance_matrix = distance_matrix.data

    def ranking_correlation(self, output, target):
        output = F.softmax(output)
        output = output.cpu().data
        target = target.cpu().data
        output = output.sort()[1]
        upper = 0
        downer = 0

        distance_rank = self.distance_matrix[target[0]].sort()[1]
        size = len(output)

        # top_list = [index for index, i in enumerate(output) if i < 6]
        # size = len(top_list)
        # output = torch.gather(output, 0, torch.LongTensor(top_list))
        # distance_rank = torch.gather(self.distance_matrix[target[0]], 0, torch.LongTensor(top_list)).sort()[1]

        for i in range(size):
            for j in range(size):
                upper += (output[j] - output[i]) * (distance_rank[j] - distance_rank[i])
                downer += (output[j] - output[i]) ** 2
        return upper / downer.__float__()

    def update(self, outputs, targets):
        for i in range(len(outputs)):
            self.added_acc += self.ranking_correlation(outputs[i], targets[i])
            self.total_number += 1

    def output(self):
        return self.added_acc / self.total_number.__float__()

    def re_init(self):
        self.added_acc = 0
        self.total_number = 0


class ClusterCenters(object):
    def __init__(self, nclass, dim):
        self.c_c_c = []
        self.c_c_n = []
        self.nclass = nclass
        self.dim = dim
        for i in range(self.nclass):
            self.c_c_c.append(Variable(torch.zeros(dim), requires_grad=False).cuda())
            self.c_c_n.append(Variable(torch.zeros(1), requires_grad=False).cuda())
        self.c_c = []
        self.c_n = []
        for item in self.c_c_c:
            self.c_c.append(item.clone())
        for item in self.c_c_n:
            self.c_n.append(item.clone())

    def record(self, features, target):
        target = target.data
        for i in range(len(target)):
            class_index = target[i] - 1
            self.c_c[class_index] += Variable(torch.FloatTensor(features[i].cpu().data), requires_grad=False).cuda()
            self.c_n[class_index] += 1

    def update(self):
        self.c_c_c = []
        self.c_c_n = []
        for item in self.c_c:
            self.c_c_c.append(item.clone())
        for item in self.c_n:
            self.c_c_n.append(item.clone())
        self.c_c = []
        self.c_n = []
        for i in range(self.nclass):
            self.c_c.append(Variable(torch.zeros(self.dim), requires_grad=False).cuda())
            self.c_n.append(Variable(torch.zeros(1), requires_grad=False).cuda())

    def output(self):
        out_c_c = []
        for i in range(self.nclass):
            out_c_c.append(self.c_c[i] / self.c_n[i])
        return out_c_c

    def distance_matrix(self):
        out_c_c = []
        metric = torch.nn.L1Loss()
        for i in range(self.nclass):
            if self.c_c_n[i].cpu().data[0] != 0:
                out_c_c.append(self.c_c_c[i] / self.c_c_n[i])
            else:
                out_c_c.append(self.c_c_c[i])
        distance_matrix = Variable(torch.zeros(self.nclass, self.nclass), requires_grad=False)
        for i in range(self.nclass):
            for j in range(self.nclass):
                distance_matrix[i, j] = 0.3 - metric(out_c_c[j], out_c_c[i])
        return distance_matrix
