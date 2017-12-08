import torch
import torch.nn as nn
import torch.nn.functional as F
import utils2
from torch.autograd import Variable


class InclusiveLoss(nn.Module):
    def __init__(self, target_list):
        self.distance_matrix = Variable(torch.FloatTensor(utils2.calculate_wordnet_distance(target_list)))
        super(InclusiveLoss, self).__init__()

    def forward(self, input, target):
        distance_vector = self.distance_matrix.narrow(0, target[0].data[0], 1)
        for i in range(1, len(input)):
            distance_vector = torch.cat([distance_vector, self.distance_matrix.narrow(0, target[i].data[0], 1)])
        inclusive_regulirizer = -0.5 * torch.sum(distance_vector.cuda() * F.log_softmax(input, 0)) / (len(
            self.distance_matrix) * len(input))
        return F.cross_entropy(input, target) + inclusive_regulirizer
        # return F.cross_entropy(input, target)
