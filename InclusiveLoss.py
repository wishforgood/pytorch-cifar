import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class InclusiveLoss(nn.Module):
    def __init__(self, target_list):
        utils.calculate_wordnet_distance(target_list)
        super(InclusiveLoss, self).__init__()

    def forward(self, input, target):

        return F.cross_entropy(input, target)
