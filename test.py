# from utils2 import calculate_wordnet_distance
# import pickle
# with open('data/cifar-100-python/meta', 'r') as f:
#     target_list = pickle.load(f)
# target_list = target_list['fine_label_names']
# print(calculate_wordnet_distance(target_list))

# import os
#
# work_path = '/root/mounted_device/tong/imagenet'
# source_path = '/root/mounted_device/tong/dataset'
# train_path = os.path.join(source_path, 'CLS-LOC/ILSVRC2012_img_train')
# train_target_path = os.path.join(work_path, 'train')
# for item in os.listdir(train_path):
#     item_path = os.path.join(train_path, item)
#     os.system('mkdir ' + train_target_path + '/' + item[:-4])
#     os.system('tar -xvf ' + item_path + ' -C ' + train_target_path + '/' + item[:-4])

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle

from models import *
from torch.autograd import Variable
import InclusiveLoss
import visdom
import matplotlib.pyplot as plt
import numpy as np

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
net = torch.load('./checkpoint/ckpt.t7')['net']
net_features = nn.Sequential(*list(net.children())[:-1])
net_classifier = list(net.modules())[-1]
net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
# target_list = {'airplane', 'automobile', 'bird', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
with open('data/cifar-100-python/meta', 'r') as f:
    target_list = pickle.load(f)
target_list = target_list['fine_label_names']
criterion = InclusiveLoss.InclusiveLoss(target_list, None).cuda()
best_acc  = 0
top2_best = 0

def test(epoch):
    global best_acc
    global top2_best
    net.eval()
    test_loss = 0
    top1 = AverageMeter()
    top2 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        features = net_features(inputs)
        features = F.avg_pool2d(features, 4)
        features = features.view(features.size(0), -1)
        outputs = net_classifier(features)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))

    print(epoch, len(testloader), 'Loss: %.3f | Top1: %.3f | Top2: %.3f'
          % (test_loss, top1.avg, top2.avg))
    if top1.avg.data[0] > best_acc:
        best_acc = top1.avg.data[0]
    if top2.avg.data[0] > top2_best:
        top2_best = top2.avg.data[0]
    print('Best acc: %f', best_acc)
    print('Best top2: %f', top2_best)
    return top1.avg.data[0]


class AverageMeter(object):
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


test_r_a = test(299)

# from models import *
# from torch.autograd import Variable
# import InclusiveLoss
# import visdom
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
# ])
# net = torch.load('./checkpoint/ckpt.t7')['net']
# net.cuda()
#
# img_pil = Image.open('rose.png')
# img_tensor = transform_test(img_pil)
# img_tensor.unsqueeze_(0)
# img_variable = Variable(img_tensor).cuda()
# output = net(img_variable)
# posi = F.softmax(output, 1)
# for item in posi:
#     print(item.data)
# net = torch.load('./checkpoint/cifar_ckpt.t7')['net']
# net.cuda()
# output = net(img_variable)
# posi = F.softmax(output)
# print(posi)
