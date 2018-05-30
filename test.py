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

# from __future__ import print_function

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
#
# import torchvision
# import torchvision.transforms as transforms
#
# import os
# import argparse
# import pickle
#
# from models import *
# from torch.autograd import Variable
# import InclusiveLoss
# import visdom
# import matplotlib.pyplot as plt
# import numpy as np
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
# ])
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# net = torch.load('./checkpoint/ckpt.t7')['net']
# net.cuda()
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
# cudnn.benchmark = True
# target_list = {'airplane', 'automobile', 'bird', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
# criterion = InclusiveLoss.InclusiveLoss(target_list).cuda()
# distance_matrix = criterion.distance_matrix
# reco_acc = InclusiveLoss.RankingCorrelation(distance_matrix)
#
# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         plt.imshow(np.transpose(inputs[0].numpy(),[1, 2, 0]))
#         inputs, targets = inputs.cuda(), targets.cuda()
#         inputs, targets = Variable(inputs, volatile=True), Variable(targets)
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#
#         test_loss += loss.data[0]
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()
#         # reco_acc.update(outputs, targets)
#         # print(reco_acc.output())
#     # r_a = reco_acc.output()
#     # reco_acc.re_init()
#
#     print(epoch, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#           % (test_loss, 100. * correct / total, correct, total))
#
#     # Save checkpoint.
#     acc = 100. * correct / total
#     return acc
# test_r_a = test(299)

from models import *
from torch.autograd import Variable
import InclusiveLoss
import visdom
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
net = torch.load('./checkpoint/ckpt2.t7')['net']
net.cuda()

img_pil = Image.open('rose.png')
img_tensor = transform_test(img_pil)
img_tensor.unsqueeze_(0)
img_variable = Variable(img_tensor).cuda()
output = net(img_variable)
posi = F.softmax(output, 1)
for item in posi:
    print(item.data)
# net = torch.load('./checkpoint/cifar_ckpt.t7')['net']
# net.cuda()
# output = net(img_variable)
# posi = F.softmax(output)
# print(posi)
