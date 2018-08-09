'''Train CIFAR10 with PyTorch.'''
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
import RankingLoss
import visdom

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
top2_best = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=16)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/epoch_29.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('epoch: ' + str(start_epoch))
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = LeNet()
    # net = ResNet152()
    # net = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100)
    # net = CifarResNeXt(cardinality=8, depth=29, nlabels=100, base_width=64, widen_factor=4)
net_features = nn.Sequential(*list(net.children())[:-1])
net_classifier = list(net.modules())[-1]
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net_features = torch.nn.DataParallel(net_features, device_ids=range(torch.cuda.device_count()))
    net_classifier = torch.nn.DataParallel(net_classifier, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
with open('data/cifar-100-python/meta', 'r') as f:
    target_list = pickle.load(f)
target_list = target_list['fine_label_names']
# target_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# criterion = InclusiveLoss.InclusiveLoss(target_list, None).cuda()
# distance_matrix = criterion.distance_matrix
# reco_acc = InclusiveLoss.InclusiveLoss(distance_matrix)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer.add_param_group({'params': criterion.parameters()})
# for group in optimizer.param_groups:
#     group.setdefault('initial_lr', args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 120], gamma=0.2, last_epoch=-1)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    top1 = AverageMeter()
    top2 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        features = net_features(inputs)
        features = F.avg_pool2d(features, 4)
        features = features.view(features.size(0), -1)
        outputs = net_classifier(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        prec1, prec2 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))
    print(epoch, len(trainloader), 'Loss: %.3f | Top1: %.3f | Top2: %.3f'
          % (train_loss, top1.avg, top2.avg))
    return top1.avg.data[0]


def test(epoch):
    global best_acc
    global top2_best
    net.eval()
    test_loss = 0
    top1 = AverageMeter()
    top2 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        features = net_features(inputs)
        features = F.avg_pool2d(features, 4)
        features = features.view(features.size(0), -1)
        outputs = net_classifier(features)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        prec1, prec2 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))

    print(epoch, len(testloader), 'Loss: %.3f | Top1: %.3f | Top2: %.3f'
          % (test_loss, top1.avg, top2.avg))
    if top1.avg.data[0] > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': top1.avg.data[0],
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt2.t7')
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


vis = visdom.Visdom()
win = vis.line(X=np.array([0]), Y=np.array([[0, 0]]),
               opts={'legend': ['train', 'test'], 'xlabel': 'epoch', 'ylabel': 'acc'})
# c_c = InclusiveLoss.ClusterCenters(10, 512)

for epoch in range(start_epoch + 1, start_epoch + 300):
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    # class_centers = torch.rand(512)
    train_r_a = train(epoch)
    test_r_a = test(epoch)
    vis.line(X=np.array([epoch]), Y=np.array([[train_r_a, test_r_a]]),
             opts={'legend': ['train', 'test'], 'xlabel': 'epoch', 'ylabel': 'acc'}, update='append', win=win)
