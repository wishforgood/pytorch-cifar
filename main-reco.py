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
criterion = InclusiveLoss.InclusiveLoss(target_list).cuda()
distance_matrix = criterion.distance_matrix
reco_acc = InclusiveLoss.RankingCorrelation(distance_matrix)
# criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# for group in optimizer.param_groups:
#     group.setdefault('initial_lr', args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 150], gamma=0.1, last_epoch=-1)


# Training
def train(epoch, c_c):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        # outputs = net(inputs)
        features = net_features(inputs)
        features = F.avg_pool2d(features, 4)
        features = features.view(features.size(0), -1)
        # if epoch > 30:
        #     c_c.record(features, targets)
        outputs = net_classifier(features)
        loss = criterion(outputs, targets, c_c)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        reco_acc.update(outputs, targets)
        print(reco_acc.output())
    r_a = reco_acc.output()
    reco_acc.re_init()
    # if epoch > 29:
    #     c_c.update()
    print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss, 100. * correct / total, correct, total))
    # return 100. * correct / total, c_c
    return r_a, c_c


def test(epoch, c_c):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        features = net_features(inputs)
        features = F.avg_pool2d(features, 4)
        features = features.view(features.size(0), -1)
        outputs = net_classifier(features)
        loss = criterion(outputs, targets, c_c)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        reco_acc.update(outputs, targets)
        print(reco_acc.output())
    r_a = reco_acc.output()
    reco_acc.re_init()

    print(epoch, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss, 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt2.t7')
        best_acc = acc
    # return acc
    return r_a

vis = visdom.Visdom()
win = vis.line(X=np.array([0]), Y=np.array([[0, 0]]),
               opts={'legend': ['train', 'test'], 'xlabel': 'epoch', 'ylabel': 'acc'})
c_c = InclusiveLoss.ClusterCenters(100, 512)

for epoch in range(start_epoch + 1, start_epoch + 300):
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    # class_centers = torch.rand(512)
    train_r_a, c_c = train(epoch, c_c)
    test_r_a = test(epoch, c_c)
    vis.line(X=np.array([epoch]), Y=np.array([[train_r_a, test_r_a]]),
             opts={'legend': ['train', 'test'], 'xlabel': 'epoch', 'ylabel': 'acc'}, update='append', win=win)