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
import json

from models import *
from torch.autograd import Variable
import InclusiveLoss
import RankingLoss
import visdom
from utils2 import FineTuneModel_Dense

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4805, 0.456, 0.4063), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4805, 0.456, 0.4063), (0.2675, 0.224, 0.225)),
])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset = torchvision.datasets.ImageFolder(root='/root/mounted_device/tong/dataset/CUB_200_2011/train',
                                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=48, shuffle=True, num_workers=16)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset = torchvision.datasets.ImageFolder(root='/root/mounted_device/tong/dataset/CUB_200_2011/val',
                                           transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=48, shuffle=False, num_workers=16)

# Model
print('==> Building model..')
net = torchvision.models.densenet121(pretrained=True)
net_features = nn.Sequential(*list(net.children())[:-1])
net_classifier = nn.Sequential(nn.Linear(1024, 200))
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
# net = VGG('VGG19')
# net = torch.load('resnet18-5c106cde.pth')
# net = ResNet18()
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
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net_features = torch.nn.DataParallel(net_features, device_ids=range(torch.cuda.device_count()))
    net_classifier = torch.nn.DataParallel(net_classifier, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])

with open('target_list.json', 'r') as f:
    target_list = json.load(f)
# target_list = {'airplane', 'automobile', 'bird', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
criterion = InclusiveLoss.InclusiveLoss(target_list).cuda()
# distance_matrix = criterion.distance_matrix
# reco_acc = InclusiveLoss.RankingCorrelation(distance_matrix)
# criterion = nn.CrossEntropyLoss().cuda()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


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
        features = net_features(inputs)
        features = F.relu(features, inplace=True)
        features = F.avg_pool2d(features, kernel_size=7).view(features.size(0), -1)
        c_c.record(features, targets)
        outputs = net_classifier(features)
        loss = criterion(outputs, targets, c_c)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    # reco_acc.update(outputs, targets)
    #     print(reco_acc.output())
    # r_a = reco_acc.output()
    # reco_acc.re_init()
    c_c.update()
    print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss, 100. * correct / total, correct, total))
    return 100. * correct / total
    # return r_a


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
        features = F.relu(features, inplace=True)
        features = F.avg_pool2d(features, kernel_size=7).view(features.size(0), -1)
        outputs = net_classifier(features)
        loss = criterion(outputs, targets, c_c)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    # reco_acc.update(outputs, targets)
    #     print(reco_acc.output())
    # r_a = reco_acc.output()
    # reco_acc.re_init()

    print(epoch, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss, 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc
    # return r_a


vis = visdom.Visdom()
win = vis.line(X=np.array([0]), Y=np.array([[0, 0]]),
               opts={'legend': ['train', 'test'], 'xlabel': 'epoch', 'ylabel': 'acc'})
c_c = InclusiveLoss.ClusterCenters(200, 1024)
for epoch in range(start_epoch, start_epoch + 300):
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    train_r_a = train(epoch, c_c)
    test_r_a = test(epoch, c_c)
    vis.line(X=np.array([epoch]), Y=np.array([[train_r_a, test_r_a]]),
             opts={'legend': ['train', 'test'], 'xlabel': 'epoch', 'ylabel': 'acc'}, update='append', win=win)
