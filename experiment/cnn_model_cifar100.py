'''
This script input CIFAR100 dataset to different network
and save mutual information about I(X; T), I(T,Y)

network choice: Baseline, Alexnet, Vgg, Resnet, Densenet

'''

import sys
sys.path.append('..')
import argparse
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.mi_cal import *
import os
import logging
import network.densenet as densenet
import network.resnet as resnet
import network.vgg as vgg
import network.alexnet as alexnet




parser = argparse.ArgumentParser(description='Network and training parameters choices')
# Network choices
parser.add_argument('--network', type=str, default='Baseline', metavar='NET',
                    help='Network. Option: Baseline, Alexnet, Vgg, Resnet, Densenet(default: Baseline)')
# Training parameters settings
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', type=int, default=10, metavar='N',
                    help='lr decay interval with epoch (default: 10)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--mask', type=int, default=3000, metavar='N',
                    help='The number of test mask')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='which gpu device (default: 0)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



# log setting
log_file = '../log/cifar10/' + args.network + '.log'
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# print setting
logging.info(args)

# Data preparation
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('../data/cifar100', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('../data/cifar100', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)






if args.network == 'Baseline':
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(500, 200)
            self.fc2 = nn.Linear(200, 100)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 500)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x, F.log_softmax(x)


    model = Net()

elif args.network == 'Alexnet':
    model = alexnet.AlexNet(num_classes=100)
elif args.network == 'Vgg':
    model = vgg.vgg16()
    print(model)
elif args.network == 'Resnet':
    model = resnet.ResNet50(num_classes=100)
elif args.network == 'Densenet':
    model = densenet.densenet_cifar(num_classes=100)
    #print(model)





if args.cuda:
    model.cuda(args.gpu)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, epoch):

    lr = args.lr * (0.1 ** (epoch // args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



mi_xt=[]
mi_ty=[]
val_acc=[]
label_test_matrix = np.load('../label_test_matrix/cifar100/label_test_matrix.npy')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(target.shape)
        if args.cuda:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        layer_T, output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #print(layer_T.data[0].cpu().numpy())

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(dataset=test_loader, stage='test'):
    model.eval()
    test_loss = 0
    correct = 0

    for idx, (data, target) in enumerate(dataset):
        if args.cuda:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        data, target = Variable(data, volatile=True), Variable(target)
        layer_T, output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if idx == 0:
            layer_T_array = layer_T
        else:
            layer_T_array = torch.cat((layer_T_array, layer_T), 0)
        test_loss /= len(dataset.dataset)
    logging.info('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        stage, test_loss, correct, len(dataset.dataset),
        100. * correct / len(dataset.dataset)))
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, 3000,
    #     100. * correct / 3000))

    value_xt, value_ty = MI_cal_v3(label_test_matrix, layer_T_array.data.cpu().numpy(), args.mask, 100)
    mi_xt.append(value_xt)
    mi_ty.append(value_ty)
    val_acc.append(correct / len(dataset.dataset))


for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    test(train_loader, stage='train')
    test(test_loader)
    train(epoch)
    torch.save(model.state_dict(), '../model/cifar100/' + args.network + '_model_epoch{}.pkl'.format(epoch))

test(train_loader, stage='train')
test(test_loader)



# save mutual information and accuracy data
path = '../MI_RESULT/CIFAR100/' + args.network
if not os.path.exists(path):
    os.mkdir(path)

xt_value_path = path + '/mi_xt.npy'
ty_value_path = path + '/mi_ty.npy'
val_acc_path = path+ '/val_acc.npy'


np.save(xt_value_path, mi_xt)
np.save(ty_value_path, mi_ty)
np.save(val_acc_path, val_acc)