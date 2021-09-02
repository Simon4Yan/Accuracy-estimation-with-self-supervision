import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from scipy import stats

from utils import densenet as dn
from utils.rotation import rotate_batch

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 40)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=1.0, type=float,
                    help='compression rate in transition stage (default: 1.0)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='./runs/DenseNet-40-12-ss/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet-40-12-ss', type=str,
                    help='name of experiment')
parser.add_argument('--rotation_type', default='rand')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    # if args.tensorboard:
    #     configure("runs/%s" % (args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # create model
    model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    # test on CIFAR10.1
    from utils.cifar_new import CIFAR_New
    print('Test on CIFAR-10.1')
    teset = CIFAR_New(root='./data/' + 'CIFAR-10.1/', transform=transform_test)
    val_loader = torch.utils.data.DataLoader(teset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # evaluate on semantic classification
    prec_cifarnew = validate(val_loader, model, criterion)
    print('\nSemantic classification accuracy on CIFAR10.1: %.2f' % (prec_cifarnew))

    # evaluate on rotation prediction
    ss_cifarnew = test(val_loader, model)
    print('\nRotation prediction accuracy on CIFAR10.1: %.2f' % (ss_cifarnew))

    # test model on each synthetic set
    cls_acc = []
    ssh_acc = []

    num_set = 500
    for i in range(num_set):
        teset_raw = np.load('/mnt/home/dwj/AutoEval/CIFAR-10_Setup/dataset_GroupC/new_data_' + str(i).zfill(3) + '.npy')
        teset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=False, download=True, transform=transform_test)
        teset.data = teset_raw
        val_loader = torch.utils.data.DataLoader(teset, batch_size=64,
                                                 shuffle=False, num_workers=2)

        # evaluate on semantic classification
        prec1 = validate(val_loader, model, criterion)
        cls_acc.append(prec1)

        # evaluate on rotation prediction
        ss = test(val_loader, model)
        ssh_acc.append(ss)

    np.save('./accuracy_cls_dense_aug.npy', cls_acc)
    np.save('./accuracy_ss_dense_aug.npy', ssh_acc)

    # analyze the statistical correlation
    ssh_acc = np.array(ssh_acc)
    cls_acc = np.array(cls_acc)

    rho, pval = stats.spearmanr(ssh_acc, cls_acc)
    print('\nRank correlation-rho', rho)
    print('Rank correlation-pval', pval)

    rho, pval = stats.pearsonr(ssh_acc, cls_acc)
    print('\nPearsons correlation-rho', rho)
    print('Pearsons correlation-pval', pval)

    from sklearn.linear_model import LinearRegression, HuberRegressor
    from sklearn.metrics import mean_squared_error

    slr = LinearRegression()
    slr.fit(np.array(ssh_acc.reshape(-1, 1)), np.array(cls_acc.reshape(-1, 1)))

    pred = slr.predict(np.array(ss_cifarnew).reshape(-1, 1))
    error = mean_squared_error(pred, np.array(prec_cifarnew).reshape(-1, 1), squared=False)
    print('Linear regression model predicts %4f and its absolute error is %4f' % (pred, error))

    robust_reg = HuberRegressor()
    robust_reg.fit(np.array(ssh_acc.reshape(-1, 1)), np.array(cls_acc.reshape(-1)))

    robust_pred = robust_reg.predict(np.array(ss_cifarnew).reshape(-1, 1))
    robust_error = mean_squared_error(robust_pred, np.array(prec_cifarnew).reshape(-1, 1), squared=False)
    print('Robust Linear regression model predicts %4f and its absolute error is %4f' % (robust_pred, robust_error))


def validate(val_loader, model, criterion):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        # compute output
        with torch.no_grad():
            output, _ = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def test(dataloader, model):
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    correct = []
    losses = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = rotate_batch(inputs, 'expand')
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    model.train()
    print('self-supervised.avg:{:.4f}'.format(correct.mean() * 100))
    return correct.mean() * 100


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
