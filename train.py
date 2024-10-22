import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import sinkhornknopp as sk
import scipy.sparse
from utils import Bar, AverageMeter, accuracy, mkdir_p
from data.cifar import CIFAR10, CIFAR100
import logging
import torch.utils.data as data
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 noisy training')
# Training options
parser.add_argument('--dataset', type=str, help='cifar10, or cifar100', default='cifar10')
parser.add_argument('--epochs', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--warm_start', default=20, type=int, metavar='N',help='warm up')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay_start', default=80, type=int, help='epoch_decay_start')
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='2,3', type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--out', default='result',help='Directory to output the result')
#Noise options
parser.add_argument('--noise_rate', type=float, default=0.2,help='Percentage of noise')
parser.add_argument('--noise_type', type=str, help='[symmetric, asymmetric]', default='asymmetric')
# Optimization options
parser.add_argument('--nopts', default=100, type=int, help='number of pseudo-opts (default: 100)')
parser.add_argument('--lamb', default=25, type=int, help='for pseudoopt: lambda (default:25) ')
parser.add_argument('--cpu', default=False, action='store_true', help='use CPU variant (slow) (default: off)')
parser.add_argument('--hc', default=1, type=int, help='number of heads (default: 1)')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
best_acc = 0  # best test accuracy
title = 'Noisy training'
LOG_FORMAT = "%(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='Accuracy.txt',level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logging.debug(args)
class Optimizer:
    def __init__(self, m, args, nb_classes, t_loader, test_loader, criterion):
        self.epochs = args.epochs
        self.lr = args.lr
        self.resume = True
        self.hc = args.hc
        self.K = nb_classes
        self.model = m 
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pseudo_loader = t_loader # can also be DataLoader with less aug.
        self.trainloader = t_loader
        self.testloader = test_loader
        self.lamb = args.lamb # the parameter lambda in the SK algorithm
        self.dtype = torch.float64 if not args.cpu else np.float64
        self.outs = [self.K]*args.hc
        self.criterion = criterion
        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [args.lr] * args.epochs
        self.beta1_plan = [mom1] * args.epochs
        for i in range(args.epoch_decay_start, args.epochs):
            self.alpha_plan[i] = float(args.epochs - i) / (args.epochs - args.epoch_decay_start) * args.lr
            self.beta1_plan[i] = mom2

    def optimize_labels(self, niter):
        if not args.cpu and torch.cuda.device_count() > 1:
            sk.gpu_sk(self)
        else:
            self.dtype = np.float64
            sk.cpu_sk(self)
        self.PS = 0

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
    

    def optimize_epoch(self, optimizer, loader, epoch, validation=False):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        XE = torch.nn.CrossEntropyLoss(reduce=False)
        bar = Bar('Training', max=len(loader))
        for iter, (data, label, real_label, selected) in enumerate(loader):
            niter = epoch * len(loader) + iter
            if niter*args.batch_size >= self.optimize_times[-1]:
                ########### optimize labels #########################################
                self.model.headcount = 1
                print('\n Optimizaton starting', flush=True)
                with torch.no_grad():
                    _ = self.optimize_times.pop()
                    self.optimize_labels(niter)
            data = data.to(self.dev)
            mass = data.size(0)
            final = self.model(data)
            #################### train DNN ####################################################
            if epoch <= args.warm_start:
                loss = XE(final,label.cuda()).mean()
            else:
                loss_all = XE(final,label.cuda())
                loss_sorted,indices = torch.sort(loss_all)
                clean_rate = 1 - args.noise_rate
                num_clean = int(clean_rate*mass)
                ind_clean = indices[:num_clean]
                loss_clean = loss_all[ind_clean].mean()
                ind_noisy = indices[num_clean:]  
                loss_noisy = XE(final[ind_noisy], self.L[0, selected[ind_noisy]]).mean()
                loss = loss_clean + loss_noisy
            prec1, prec5 = accuracy(final, label.cuda(), topk=(1, 5))
            top1.update(prec1.item(), mass)
            top5.update(prec5.item(), mass)           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), mass)        
            # plot progress
            bar.suffix  = '({batch}/{size}) | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=iter + 1,
                    size=len(loader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )               
            bar.next()
        bar.finish()
        return losses.avg, top1.avg  

    def optimize(self):
        """Perform full optimization."""
        self.model = self.model.to(self.dev)
        N = len(self.pseudo_loader.dataset)
        # optimization times (spread exponentially), can also just be linear in practice (i.e. every n-th epoch)
        self.optimize_times = [(self.epochs+2)*N] + \
                              ((self.epochs+1.01)*N*(np.linspace(0, 1, args.nopts)**2)[::-1]).tolist()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.L = np.zeros((self.hc, N), dtype=np.int32)
        for nh in range(self.hc): 
            self.L[nh, :] = self.trainloader.dataset.train_noisy_labels
        self.L = torch.LongTensor(self.L).to(self.dev)
        # Perform optmization ###############################################################
        epoch = 0
        acc_list = []
        while epoch < self.epochs:
            self.adjust_learning_rate(optimizer,epoch)
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
            train_loss, train_acc = self.optimize_epoch(optimizer, self.trainloader, epoch, validation=False)
            test_loss, test_acc = test(self.testloader, self.model, self.criterion, use_cuda)
            logging.info('\t%.2f \t %.2f \t %.2f \t %.2f \t'%(train_loss,train_acc,test_loss,test_acc))
            epoch += 1
            if epoch in range(self.epochs-10,self.epochs):
                acc_list.extend([test_acc])
        avg_acc = sum(acc_list)/len(acc_list)
        print("The average test accuracy in last 10 epochs: {}".format(str(avg_acc)))
        return self.model

def test(testloader, model, criterion, use_cuda):
    global best_acc
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing ', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets, selected) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # plot progress
            bar.suffix  = '({batch}/{size}) | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)
def create_model(nb_classes):
    from resnet import resnet18
    model = resnet18(num_classes=nb_classes)
    model = model.cuda()
    return model
def main():
    global best_acc
    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    # load dataset
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

        test_dataset = CIFAR10(root='./data/',
                            download=True,
                            train=False,
                            transform=transform_test,
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

    if args.dataset == 'cifar100':

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        train_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=True,
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

        test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    nb_classes = train_dataset.nb_classes
    # Model
    print("==> creating resnet")
    model = create_model(nb_classes)
    model = torch.nn.DataParallel(model).cuda()
    use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    o = Optimizer(model, args, nb_classes, t_loader=train_loader, test_loader=test_loader, criterion=criterion)
    o.optimize() 

if __name__ == '__main__':
    main()