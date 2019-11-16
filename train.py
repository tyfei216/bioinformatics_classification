# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

from FetchData import *

step = 0.199999999999

def train(epoch, ifprint):

    net.train()
    cnt = 0
    stone = 0
    for _, (images, labels) in enumerate(trainset):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)
        if len(images.shape) < 4:
            images = images.unsqueeze(0)
            labels - labels.unsqueeze(0)

        cnt+=labels.shape[0]

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch) * trainset_size + cnt + 1

        last_layer = list(net.children())[-1]
        #for name, para in last_layer.named_parameters():
        #    if 'weight' in name:
        #        writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #    if 'bias' in name:
        #        writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        if ifprint:
            if float(cnt)/trainset_size>stone:
                stone+=step
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=cnt,
                    total_samples=trainset_size
                ))

        #update training loss for each iteration
        #writer.add_scalar('Train/loss', loss.item(), n_iter)

    #for name, param in net.named_parameters():
    #    layer, attr = os.path.splitext(name)
    #    attr = attr[1:]
    #    writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for _, (images, labels) in enumerate(testset):
        images = Variable(images)
        # images = images.unsqueeze(0)
        # print(images.shape)
        labels = Variable(torch.tensor(labels))
        # labels = labels.unsqueeze(0)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / testset_size,
        correct.float() / testset_size
    ))
    print()

    #add informations to tensorboard
    # writer.add_scalar('Test/Average loss', test_loss / testset_size, epoch)
    # writer.add_scalar('Test/Accuracy', correct.float() / testset_size, epoch)

    return correct.float() / testset_size

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    # parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument("-cp", type=str,default="checkpoints",help="save models")
    parser.add_argument("-label",type=str,default="",help="label of checkpoints")
    parser.add_argument("-nc",type=int,default=2,help="num of classes")
    parser.add_argument("-imgs",type=int,default=32,help="imagesize")
    parser.add_argument("-split",type=str,default=None,help="the split dataset")
    args = parser.parse_args()
    
    image_size = args.imgs
    net = get_network(args, use_gpu=args.gpu)
    trainset, testset,_ = getdataset2("..\\..\\datasets\\tumor\\pos", "..\\..\\datasets\\tumor\\neg", args.b, args.split, None, True)    
    
    trainset_size = len(trainset.dataset) - len(trainset.dataset) // SPLIT_SIZE
    testset_size = len(trainset.dataset) // SPLIT_SIZE
    
    # trainset, testset,_ = getdataset2("Y:\\生物信息学方法\\datasets\\tumor\\pos", "Y:\\生物信息学方法\\datasets\\tumor\\neg", 4, None, None, True)
    
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(trainset)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    # settings.CHECKPOINT_PATH = ""
    settings.TIME_NOW += args.label
    checkpoint_path = os.path.join(args.cp, args.net, settings.TIME_NOW)

    #use tensorboard
    # if not os.path.exists(settings.LOG_DIR):
    #    os.mkdir(settings.LOG_DIR)
    # writer = SummaryWriter(log_dir=os.path.join(
    #        settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 1, image_size, image_size).cuda()
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        train(epoch, True)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > 100 and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    # writer.close()
