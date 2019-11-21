#test.py
#!/usr/bin/env python3

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_test_dataloader
from FetchData import *

import matplotlib
#matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import sklearn

def auc_curve(y,prob):
    fpr,tpr,_ = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    print("value: ", roc_auc)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("./results/"+args.name+"aoc.jpg")
    plt.show()

def pr_curve(y,prob):
    precision, recall,_ = precision_recall_curve(y, prob)
    plt.figure(1) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    plt.figure(1)
    plt.plot(precision, recall)
    plt.savefig("./results/"+args.name+"pr.jpg")
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument("-imgs",type=int,default=32,help="imagesize")
    parser.add_argument("-nc",type=int,default=2,help="num of classes")
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument("-split", type=str,help="give split")
    parser.add_argument("-ds", type=int,default=2,help="which dataset")
    parser.add_argument("-name",type=str,help="give name")
    args = parser.parse_args()
    image_size = args.imgs

    if args.ds == 2:
        trainset, testset,_ = getdataset2("..\\..\\datasets\\tumor\\pos", "..\\..\\datasets\\tumor\\neg", args.b, args.split, None, True, False)
    elif args.ds == 3:
        trainset, testset,_ = getdataset3("..\\..\\datasets\\tumor\\pos", "..\\..\\datasets\\tumor\\neg",
        "..\\..\\datasets\\hemorrhage\\pos","..\\..\\datasets\\hemorrhage\\neg",args.b, args.split, None, True, False)
    elif args.ds == 4:
        trainset, testset,_ = getdataset2("..\\..\\datasets\\all\\pos", "..\\..\\datasets\\all\\neg", args.b, args.split, None, True, False, args.poslabel)
    elif args.ds == 5:
        trainset, testset,_ = getdataset2("..\\..\\datasets\\hemorrhage\\pos", "..\\..\\datasets\\hemorrhage\\neg", args.b, args.split, None, True, False, args.poslabel)
    elif args.ds == 6:
        testset = getdataset("..\\..\\datasets\\hemorrhage\\pos","..\\..\\datasets\\hemorrhage\\neg")
    
    testset_size = len(testset.dataset)//5

    net = get_network(args)

    net.load_state_dict(torch.load(args.weights), args.gpu)
    # print(net)
    net.eval()

    
    correct = 0.0

    y_trueA = np.zeros((0),dtype=int)
    y_scoreA = np.zeros((0),dtype=float)
    predA = np.zeros((0),dtype=int)

    for n_iter, (image, label) in enumerate(testset):
        # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        y_true = label.cpu().detach().numpy()
        output = net(image)
        output1 = output[:,1]
        y_scores = output1.cpu().detach().numpy()
        y_scoreA = np.concatenate((y_scoreA, y_scores))
        y_trueA = np.concatenate((y_trueA, y_true))
        # pr_curve(y_true,y_scores)
        # auc_curve(y_true,y_scores)
        # print( "f1_score", sklearn.metrics.f1_score(y_true, y_scores))
        # print(output.shape)
        # print(y_label.shape)
        _, preds = output.max(1)
        pred = preds.cpu().detach().numpy()
        predA = np.concatenate((predA, pred)) 
        correct += preds.eq(label).sum()
    # pr_curve(y_trueA,y_scoreA)
    # print(y_trueA)
    # auc_curve(y_trueA,y_scoreA)
    print( "f1_score micro", sklearn.metrics.f1_score(y_trueA, predA,average='micro'))
    print( "f1_score macro", sklearn.metrics.f1_score(y_trueA, predA,average='macro'))
    print('Test set: Accuracy: {:.4f}'.format(
        correct.float() / testset_size 
    ))

