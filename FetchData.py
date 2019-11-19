import torch
from torch.utils import data
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import pickle
import random

SPLIT_SIZE = 5

def picload(filename):
    with open(filename,'rb') as f:
        a = pickle.load(f)
    return a

def picdump(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

image_size = 32

def gettransc():
    transc =  transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    return transc

def gettrans():
    trans =  transforms.Compose([
        # transforms.Resize(image_size),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    return trans

class dataset2(data.Dataset):
    def __init__(self, pathpos, pathneg, corp, transforms=None):
        super(dataset2).__init__()
        if transforms == None:
            if corp:
                self.transforms = gettransc()
            else:
                self.transforms = gettrans()
        img1 = os.listdir(pathpos)
        self.pos = [os.path.join(pathpos, i) for i in img1]
        # print(self.pos)
        
        img2 = os.listdir(pathneg)
        self.neg = [os.path.join(pathneg, i) for i in img2]
        # print(self.neg)
        self.corp = not corp

    def __getitem__(self, index):
        if index >= len(self.neg):
            index -= len(self.neg)
            imgpath = self.pos[index]
            if self.corp:
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = cv2.resize(img_gray, (image_size, image_size))
                img_out = Image.fromarray(img_out)
                if self.transforms:
                    img_out = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img_out))
                # print(img_out.shape)
                return img, 1
            else:
                # print("in else")
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = Image.fromarray(img_gray)
                if self.transforms:
                    img = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img))
                # print(img.shape)
                return img, 1
        else:
            imgpath = self.neg[index]
            if self.corp:
                img = cv2.imread(imgpath)
                 
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = cv2.resize(img_gray, (image_size, image_size))
                img_out = Image.fromarray(img_out)
                if self.transforms:
                    img_out = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img_out))
                return img, 0
            else:
                # print("in else")
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = Image.fromarray(img_gray)
                if self.transforms:
                    img = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img))
                # print(img.shape)
                return img, 0

    def __len__(self):
        return len(self.pos)+len(self.neg)


class dataset3(data.Dataset):
    def __init__(self, pathpos1, pathneg1, pathpos2, pathneg2, corp, transforms):
        super(dataset3).__init__()
        img1 = os.listdir(pathpos1)
        self.pos1 = [os.path.join(pathpos1, i) for i in img1]
        # print(self.pos1)
        
        img2 = os.listdir(pathneg1)
        self.neg = [os.path.join(pathneg1, i) for i in img2]
        # print(self.neg)
        if pathpos2 is not None:
            img1 = os.listdir(pathpos2)
            self.pos2 = [os.path.join(pathpos2, i) for i in img1]
        else:
            self.pos2 = []
        # print(self.pos1)
        
        if pathneg2 is not None:
            img2 = os.listdir(pathneg2)
            self.neg += [os.path.join(pathneg2, i) for i in img2]
        
        print(transforms, corp)
        self.transforms = None
        if transforms == None:
            if corp:
                self.transforms = gettransc()
            else:
                print("here!")
                self.transforms = trans
        print(self.transforms)
        self.corp = not corp

    def __getitem__(self, index):
        if index < len(self.neg):
            imgpath = self.neg[index]
            if self.corp:
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = cv2.resize(img_gray, (image_size, image_size))
                img_out = Image.fromarray(img_out)
                if self.transforms:
                    img_out = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img_out))
                return img, 0
            else:
                # print("in else")
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = Image.fromarray(img_gray)
                if self.transforms:
                    img = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img))
                # print(img.shape)
                return img, 0
        index -= len(self.neg)
        if index < len(self.pos1):
            imgpath = self.pos1[index]
            if self.corp:
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = cv2.resize(img_gray, (image_size, image_size))
                img_out = Image.fromarray(img_out)
                if self.transforms:
                    img_out = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img_out))
                return img, 1
            else:
                # print("in else")
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = Image.fromarray(img_gray)
                if self.transforms:
                    img = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img))
                # print(img.shape)
                return img, 1
        index -= len(self.pos1)
        if index <= len(self.pos2):
            imgpath = self.pos2[index]
            if self.corp:
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = cv2.resize(img_gray, (image_size, image_size))
                img_out = Image.fromarray(img_out)
                if self.transforms:
                    img_out = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img_out))
                return img, 2
            else:
                # print("in else")
                img = cv2.imread(imgpath)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_out = Image.fromarray(img_gray)
                if self.transforms:
                    img = self.transforms(img_out)
                img = torch.from_numpy(np.asarray(img))
                # print(img.shape)
                return img, 2
        
    def __len__(self):
        return len(self.pos1)+len(self.neg)+len(self.pos2) 


def getdataset3(pathpos1, pathneg1, pathpos2, pathneg2, batchSize, split=None, transforms = None, corp = False, iftest = False):
    dataset = dataset3(pathpos1, pathneg1, pathpos2, pathneg2, corp, transforms)
    if split:
        indices = picload(split)
    else:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        picdump("temp.pkl", indices)
    split = len(dataset) // 5
    train_indices = indices[split:]
    test_indices = indices[:split]
    train_sampler = data.SubsetRandomSampler(train_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, 
                                           sampler=train_sampler)
    if iftest:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=split, 
                                           sampler=test_sampler)
    else:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                           sampler=test_sampler)
    return train_loader, test_loader, indices

def getdataset2(pathpos, pathneg, batchSize, split=None, transforms = None, corp = False, iftest=True):
    dataset = dataset2(pathpos, pathneg, corp, transforms)
    if split:
        indices = picload(split)
    else:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        picdump("temp.pkl", indices)
    split = len(dataset) // 5
    train_indices = indices[split:]
    test_indices = indices[:split]
    train_sampler = data.SubsetRandomSampler(train_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, 
                                           sampler=train_sampler)
    if iftest:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                           sampler=test_sampler)
    else:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=split, 
                                           sampler=test_sampler)
    return train_loader, test_loader, indices

if __name__ == '__main__':
    trainset, testset,_ = getdataset2("..\\..\\datasets\\tumor\\pos", "..\\..\\datasets\\tumor\\neg", 4, None, None, True)
    # train, test,_ = getdataset3(".\\tumor\\pos", ".\\tumor\\neg", ".\\hemorrhage\\pos", ".\\hemorrhage\\neg",8, None, None, True)
    for i, (img, label) in enumerate(trainset):
        print(img.shape, label.shape)

        

