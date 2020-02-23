from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import cv2

class PartDataset(data.Dataset):
    def __init__(self, root, npoints=2500, pic2point=True, class_choice=None, train=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {} # a searching dictionary

        self.pic2point = pic2point

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1] # cat[Chair] = 	03001627

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}  # meta[03001627] [pts, seg]
        for item in self.cat:

            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_pic = os.path.join(self.root, self.cat[item], 'expert_verified/seg_img')

            fns = sorted(os.listdir(dir_pic))
            # train 0.9, test 0.1
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'), os.path.join(dir_pic, token + '.png')))
        self.datapath = []  # car,  xxx_v0/02691156/xxxx.pts, xxx_v0/02691156/xxxx.seg, xxx_v0/02691156/xxxx.png
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))

        # match class name with class id number
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.num_seg_classes = 0
        if not self.pic2point:
            for i in range(len(self.datapath)/50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        #seg = np.loadtxt(fn[2]).astype(np.int64)
     
        im = cv2.imread(fn[3])
        if (type(im)!=type(None)):
            im = cv2.resize(im, (227, 227))
            im = cv2.bitwise_not(im)

            mean = im.mean(axis=(0,1,2))/255
            std = im.std(axis=(0,1,2))/255

            transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((mean, mean, mean), (std, std, std))])

            im = transform(im)

        #print(point_set.shape, im.shape)

        # resample
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)  # random choice n id for selection
            point_set = point_set[choice, :]
        #seg = seg[choice]

            point_set = torch.from_numpy(point_set)


        #seg = torch.from_numpy(seg)
            cls = torch.from_numpy(np.array([cls]).astype(np.int64))
            if self.pic2point:
                return im, point_set
            else:
                return point_set, seg, cls

    def __len__(self):
        return len(self.datapath)