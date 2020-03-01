#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import urllib.request

# url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip'
# urllib.request.urlretrieve(url, 'data.zip')
# from zipfile import ZipFile
# with ZipFile('data.zip', 'r') as zipObj:
#    # Extract all the contents of zip file in different directory
#    zipObj.extractall('data')
    
# url = 'https://github.com/chrdiller/pyTorchChamferDistance/archive/master.zip'
# urllib.request.urlretrieve(url, 'chamfer.zip')
# with ZipFile('chamfer.zip', 'r') as zipObj:
#    # Extract all the contents of zip file in different directory
#    zipObj.extractall('')
    
# url = 'https://github.com/meder411/PyTorch-EMDLoss/archive/master.zip'
# urllib.request.urlretrieve(url, 'emd.zip')
# with ZipFile('emd.zip', 'r') as zipObj:
#    # Extract all the contents of zip file in different directory
#    zipObj.extractall('')
    
# !python setup.py install


# In[1]:


from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
import torch.nn.functional as F
import torch.cuda as cuda
from pic2points_model import pic2points
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torch
from chamfer_distance import ChamferDistance
from data_loader import XDataset
#from emd import EMDLoss

#dist =  EMDLoss()

chamferDist = ChamferDistance()

def main():
    manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

#     dataset = PartDataset(root = 'data/PartAnnotation/', pic2point = True, npoints = 2500)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
#     print("number of training data:"+ str(len(dataset)))
#     test_dataset = PartDataset(root = 'data/PartAnnotation/', pic2point = True, train = False, npoints = 2500)
#     testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16,shuffle=True, num_workers=8)
#     print("number of testing data:"+ str(len(test_dataset)))

    
    # Decide on GPU or CPU
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')
    
    # Training Configuration
    image_root = "./../../../datasets/cs253-wi20-public/ShapeNetRendering/"
    point_cloud_root = "./../../../datasets/cs253-wi20-public/ShapeNet_pointclouds/"

    batch_size = 4
    shuffle = True
    num_workers = 8
    use_2048 = True
    img_size = 256
    learning_rate = 1e-3
    num_points = 2500
    transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),transforms.CenterCrop(img_size)])

    # Data loader
    data_loader = get_loader(image_root, point_cloud_root, use_2048, transform, batch_size, shuffle, num_workers)

    # create model
    print("model building...")
    model = pic2points(num_points=num_points)
    model.to(device=gpu_or_cpu)

    # load pre-existing weights
  #  if opt.model != '':
   #     model.load_state_dict(torch.load(opt.model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     num_batch = len(dataset) / batch_size

    print('training mode ------------------')
    for epoch in range(num_epochs):
        print("epoch:"+str(epoch))
        for i, (image, point_cloud) in enumerate(data_loader):
            image, point_cloud = data
            image, point_cloud = Variable(image), Variable(point_cloud)
            image, point_cloud = image.to(device=gpu_or_cpu), point_cloud.to(device=gpu_or_cpu)
            pred = model(image)
            dist1, dist2 = chamferDist(pred, points)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
#             emd_cost = torch.sum(dist(pred.cuda().double(), points.cuda().double()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 is 0:
                print("training loss is:" + str(loss.item()))

#         loss_test = 0
#         for i, data in enumerate(dataloader, 0):
#             im_test, points_test = data
#             im_test, points_test = Variable(im_test), Variable(points_test)
#             im_test, points_test = im_test.cuda(), points_test.cuda()
#             pred_test = model(im_test)
#             dist1, dist2 = chamferDist(pred_test, points_test)
#             loss_test = (torch.mean(dist1)) + (torch.mean(dist2))
# #             emd_test = torch.sum(dist(pred_test.cuda().double(), points_test.cuda().double()))
#         print("Testing loss is:" + str(loss_test.item()))

if __name__ == '__main__':
    num_cuda = cuda.device_count()
    print("number of GPUs have been detected:" + str(num_cuda))
    #with torch.cuda.device(1):
    main()


# In[ ]:




