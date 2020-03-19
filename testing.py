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
from data_loader import XDataset, get_loader
from split_data import read_from_file, write_to_file, split_data
import time
from train_ajit import train
from visualize import Visualize



def main():
    
    chamferDist = ChamferDistance()
    
    # Decide on GPU or CPU
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')

    image_root = "/datasets/cs253-wi20-public/ShapeNetRendering/"
    point_cloud_root = "/datasets/cs253-wi20-public/ShapeNet_pointclouds/"

    num_epochs = 20
    batch_size = 64
    shuffle = True
    num_workers = 8
    use_2048 = True
    img_size = 227 # I don't know why, but this has to be 227!
    learning_rate = 5e-4
    num_points = 2500
    transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                    transforms.CenterCrop(img_size),transforms.ToTensor()])
    
    
    
    path_test = 'test_data.txt'
    test_data = read_from_file(path_test)
    
    test_data_loader = get_loader(image_root, point_cloud_root, test_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)
    
    
    model = torch.load('best-Baseline_FixedDL.pt').to(device=gpu_or_cpu)
    model.eval()
    

    
    
    for i, (image, point_cloud) in enumerate(test_data_loader):
        image, point_cloud = Variable(image, requires_grad = False), Variable(point_cloud, requires_grad = False)

        image, point_cloud = image.float().to(device=gpu_or_cpu), point_cloud.float().to(device=gpu_or_cpu)
        pred = model(image)
        dist1, dist2 = chamferDist(pred, point_cloud)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        
        print('pred size = ', pred.size())
        pred = pred.to('cpu')
        
        out = []
        
        for p in pred:
            out.append(p.detach().numpy())
            
        print('type(out) = ', type(out))
        
        
        
        

        # Visualize the prediction
        print('asdf')
        Visualize(out).ShowRandom()
        break
        
        
if __name__ == "__main__":
    main()
        


    
    
    

