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
#from emd import EMDLoss


def main():
    chamferDist = ChamferDistance()
    
    # Decide on GPU or CPU
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')
    
    # Training Configuration
    image_root = "./../../../datasets/cs253-wi20-public/ShapeNetRendering/"
    point_cloud_root = "./../../../datasets/cs253-wi20-public/ShapeNet_pointclouds/"

    num_epochs = 20
    batch_size = 32
    shuffle = True
    num_workers = 8
    use_2048 = True
    img_size = 227 # I don't know why, but this has to be 227!
    learning_rate = 1e-3
    num_points = 2500
    transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),transforms.CenterCrop(img_size)])
    
    # Checkpoint
    use_checkpoint = True

    # Split and Get data. Override the saved files if you change the ratios.
    train_ratio = 0.02
    val_ratio = 0.01
    test_ratio = 0.01

    split_data(train_ratio, val_ratio, test_ratio, overrideFiles = False)

    path_train = 'train_data.txt'
    path_val = 'val_data.txt'
    path_test = 'test_data.txt'

    train_data = read_from_file(path_train)
    val_data = read_from_file(path_val)
    test_data = read_from_file(path_test)


    # Data loader
    train_data_loader = get_loader(image_root, point_cloud_root, train_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)

    val_data_loader = get_loader(image_root, point_cloud_root, val_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)
    test_data_loader = get_loader(image_root, point_cloud_root, test_data, use_2048, 
                             transform, batch_size, shuffle, num_workers)

    
    print('Len of train loader = ', len(train_data_loader))

    # create model
    print("model building...")
    model = pic2points(num_points=num_points)
    model.to(device=gpu_or_cpu)
    
    
    # Train
    train_losses, val_loss, best_model = train(model, train_data_loader, val_data_loader, chamferDist, model_name="Baseline", num_epochs=num_epochs, lr=learning_rate, use_checkpoint = use_checkpoint)
    
    
if __name__ == "__main__":
    main()
    
