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
from data_loader_pix3d import TestDataset
import argparse, sys

#from emd import EMDLoss


def main():
    # Parse the training argument
    parser=argparse.ArgumentParser()
    parser.add_argument('--training', help='Decide whether to train the model or just run testing on previously saved model.')
    args=parser.parse_args()

    is_training = args.training
    
    if (is_training is None) or (is_training == 'True'):
        is_training = True
    else:
        is_training = False
    print('is_training  mode = ', is_training)

    
    chamferDist = ChamferDistance()
    
    # Decide on GPU or CPU
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')
    
    # Training Configuration
#     image_root = "./../../../datasets/cs253-wi20-public/ShapeNetRendering/"
#     point_cloud_root = "./../../../datasets/cs253-wi20-public/ShapeNet_pointclouds/"
    image_root = "/datasets/cs253-wi20-public/ShapeNetRendering/"
    point_cloud_root = "/datasets/cs253-wi20-public/ShapeNet_pointclouds/"

    num_epochs = 1000
    batch_size = 64
    shuffle = True
    num_workers = 8
    use_2048 = True
    img_size = 227 # I don't know why, but this has to be 227!
    learning_rate = 1e-4
    num_points = 2048
    transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                    transforms.CenterCrop(img_size),transforms.ToTensor()])    
    # Checkpoint
    use_checkpoint = False

    # Split and Get data. Override the saved files if you change the ratios.
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

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
    
    if is_training:
        # Train
        print('Starting training...')
        train_losses, val_loss, best_model = train(model, train_data_loader, val_data_loader, chamferDist,
                                                   model_name="Baseline_DL_Vis_Demo", num_epochs=num_epochs, 
                                                   lr=learning_rate, use_checkpoint = use_checkpoint)
    else:
        best_model = torch.load('best-Baseline_DL_Vis_Demo.pt')
        print('Loaded previously saved model.')
        
    model = best_model.cuda()
    model.eval()
    

    # Compute chamfer distance on Pix3D dataset.
    img_path = "/datasets/cs253-wi20-public/pix3d/"
    pc_path = "/datasets/cs253-wi20-public/pix_pointclouds/"

    objects = ['table', 'sofa']

    test_dataset = TestDataset(img_path,pc_path, objects)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=8)
    
    print('Starting testing on Pix3D dataset...')
    total_test_loss = 0.
    # Get loss on training data.
    with torch.no_grad():
        for i, (image, point_cloud) in enumerate(test_data_loader):

            image, point_cloud = Variable(image), Variable(point_cloud)

    #         print(image.size())
            if (image.size(1) != 3):
                continue
    #         print('reaching.')

            image, point_cloud = image.float().to(device=gpu_or_cpu), point_cloud.float().to(device=gpu_or_cpu)
            pred = model(image)
            dist1, dist2 = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
    #             emd_cost = torch.sum(dist(pred.cuda().double(), points.cuda().double()))
            total_test_loss += loss.item()

    #         print(total_test_loss)
    #         break

            if i%100 == 0:
                print('Batch '+str(i)+' finished.')

    print('Chamfer distance on Pix3D dataset = ', total_test_loss / len(test_data_loader))
    
if __name__ == "__main__":
    main()
    
