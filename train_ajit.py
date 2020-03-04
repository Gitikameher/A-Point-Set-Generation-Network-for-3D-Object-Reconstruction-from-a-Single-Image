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
import json, os
import time
#from emd import EMDLoss


def train(model: nn.Module, train_loader, val_loader, chamferDist, model_name="Baseline", num_epochs=100, lr=5e-3, use_checkpoint = False):
    # Decide on GPU or CPU
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    experiment_data = read_json_file(model_name)
    
    if use_checkpoint:
        num_of_epochs_already_performed = len(experiment_data)
    else:
        num_of_epochs_already_performed = 0
        
    if num_of_epochs_already_performed is not 0:
        model = torch.load(model_name + ".pt")
        training_losses = [e['training_loss'] for e in experiment_data]
        validation_losses = [e['val_loss'] for e in experiment_data]
    else:
        training_losses = []
        validation_losses = []
        
    print("Continuing from Epoch ", num_of_epochs_already_performed)
    
    model.train()
    
    for epoch in range(num_of_epochs_already_performed, num_epochs):
        training_loss = 0.
        ts = time.time()
        
        for i, (image, point_cloud) in enumerate(train_loader):
#         print('image type = ', type(image))
#         print('image size = ', image.size())
#         print('point_cloud size = ', point_cloud.size())
            image, point_cloud = Variable(image), Variable(point_cloud)

            image, point_cloud = image.float().to(device=gpu_or_cpu), point_cloud.float().to(device=gpu_or_cpu)
            pred = model(image)
            dist1, dist2 = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
    #             emd_cost = torch.sum(dist(pred.cuda().double(), points.cuda().double()))
    
            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("training loss, " + "Batch "+ str(i) + ": " + str(loss.item()))
                
        training_loss = training_loss / len(train_loader)
        val_loss = val(model, chamferDist, val_loader)
        
        training_losses.append(training_loss)
        validation_losses.append(val_loss)
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        print("Training Loss {}, Validation Loss {}".format(training_loss, val_loss))
        
        if val_loss <= min(validation_losses):
            best_epoch = epoch
            torch.save(model, "best-" + model_name + ".pt")
            
        torch.save(model, model_name + ".pt")
        
        epoch_data = {'training_loss': training_loss, 'val_loss': val_loss}
        
        
        experiment_data.append(epoch_data)
        write_json_to_file(model_name, experiment_data)
        
    return training_losses, validation_losses, torch.load("best-" + model_name + ".pt")
    
    
def val(model, chamferDist, val_loader):
    model.eval()
    total_val_loss = 0
    
    # Decide on GPU or CPU
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device('cuda')
    else:
        gpu_or_cpu = torch.device('cpu')
    
    # Get loss on validation data.
    with torch.no_grad():
        for i, (image, point_cloud) in enumerate(val_loader):

            image, point_cloud = Variable(image), Variable(point_cloud)

            image, point_cloud = image.float().to(device=gpu_or_cpu), point_cloud.float().to(device=gpu_or_cpu)
            pred = model(image)
            dist1, dist2 = chamferDist(pred, point_cloud)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
    #             emd_cost = torch.sum(dist(pred.cuda().double(), points.cuda().double()))
            total_val_loss += loss.item()

    model.train()
    
    return total_val_loss/len(val_loader)


def write_json_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)

        
## File Utility methods
def read_json_file(path):
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        return []