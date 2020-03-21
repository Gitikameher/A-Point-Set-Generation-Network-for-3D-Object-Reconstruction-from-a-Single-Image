import json
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import imageio
import torch
from PIL import Image
# Path for the dataset:
# For eg. path = "/datasets/cs253-wi20-public/Pix3d/"
# At this address you shall find a folder img with images and a folder model with the pointclouds
# Inside these folders you should have the corresponding images and models for different objects 



class TestDataset():
    def __init__(self, img_path=None, pc_path=None, objects = ['chair','sofa','table']):
        a = json.load(open('/datasets/cs253-wi20-public/pix3d.json'))
        
        img_size = 227
        self.transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                transforms.CenterCrop(img_size),transforms.ToTensor()])
        
        self.Data = []
        for i in range(len(a)):
            if(a[i]['category'] in objects):
                x = a[i]['model']
                x = x.rsplit('/', 1)[0]
                try:
                    for o in os.listdir(pc_path + str(x)):
                        self.Data.append([pc_path + str(x)+'/'+ o, img_path+a[i]['img'].split('/',1)[1]])
                except:
                    #print("File not found")
                    pass
        
    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, idx):
        pointcloud = np.array(np.load(self.Data[idx][0]))
        if self.Data[idx][1][-1]=='f':
            img = Image.open(self.Data[idx][1])
        else:
#             img = imageio.imread(self.Data[idx][1])
#             img = Image.fromarray(img)
            img = Image.open(self.Data[idx][1])
            
            
#         img = Image.open(self.Data[idx][1])
        
        if self.transform is not None:
            img = self.transform(img)
            
        return [img,pointcloud]
