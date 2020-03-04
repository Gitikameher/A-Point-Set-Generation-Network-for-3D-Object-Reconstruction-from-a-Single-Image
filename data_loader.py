import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image
import numpy as np

class XDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, image_root, point_cloud_root, id_pairs, transform=None, use_2048 = True):
        
        """
        Args:
            path: image directory.
            
        """
        self.image_root = image_root
        self.point_cloud_root = point_cloud_root
        self.transform = transform
        self.use_2048 = use_2048
        self.normalize = transforms.Compose([transforms.ToTensor()])
        self.id_pairs_set = set(id_pairs)
        
        
        # Init ids
        self.ids = []
        
        # Store ids
        
        image_types = os.listdir(image_root)
        file_names = []
        
        for image_type in image_types:
            if image_type.endswith('.tgz'):
                    continue
            
            specific_types = os.listdir(os.path.join(image_root, image_type))
            
            for specific_type in specific_types:
                if specific_type.endswith('.tgz'):
                    continue
                
                if (image_type, specific_type) not in self.id_pairs_set:
                    continue # Consider only those pairs which are supposed to be in the train/test/val data.
                
                path1 = os.path.join(os.path.join(os.path.join(image_root, image_type), specific_type), 'rendering')
                temp = os.listdir(path1)
                
                for file_name in temp:
                    
                    if file_name.endswith('.png'):
                        path = os.path.join(path1, file_name)
                        
                        self.ids.append((image_type, specific_type, path))
            

    

    def __getitem__(self, index):
        """Returns one data pair (actual image and point cloud)."""
        image_root = self.image_root
        point_cloud_root = self.point_cloud_root
        
        image_type, specific_type, image_path = self.ids[index]
        
        # Load image and point cloud and return
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        image = np.asarray(image)
#         print('Original image size = ', image.shape)
        
#         print('After transpose image size = ', image.shape)
        image = self.normalize(image) # Change from (H, W, C) to (C, H, W)
        
#         print('After normalize image size = ', image.size())
            
            
        
        if self.use_2048:
            point_cloud_path = os.path.join(os.path.join(os.path.join(point_cloud_root, image_type), 
                                                     specific_type), 'pointcloud_2048.npy')
        else:
            point_cloud_path = os.path.join(os.path.join(os.path.join(point_cloud_root, image_type), 
                                                     specific_type), 'pointcloud_1024.npy')
            
        point_cloud = np.load(point_cloud_path)
        
        return image, point_cloud

    def __len__(self):
        return len(self.ids)

def get_loader(image_root, point_cloud_root, id_pairs, use_2048, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom X dataset."""
    
    xdataset = XDataset(image_root, point_cloud_root, id_pairs, transform = transform, use_2048 = use_2048)
    
    data_loader = torch.utils.data.DataLoader(dataset=xdataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader

