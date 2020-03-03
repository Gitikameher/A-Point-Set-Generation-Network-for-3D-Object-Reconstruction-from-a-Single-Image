import os, csv



def split_data(train_ratio, val_ratio, test_ratio):
    image_root = "./../../../datasets/cs253-wi20-public/ShapeNetRendering/"
    point_cloud_root = "./../../../datasets/cs253-wi20-public/ShapeNet_pointclouds/"
    
    train_data = []
    val_data = []
    test_data = []
    data = []
    
    image_types = os.listdir(image_root)
    file_names = []

    for image_type in image_types:
        if image_type.endswith('.tgz'):
                continue

        specific_types = os.listdir(os.path.join(image_root, image_type))

        for specific_type in specific_types:
            if specific_type.endswith('.tgz'):
                continue

            data.append((image_type, specific_type))
    
    random.seed(169)
    random.shuffle(data)
    
    train_data = data[:train_ratio * len(data)]
    val_data = data[train_ratio * len(data) : (train_ratio + val_ratio) * len(data)]
    test_data = data[(train_ratio + val_ratio) * len(data) : ]
    
    # Write to file.
    
    
    
                
                
                
                
                
                
                
                
                
                