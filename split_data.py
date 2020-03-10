import os, csv, random

def read_from_file(path):
    data = []
    
    with open(path, 'r') as file:
        reader = csv.reader(file)
        
        for line in reader:
#             print('line = ', line)
            img_id, specific_id = line[0], line[1]
            data.append((img_id, specific_id))
    return data

def write_to_file(path, data):
    with open(path, 'w') as file:
        writer = csv.writer(file, delimiter = ',')
        for d in data:
            writer.writerow(d)
            

def split_data(train_ratio, val_ratio, test_ratio, overrideFiles = False):
    # Check if files already exist.
    path_train = 'train_data.txt'
    path_val = 'val_data.txt'
    path_test = 'test_data.txt'
    if os.path.isfile(path_train) and os.path.isfile(path_val) and os.path.isfile(path_test) and overrideFiles == False:
        return
    
    # Otherwise, generate files.
#     image_root = "./../../../datasets/cs253-wi20-public/ShapeNetRendering/"
#     point_cloud_root = "./../../../datasets/cs253-wi20-public/ShapeNet_pointclouds/"
    image_root = "/datasets/cs253-wi20-public/ShapeNetRendering/"
    point_cloud_root = "/datasets/cs253-wi20-public/ShapeNet_pointclouds/"
    
    train_data = []
    val_data = []
    test_data = []
    data = []
    
    print('Current dir = ', os.getcwd())
    print(os.listdir('.'))
    
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
    
    train_data = data[:int(train_ratio * len(data))]
    val_data = data[int(train_ratio * len(data)) : int((train_ratio + val_ratio) * len(data))]
    test_data = data[int((train_ratio + val_ratio) * len(data)) : int((train_ratio + val_ratio + test_ratio) * len(data))]
    
    # Write to file.
    write_to_file(path_train, train_data)
    write_to_file(path_val, val_data)
    write_to_file(path_test, test_data)
    
    
    
    
                
                
                
                
                
                
                
                
                
                