import glob, os
from PIL import Image
from matplotlib import pyplot as plt

filenames = []
paths = []

for filename in os.listdir('img/bed'):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        filenames.append(filename)
        paths.append(os.path.join('img/bed', filename))
        
mask_paths = []

idx = 0
for filename in filenames:
    for file in os.listdir('mask/bed'):
        image_id = filenames[idx].split('.')[0]
        
        if file.startswith(image_id):
            mask_paths.append(os.path.join('mask/bed', file))
            idx += 1
            break
            
            
img_size = 227

transform = transforms.Compose([transforms.Resize(img_size,interpolation=2),
                                    transforms.CenterCrop(img_size),transforms.ToTensor()])



out = []



for idx in range(len(paths)):
    image_path = paths[idx]
    mask_path = mask_paths[idx]
    image = Image.open(image_path).convert('RGB')
    
    mask = Image.open(mask_path)
    
#     plt.imshow(image)
    
    
    image = transform(image)
    mask = transform(mask)
    
    # Multiply
    print(image.size(), mask.size())
    
    image = image* mask
#     break
    
    image = image.float().to(device=gpu_or_cpu)
    
    
    # Reshape to include batch size
    image = image.unsqueeze(0)
    
    
    pred = model(image)
    
    pred = pred.to('cpu')
    
    for p in pred:
        out.append(p.detach().numpy())
    
    if len(out) == 10:
        break
    
Visualize(out).ShowRandom()

mask_paths[:10]