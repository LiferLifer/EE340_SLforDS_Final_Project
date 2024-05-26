import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple[0], original_tuple[1], path)
        return tuple_with_path

# data pre-processing
transform = transforms.Compose([
        #transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

batch_size = 1

trainset = ImageFolderWithPaths(root='./FundusDomainTrain', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)


testset = ImageFolderWithPaths(root='./FundusDomainTest', transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# load pre-trained resnet18
net = resnet18(weights=ResNet18_Weights.DEFAULT, progress=True).cuda()
net.load_state_dict(torch.load('/data/xupx/SLDS-proj2/domain_net.pth'))
net.eval()

net = nn.Sequential(*list(net.children())[:-1]) # get feature extractor

# just like testing step??
def extract_and_save_features(loader, model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.cuda()
            features = model(images)
            features = features.view(features.size(0), -1).cpu().numpy()
            
            for i, path in enumerate(paths):
                base_name = os.path.splitext(os.path.basename(path))[0]
                feature_filename = f"{base_name}_features.npy"
                
                label_dir = int(labels[i].item()) + 1
                
                label_dir_path = os.path.join(output_dir, str(label_dir))
                if not os.path.exists(label_dir_path):
                    os.makedirs(label_dir_path)
                
                feature_path = os.path.join(label_dir_path, feature_filename)
                np.save(feature_path, features[i])

output_dir = './FundusDomainTest_features'
extract_and_save_features(testloader, net, output_dir)