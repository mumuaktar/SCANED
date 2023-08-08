# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:28:36 2023

@author: m_ktar
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from numpy import expand_dims
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import monai
from monai.data import ArrayDataset, GridPatchDataset, create_test_image_3d
from monai.transforms import (Compose,ThresholdIntensity, CropForegroundd,LoadImaged, EnsureChannelFirst,AddChanneld, Transpose, Resized, CropForegroundd,      RandGaussianSmoothd,
                          ScaleIntensityd, ToTensord,CenterSpatialCropd, RandSpatialCropd, RandSpatialCropSamples, Rand3DElasticd, RandAffined, SpatialPadd,Rotated,RandRotate90d,Lambda,
Spacingd, Orientationd, RandCropByPosNegLabeld,EnsureTyped,RandZoomd, ThresholdIntensityd, RandShiftIntensityd, RandGaussianNoised, BorderPadd,RandAdjustContrastd, NormalizeIntensityd,RandFlipd, ScaleIntensityRanged)


   # stride = tuple(int(p * overlap) for p in patch_size)
   # Define patch size and number of patches
patch_size = (64,64,64)  # Size of each patch
num_patches = 10  # Number of patches to extract per image

# Create the Random Crop Samples transform


# Define the ResNet3D model
import torch
import torch.nn as nn

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet-50 Model
class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 64)
        
        self.rel = nn.ReLU(inplace=True)
        self.fc1= nn.Linear(64,num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels * 4, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
     
        x=self.rel(x)
        x=self.fc1(x)

        return x

# Create an instance of the ResNet-50 model with 2 output classes
model = ResNet(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
# Define a custom dataset class for reading NIfTI images
class BrainDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = sorted(os.listdir(folder_path))
       

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        # print('dddd:',file_path)
        labels = pd.read_csv('label_prev.csv')
        img_path = file_path
        
        cr=str(img_path).split("_")[1]
        # print(cr)
        import re
        number = re.search(r'brain(\d+)', cr)
        # print(number)
        if number is not None:
            # Convert the extracted number to an integer
            number = int(number.group(1))
            # print(number)  # Output: 10
    
            # print(number)
        cr=number
        for i in range (len(labels)):
             
                if str(cr) == (str(labels.subject[i])):
             
                    lb=labels.ID1[i]
        # print(lb)
        from PIL import Image
        import torchvision.transforms as transforms
       
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define the transformations to apply to the image
        transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Adjust color jitter
            transforms.RandomRotation(10),
         
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        transform1 = transforms.Compose([
            transforms.Resize(224),  
            
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        if lb==1:
            image=transform(image)
        else:
            image=transform1(image)
        # image = torch.unsqueeze(image,dim=0)   
        # image = nib.load(file_path).get_fdata(dtype=np.float32)
        # image=expand_dims(image,axis=0)
     
        # sample = {'image': image,
                 
        #           'label': lb}
        
        # if (lb==1) and self.augmentation_transforms:
        #     sample = self.augmentation_transforms(sample)
        # else:
        #     sample= self.transforms(sample)
        # Preprocess the image as needed (e.g., normalization, resizing)
        # Return the preprocessed image and any labels if available
        # return sample['image'],sample['label']
        return image,lb
class BrainDataset_val(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = sorted(os.listdir(folder_path))
     
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        # print('dddd:',file_path)
        labels = pd.read_csv('label_prev.csv')
        img_path = file_path
        # print(img_path)
        cr=str(img_path).split("_")[2]
        # print(cr)
        import re
        number = re.search(r'brain(\d+)', cr)
        # print(number)
        if number is not None:
            # Convert the extracted number to an integer
            number = int(number.group(1))
            # print(number)  # Output: 10
    
            # print(number)
        cr=number
        for i in range (len(labels)):
             
                if str(cr) == (str(labels.subject[i])):
             
                    lb=labels.ID1[i]
        # print(lb)
        from PIL import Image
        import torchvision.transforms as transforms
       
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define the transformations to apply to the image
        transform = transforms.Compose([
            transforms.Resize(224),  # Resize the image to 256x256 pixels
         
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
      
        image=transform(image)

        return image,lb

import torch.nn as nn

data_folder='/home/mumu/project4/original_data/slices'
data_folder1='/home/mumu/project4/original_data/val_slices'
# Set the number of cross-validation folds
num_folds = 5

# Set hyperparameters and other settings
batch_size = 64
num_epochs = 100
learning_rate = 0.001
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

gamma=.1
# Create the ResNet3D model
# model = ResNet3D()
sensitivity_list = []
specificity_list = []
f1_score_list = []
accuracy_list = []
high=1000
# Create a StratifiedKFold cross-validator
skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=gamma)
# Create a KFold cross-validator
kfold = KFold(n_splits=num_folds, shuffle=True)
labels = pd.read_csv('label_prev.csv')
lb=labels.ID1
# Iterate over the folds
weights = [.51, 2.81]
label_counts = {0: 0, 1: 0}

from torch.utils.data import DataLoader, WeightedRandomSampler

# for fold in range(5):
#     print(f'Fold {fold + 1}')
    # Create the data loaders for training and validation
train_dataset1 = BrainDataset(data_folder)
# val_dataset = BrainDataset_val(data_folder,val_transforms)
   
labels = [label for _, label in train_dataset1]
    # print(labels)
samples_weight = [weights[label] for label in labels]
    
sampler = WeightedRandomSampler(samples_weight, len(train_dataset1), replacement=True)
# sampler1 = WeightedRandomSampler(samples_weight, len(val_dataset), replacement=True)


train_loader1 = DataLoader(train_dataset1, batch_size=batch_size)

import torch
from torch.utils.data import DataLoader, random_split


validation_split = 0.2  # 20% of the data will be used for validation

# Calculate the size of the validation set based on the validation_split
dataset_size = len(train_dataset1)
validation_size = int(validation_split * dataset_size)
train_size = dataset_size - validation_size

folds = []

# Create five folds for cross-validation
fold_size = len(train_dataset1) // num_folds
for i in range(num_folds):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(train_dataset1)
    train_fold = torch.utils.data.Subset(train_dataset1, list(range(start_idx, end_idx)))
    val_fold = torch.utils.data.Subset(train_dataset1, list(range(end_idx, end_idx + fold_size)))
    folds.append((train_fold, val_fold))

# Training loop with five-fold cross-validation
for fold_idx, (train_fold, val_fold) in enumerate(folds):
    # Set the model to training mode
    model.train()
    losses = []

    # Separate data loaders for training and validation sets for this fold
    labels = [label for _, label in train_fold]
   #     # print(labels)
    samples_weight = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(samples_weight, len(train_fold), replacement=True)
    labels = [label for _, label in val_fold]
   #     # print(labels)
    samples_weight = [weights[label] for label in labels]
    sampler1 = WeightedRandomSampler(samples_weight, len(val_fold), replacement=True)
    train_loader = DataLoader(train_fold, batch_size=batch_size, sampler=sampler)
    validation_loader = DataLoader(val_fold, batch_size=batch_size, sampler=sampler1)

# Training loop
    for epoch in range(num_epochs):
    # Set the model to training mode
        model.train()
        losses=[]
        # Iterate over the training data
        for images, labels in train_loader:
            # Forward pass
            # print(images.shape)
            # images=torch.unsqueeze(images,dim=1)
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            # print(outputs)
            loss = criterion(outputs, labels)
           
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for images, labels in validation_loader:
                # images=torch.unsqueeze(images,dim=1)
                images=images.to(device)
                labels=labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                losses.append(val_loss.item())
    
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        #         save_path_lb = f"com{fold}.pth"
        t_loss = sum(losses)/max(1, len(losses))
        if t_loss<high:
                high=t_loss
                print("true better val:",high)
                counter=0
                print(f"\nSaving best model for epoch: {epoch+1}\n")
                torch.save(
                                          {
                                              "epoch": epoch + 1,
                                          "model_state_dict": model.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                    
                  #                            
                                          },
                                          'comp.pth'
                                      )      
            
              
data_folder1='/home/mumu/project4/original_data/val_slices'
model_path = 'comp.pth'
total=0
correct=0
checkpoint=torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
test_dataset = BrainDataset_val(data_folder1)
batch_size=1
pr=[]
tr=[]
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

with torch.no_grad():
    for images, labels in test_loader:
    # images=torch.unsqueeze(images,dim=1)
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        # print(outputs) 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        tr.append(labels)
        pr.append(predicted)
    # print(predicted,labels,total)
# Calculate evaluation metrics for the current fold
tr=torch.stack(tr)
pr=torch.stack(pr)
true_labels=tr.cpu().numpy()
predictions=pr.cpu().numpy()
accuracy = accuracy_score(true_labels, predictions)
sensitivity = recall_score(true_labels, predictions, average='macro')
specificity = recall_score(true_labels, predictions, labels=[0], average='macro')
f1 = f1_score(true_labels, predictions, average='macro')
print(accuracy,sensitivity,specificity,f1)

target_names = ['0', '1']
from sklearn.metrics import classification_report
print(classification_report(true_labels, predictions, target_names=target_names))


from sklearn.metrics import confusion_matrix


# Compute the confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Get the number of correctly classified instances for each class
correct_class_0 = cm[0, 0]
correct_class_1 = cm[1, 1]

print("Correctly classified instances for class 0:", correct_class_0)
print("Correctly classified instances for class 1:", correct_class_1)

