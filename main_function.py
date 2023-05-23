import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import csv
import glob
import nibabel as nib
from nilearn.image import resample_img
import skimage.transform as skTrans
torch.manual_seed(0)
import scipy.ndimage as ndi
from skimage import morphology
import math


import train_eval from model_training
import test from model_testing


class CustomDataset_train(Dataset):
    def __init__(self,transforms =None,augmentation_transforms=None):
        import glob
        import skimage.transform as skTrans
        import os
        n=0
      
        import skimage.transform as skTrans
        lb1=[]
 
        self.imgs_path1='/root/left_train/'
        self.imgs_path2= '/root/right_train/'
        self.transforms=transforms
        self.augmentation_transforms=augmentation_transforms
        file_list1 = sorted(glob.glob(self.imgs_path1 + "*.nii"))
        file_list2 = sorted(glob.glob(self.imgs_path2 + "*.nii"))
        self.data = []
        for class_path in file_list1:
            self.data.append([class_path])
        self.data1 = []
        for class_path1 in file_list2:
            self.data1.append([class_path1])
  
        self.img_dim = (221,221,160)
    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        import os
        import skimage.transform as skTrans
        from scipy import ndimage
        zz1=[]
        zz2=[]

        labels = pd.read_csv('label_prev.csv')
        img_path = self.data[idx]
        im="".join(str(elem) for elem in img_path)

        im=os.path.join(im)
     
        cr=str(img_path).split("_")[2]
        cr=cr.split(".")
        cr=cr[0]
      
        for i in range (len(labels)):
            
                if cr == (str(labels.subject[i]))
                    lb=labels.ID1[i]
       
        z1=nib.load(str(im))
        
        z1=z1.get_fdata()
        img_path1 = self.data1[idx]
    
        im1="".join(str(elem) for elem in img_path1)

        im1=os.path.join(im1)


        im1=os.path.join(im1)
        z2=nib.load(str(im1))
        
        
        z2=z2.get_fdata()
        z1=np.array(z1)
        z1=expand_dims(z1,axis=0)
        z2=np.array(z2)
        z2=expand_dims(z2,axis=0)

        sample = {'image1': z1,
                  'image2': z2,
                  'label': lb}


        if (sample['label'] == 1) and self.augmentation_transforms:
            sample = self.augmentation_transforms(sample)
        else:
            sample= self.transforms(sample)

        sample['image1']=np.squeeze(sample['image1'])
        sample['image2']=np.squeeze(sample['image2'])
        
        sample['image1']=np.moveaxis(sample['image1'], -1, 0)
        sample['image2']=np.moveaxis(sample['image2'], -1, 0)
            
        return sample['image1'],sample['image2'], sample['label']




class CustomDataset_test(Dataset):
    def __init__(self,val_transforms=None):
        import glob
        import skimage.transform as skTrans
        import os
        n=0
      
        import skimage.transform as skTrans
        lb1=[]
 
        self.imgs_path1='/root/left_test/'
        self.imgs_path2= '/root/right_test/'
        self.val_transforms=val_transforms
        file_list1 = sorted(glob.glob(self.imgs_path1 + "*.nii"))
        file_list2 = sorted(glob.glob(self.imgs_path2 + "*.nii"))
        self.data = []
        for class_path in file_list1:
            self.data.append([class_path])

        self.data1 = []
        for class_path1 in file_list2:
         
            self.data1.append([class_path1])
  

       
        self.img_dim = (221,221,160)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        import os
        import skimage.transform as skTrans
        from scipy import ndimage
        zz1=[]
        zz2=[]

        labels = pd.read_csv('label_prev.csv')
        img_path = self.data[idx]
        im="".join(str(elem) for elem in img_path)
        im=os.path.join(im)
        cr=str(img_path).split("_")[2]
    
        cr=cr.split(".")
        cr=cr[0]
        for i in range (len(labels)):
             
                if cr == (str(labels.subject[i])):
             
                    lb=labels.ID1[i]
       
        z1=nib.load(str(im))
        z1=z1.get_fdata()

        img_path1 = self.data1[idx]
        im1="".join(str(elem) for elem in img_path1)

        im1=os.path.join(im1)
        z2=nib.load(str(im1))
        
        
        z2=z2.get_fdata()
        
        z1=np.array(z1)
        z1=expand_dims(z1,axis=0)
        z2=np.array(z2)
        z2=expand_dims(z2,axis=0)


        sample = {'image1': z1,
                  'image2': z2,
                  'label': lb}

        if self.val_transforms:
           sample=self.val_transforms(sample)
        sample['image1']=np.squeeze(sample['image1'])
        sample['image2']=np.squeeze(sample['image2'])
        
        sample['image1']=np.moveaxis(sample['image1'], -1, 0)
        sample['image2']=np.moveaxis(sample['image2'], -1, 0)

        return sample['image1'],sample['image2'], sample['label']


class CustomDataset_val(Dataset):
    def __init__(self,val_transforms = None):
        import glob
        import skimage.transform as skTrans
        import os
        n=0
      
        import skimage.transform as skTrans
        lb1=[]
 
        self.imgs_path1='/root/left_val/'
        self.imgs_path2= '/root/right_val/'
        self.val_transforms=val_transforms
        file_list1 = sorted(glob.glob(self.imgs_path1 + "*.nii"))
        file_list2 = sorted(glob.glob(self.imgs_path2 + "*.nii"))
   
        self.data = []
        for class_path in file_list1:
            self.data.append([class_path])

        self.data1 = []
        for class_path1 in file_list2:
            self.data1.append([class_path1])
  
       
        self.img_dim = (221,221,160)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        import os
        import skimage.transform as skTrans
        from scipy import ndimage
        zz1=[]
        zz2=[]
        labels = pd.read_csv('label_prev.csv')
        img_path = self.data[idx]
        im="".join(str(elem) for elem in img_path)
        im=os.path.join(im)
        cr=str(img_path).split("_")[2]
        cr=cr.split(".")
        cr=cr[0]
       
        for i in range (len(labels)):
              
                if cr == (str(labels.subject[i])):
                 
                    lb=labels.ID1[i]

        z1=nib.load(str(im))
        z1=z1.get_fdata()
        img_path1 = self.data1[idx]
        im1="".join(str(elem) for elem in img_path1)

        im1=os.path.join(im1)
        z2=nib.load(str(im1))
        z2=z2.get_fdata()


        z1=np.array(z1)
        z1=expand_dims(z1,axis=0)
        z2=np.array(z2)
        z2=expand_dims(z2,axis=0)


        sample = {'image1': z1,
                  'image2': z2,
                  'label': lb}

        if self.val_transforms:
            sample=self.val_transforms(sample)
     
        sample['image1']=np.squeeze(sample['image1'])
        sample['image2']=np.squeeze(sample['image2'])
        sample['image1']=np.moveaxis(sample['image1'], -1, 0)
        sample['image2']=np.moveaxis(sample['image2'], -1, 0)
        return sample['image1'],sample['image2'], sample['label']
       

def main():

    ################monai transform###################
    import monai
    from monai.data import ArrayDataset, GridPatchDataset, create_test_image_3d
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirst,AddChanneld, Transpose, Resized, CropForegroundd,      RandGaussianSmoothd,
                                  ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, SpatialPadd,Rotated,RandRotate90d,Lambda,
        Spacingd, Orientationd, EnsureTyped,RandZoomd, ThresholdIntensityd, RandShiftIntensityd, RandGaussianNoised, BorderPadd,RandAdjustContrastd, NormalizeIntensityd,RandFlipd, ScaleIntensityRanged)

    

    augmentation_transforms = Compose([ScaleIntensityd(keys=['image1', 'image2'], minv=0.0, maxv=1.0),  
                                       RandFlipd(keys=['image1','image2'], prob=0.25, spatial_axis=None),
                                       RandGaussianNoised(keys=['image1','image2'], prob=0.25, mean=0.0, std=0.1),
                                       RandZoomd(keys=['image1','image2'], prob=0.25, min_zoom=0.7, max_zoom=1.3),
                                       RandRotate90d(keys=['image1','image2'], prob=0.25, max_k=3, spatial_axes=(0, 1)),
                                       RandSpatialCropd(keys=['image1', 'image2'], roi_size=(64,64,64), random_size=False),
                                       EnsureTyped(keys=['image1', 'image2'])])

    
    transforms = Compose([
                                ScaleIntensityd(keys=['image1', 'image2'], minv=0.0, maxv=1.0),  
                                RandSpatialCropd(keys=['image1', 'image2'], roi_size=(64,64,64), random_size=False),
                                EnsureTyped(keys=['image1', 'image2'])])  



    val_transforms = Compose([
                                ScaleIntensityd(keys=['image1', 'image2'], minv=0.0, maxv=1.0), 
                                RandSpatialCropd(keys=['image1', 'image2'], roi_size=(64,64,64), random_size=False),
                                EnsureTyped(keys=['image1', 'image2'])])

    
    
    train_dataset=CustomDataset_train(transforms,augmentation_transforms)
    test_dataset=CustomDataset_test(val_transforms)
    val_dataset=CustomDataset_val(val_transforms)


    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
           batch_size=1, shuffle=True)

    
    
    train_eval(train_loader,val_loader)
    test(test_loader)
    
if __name__ == "__main__":
    main()

