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
import time
from torch.autograd import Variable
from numpy import expand_dims
from torch import optim
from torch.optim import lr_scheduler    
import warnings

 


def train_eval(train_loader,val_loader,fold):
 
    class ContrastiveLoss(torch.nn.Module):

          def __init__(self, margin=1.0):
                super(ContrastiveLoss, self).__init__()
                self.margin = margin
          def forward(self, output1, output2,label):
                # Find the pairwise distance or eucledian distance of two output feature vectors
                euclidean_distance = F.pairwise_distance(output1, output2)

                #euclidean_distance=output1
                loss_contrastive = torch.mean(((1-label) * torch.pow(euclidean_distance, 2)) +
                ((label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)))

                return loss_contrastive


 
    from model import resnet50
    import torch.nn as nn
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet50(

                        sample_input_D=64,
                        sample_input_W=64,
                        sample_input_H=64,
                        shortcut_type='B',
                        no_cuda=False,
                        num_seg_classes=2)

    net_dict = model.state_dict()
    pretrain = torch.load('resnet_101.pth')

    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    model.to(device)
    
 
    train_loss = []
    criterion=ContrastiveLoss().to(device)
    epochs=300
    best_loss=1000
    best_auc=0
    lab_or=[]
    o_app=[]
    total=0
    total1=0
    train_loss=0
    train_acc=0
    train_acc=0
    best_val_loss=1000
    e=[]
    acc=0
    g1=0
    p1=0
    best_val_auc=0
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    pat=20
    counter=0
    counter1=0
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(epochs):
        all_targets=[]
        all_predictions=[]
        all_targets1=[]
        all_predictions1=[]
        train_loss=0

        for batch_idx, (x0, x1, labels) in enumerate(train_loader):
                counter1=counter1+0
                scheduler.step()                 
                x0 = expand_dims(x0, axis=1)
                x1 = expand_dims(x1, axis=1)
                x0=torch.from_numpy(x0)
                x1=torch.from_numpy(x1)
                x0, x1, labels = x0.to(device).float(), x1.to(device).float(), labels.to(device).float()

                f1=model(x0)
                f2=model(x1)
                optimizer.zero_grad()

                dis1 = F.pairwise_distance(f1, f2)
               # dis1=sg(dis1)
                loss_contrastive = criterion(f1,f2,labels)
                train_loss =+ loss_contrastive.item()
                loss_contrastive.backward()
                optimizer.step()

                all_targets.extend(labels.cpu().detach().numpy().tolist())
                all_predictions.extend(dis1.cpu().detach().numpy().tolist())     

              #  print('after 1 batch:', counter1)
        train_losses.append(train_loss/len(train_loader))
        #print( 'train_losses:',train_losses)    
        np.save('training_losst.npy',train_losses)
        #train_roc_auc = roc_auc_score(all_targets, all_predictions,multi_class='ovr')
        fpr, tpr, th = metrics.roc_curve(all_targets, all_predictions)
        train_roc_auc = auc(fpr, tpr)
        


        losses = []
        correct = 0
        total = 0
        val_acc=0
        pred=[]
        val_dis=[]
        val_lb=[]
        test_loss=0


        with torch.no_grad():
             model.eval()
             for batch_idx1, (a0, a1, labels1) in enumerate(val_loader):
                      
                            a0 = expand_dims(a0, axis=1)
                            a1 = expand_dims(a1, axis=1)
                            a0=torch.from_numpy(a0)
                            a1=torch.from_numpy(a1)
                            a0, a1, labels1 = a0.to(device).float(), a1.to(device).float(), labels1.to(device).float()
                            f1=model(a0)
                            f2=model(a1)
                    
                            dis1 = F.pairwise_distance(f1, f2)
                        #    dis1=sg(dis1)
                            loss2=criterion(f1,f2,labels1)
                            loss=loss2
                            losses.append(loss.item())
                            all_targets1.extend(labels1.cpu().detach().numpy().tolist())
                            all_predictions1.extend(dis1.cpu().detach().numpy().tolist())  

        model.train() 
        #val_roc_auc = roc_auc_score(all_targets1, all_predictions1,multi_class='ovr')
        fpr, tpr, th = metrics.roc_curve(all_targets1, all_predictions1)
        val_roc_auc = auc(fpr, tpr)
        print('validation AUC:',val_roc_auc)    
        
        ##########f1##########
    #     optimal_idx = np.argmax(tpr - fpr)
    #     optimal_threshold = th[optimal_idx]
    #     predicted_classes1 = []
    #     for score in all_predictions1:
    #         if score > optimal_threshold:
    #             predicted_classes1.append(1)
    #         else:
    #             predicted_classes1.append(0)
    # # Calculate F1-score
    #     from sklearn.metrics import f1_score
    #     f1 = f1_score(all_targets1, predicted_classes1, average='weighted')
      #  print('training AUC:',train_roc_auc)
        
        
        
        val_loss = sum(losses)/max(1, len(losses))
        val_losses.append(val_loss)
        np.save('val_loss.npy',val_losses)

        counter=counter + 1
        print('counter:',counter)
        save_path_lb = f"model{fold}.pth"
        #print('%%%%Completed  epoch%%% best_val_loss val_loss:',epoch+1,best_val_loss,val_loss)
        if val_loss < best_val_loss:
                               print("true better val:",best_val_auc)
                               counter=0
                               print('counter:',counter)
                               best_val_loss = val_loss
                               print(f"\nSaving best model for epoch: {epoch+1}\n")
                               torch.save(
                                                        {
                                                            "epoch": epoch + 1,
                                                        "model_state_dict": model.state_dict(),
                                                           'optimizer_state_dict': optimizer.state_dict(),
                                                            'best_AUC': best_val_auc,
                                #                            
                                                        },
                                                        save_path_lb
                                                    )      



        # if counter > pat :
        #    break                









