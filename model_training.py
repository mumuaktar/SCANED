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
import time
from torch.autograd import Variable
from numpy import expand_dims
from torch import optim
from torch.optim import lr_scheduler    
import warnings

import resnet50 from model


def train_val(train_loader,val_loader):
 
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


    def show_plot(iteration,loss):
        plt.plot(iteration,loss)
        plt.show()

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
    pretrain = torch.load('resnet_50_23dataset.pth')

    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    model.to(device)
    
 
    train_loss = []
    loss_history = [] 
    iteration_number= 0
    start = time.time()
    start_epoch = time.time()
    criterion=ContrastiveLoss().to(device)
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import roc_auc_score
    sim=[]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    pat=20
    counter=0
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
               
                scheduler.step()                 
                x0 = expand_dims(x0, axis=1)
                x1 = expand_dims(x1, axis=1)
                x0=torch.from_numpy(x0)
                x1=torch.from_numpy(x1)
                x0, x1, labels = x0.to(device).float(), x1.to(device).float(), labels.to(device).float()

                f1=model(x0)
                f2=model(x1)
                optimizer.zero_grad()

                sg=nn.Sigmoid()
                dis1 = F.pairwise_distance(f1, f2)
                dis1=sg(dis1)
                loss_contrastive = criterion(f1,f2,labels)
                train_loss =+ loss_contrastive.item()
                loss_contrastive.backward()
                optimizer.step()

                all_targets.extend(labels.cpu().detach().numpy().tolist())
                all_predictions.extend(dis1.cpu().detach().numpy().tolist())     


        train_losses.append(train_loss/len(train_loader))
        #print( 'train_losses:',train_losses)    
        np.save('training_losst.npy',train_losses)
        #train_roc_auc = roc_auc_score(all_targets, all_predictions,multi_class='ovr')
        fpr, tpr, th = metrics.roc_curve(all_targets, all_predictions,pos_label=0)
        train_roc_auc = auc(fpr, tpr)
        print('training AUC:',train_roc_auc)

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


                            sg=nn.Sigmoid()
                            dis1 = F.pairwise_distance(f1, f2)
                            dis1=sg(dis1)
                            loss2=criterion(f1,f2,labels1)
                            loss=loss2
                            losses.append(loss.item())
                            all_targets1.extend(labels1.cpu().detach().numpy().tolist())
                            all_predictions1.extend(dis1.cpu().detach().numpy().tolist())  

        model.train() 
        #val_roc_auc = roc_auc_score(all_targets1, all_predictions1,multi_class='ovr')
        fpr, tpr, th = metrics.roc_curve(all_targets1, all_predictions1,pos_label=0)
        val_roc_auc = auc(fpr, tpr)
        print('validation AUC:',val_roc_auc)    
       
        val_loss = sum(losses)/max(1, len(losses))
        val_losses.append(val_loss)
        np.save('val_loss.npy',val_losses)

        counter=counter + 1
        print('counter:',counter)
        #print('%%%%Completed  epoch%%% best_val_loss val_loss:',epoch+1,best_val_loss,val_loss)
        if val_loss < best_val_loss:
                               print("true")
                               counter=0
                               print('counter:',counter)
                               best_val_loss = val_loss
                               print(f"\nSaving best model for epoch: {epoch+1}\n")
                               torch.save(
                                                        {
                                                            "epoch": epoch + 1,
                                                        "model_state_dict": model.state_dict(),
                                                           'optimizer_state_dict': optimizer.state_dict(),

                                #                            
                                                        },
                                                        'val_model.pth'
                                                    )      



        #if counter > pat :
         #  break                









