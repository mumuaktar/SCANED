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



def test(test_loader):

    model = resnet50(

                    sample_input_D=64,
                    sample_input_W=64,
                    sample_input_H=64,
                    shortcut_type='B',
                    no_cuda=False,
                    num_seg_classes=2)



    checkpoint=torch.load('val_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    e=checkpoint['epoch']
    #a=checkpoint['auc']
    print('epoch:',e)
    #print('auc:',a)
    model.to(device)
    print('testing new data:................')


    import time
    from torch.autograd import Variable
    from numpy import expand_dims
    accuracy=0
    correct=0
    test_accuracy=[]
    count=0
    dis=[]
    lb=[]
    original=[]
    correct=0
    correct1=1
    dis=[]
    c=0
    c1=0
    p=0
    s=0
    g=0
    original=[]
    total=0
    acc=0
    all_targets1=[]
    all_predictions1=[]
    model.eval()
    for  i,(x0, x1, labels) in enumerate(test_loader):

            count=count + 1
            print('testing subject:',count)

            with torch.no_grad():
               
                    x0 = expand_dims(x0, axis=1)
                    x1 = expand_dims(x1, axis=1)
                    x0=torch.from_numpy(x0)
                    x1=torch.from_numpy(x1)
                    x0, x1, labels = x0.to(device).float(), x1.to(device).float(), labels.to(device).float()

                    f1=model(x0)
                    f2=model(x1)
                    sg=nn.Sigmoid()
                    dis1 = F.pairwise_distance(f1, f2)
                    #print('dis:',dis1)
                    dis1=sg(dis1)
                    print('dis:',dis1)
                    dis.append(dis1)
                    lb.append(labels)
                    all_targets1.extend(labels.cpu().detach().numpy().tolist())
                    all_predictions1.extend(dis1.cpu().detach().numpy().tolist())  

    #test_roc_auc = roc_auc_score(all_targets1, all_predictions1,pos_label=0)
    #print('test AUC:',test_roc_auc) 
#     from sklearn.preprocessing import OneHotEncoder
#     onehot_encoder = OneHotEncoder(sparse=False)
#     all_targets1=np.array(all_targets1)
#     all_targets1 = all_targets1.reshape(-1, 1)
#     onehot_encoded = onehot_encoder.fit_transform(all_targets1)
#     print(onehot_encoded)
    #test_roc_auc = roc_auc_score(onehot_encoded, all_predictions1,multi_class='ovr')
    fpr, tpr, th = metrics.roc_curve(all_targets1, all_predictions1,pos_label=0)
    test_roc_auc = auc(fpr, tpr)
    print('test AUC:',test_roc_auc)               
   


    print('%%%%%%%%%%%testing with roc%%%%%%%%%%%')
    from sklearn.metrics import roc_curve, auc
    from sklearn import metrics
    dis1=torch.stack(dis).to(device)
    lb1=torch.stack(lb)
    lb1 = lb1.cpu().numpy()
    score=dis1.cpu().numpy()
    score=score.squeeze()
    print(score.shape)
    fpr, tpr, th = metrics.roc_curve(lb1,score,pos_label=0)
    roc_auc = auc(fpr, tpr)
    #roc_auc=roc_auc_score(lb1, score, multi_class='ovr')
    print(roc_auc)

    p=0
    g=0
    correct=0
    t=torch.from_numpy(th)
    t=t.to(device)

    new=[]
    new_l=[]
    count=0
    count1=0

    for j in range(len(t)):
        print('for th:',t[j],j)
        for i in range(len(dis)):
        #     print(dis[i])
            if dis1[i]<t[j]:

                        pred=1
                        count=count + 1

            else:
                        pred=0
                        count1= count1 + 1

            if lb[i] == pred:
        #                     print(('true'))
                            correct=correct+1
                            if lb[i]==0 and pred==0:
                                g=g+1
                            if lb[i]==1 and pred==1:
                                p=p+1


        accuracy=(correct/len(test_loader))*100
        print('overall:',accuracy)
        print('correctly classified-good:',g,'poor:',p)
        correct=0
        g=0
        p=0
        
        
        
        #########plotting roc###############
        ######for 3 classes################
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np
        
        import numpy as np
        y_true_fold1=np.load('target1.npy')
        y_pred_fold1=np.load('prediction1.npy')
        y_true_fold2=np.load('target2.npy')
        y_pred_fold2=np.load('prediction2.npy')
        y_true_fold3=np.load('target3.npy')
        y_pred_fold3=np.load('prediction3.npy')
        y_true_fold4=np.load('target4.npy')
        y_pred_fold4=np.load('prediction4.npy')
        y_true_fold5=np.load('target5.npy')
        y_pred_fold5=np.load('prediction5.npy')

        # Assume y_true and y_pred are numpy arrays containing true labels and predictions
        # for each fold of the cross-validation
        y_true_folds = [y_true_fold1, y_true_fold2, y_true_fold3, y_true_fold4, y_true_fold5]
        y_pred_folds = [y_pred_fold1, y_pred_fold2, y_pred_fold3, y_pred_fold4, y_pred_fold5]

        # Concatenate the true labels and predictions for all folds
        true_labels = np.concatenate((y_true_fold1, y_true_fold2, y_true_fold3, y_true_fold4, y_true_fold5))
        preds = np.concatenate((y_pred_fold1, y_pred_fold2, y_pred_fold3, y_pred_fold4, y_pred_fold5))

        # Compute the fpr, tpr, and auc for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot the ROC curves for each class
        plt.figure(figsize=(8, 6))
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        labels = ['Class 0', 'Class 1', 'Class 2', 'One-vs-rest (micro)', 'One-vs-rest (macro)']

        for i in range(3):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                     label=labels[i] + ' (AUC = %0.2f)' % roc_auc[i])

        fpr_macro = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        tpr_macro = np.zeros_like(fpr_macro)
        for i in range(3):
            tpr_macro += np.interp(fpr_macro, fpr[i], tpr[i])
        tpr_macro /= 3
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        plt.plot(fpr_macro, tpr_macro, color=colors[4], lw=2,
                 label=labels[4] + ' (AUC = %0.2f)' % roc_auc_macro)

        # Add the legend, axis labels, and title
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')

        plt.show()
        
        
        ##########for 2 classes###############
        fpr, tpr, thresholds = roc_curve(true_labels, preds,pos_label=0)

        # calculate AUC
        roc_auc = auc(fpr, tpr)

        # plot ROC curve
        plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--') # random predictions curve
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()