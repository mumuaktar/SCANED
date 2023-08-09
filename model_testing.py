
def test(val_loader,test_loader,fold,val_f1,val_target,val_predicted,final_f1,final_target,final_predicted,to_save,to_save1,lab):
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

    from model import resnet50
    import torch.nn as nn
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(

                    sample_input_D=64,
                    sample_input_W=64,
                    sample_input_H=64,
                    shortcut_type='B',
                    no_cuda=False,
                    num_seg_classes=2)



    model_path = f'model{fold}.pth'
    print('check_model_name:',model_path)
     #checkpoint=torch.load('val_model11.pth')
    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    e=checkpoint['epoch']
 #   a=checkpoint['best_AUC']
    print('epoch:',e)
 #   print('auc:',a)
    model.to(device)
    print('testing new data:................')
    

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
    llb=[]
    all_targets1=[]
    all_predictions1=[]
    # to_save=[]
    # to_save1=[]
    # lab=[]
    c=0
    model.eval()
    image_shape=(140,70,170)
    with torch.no_grad():
        for  i,(x0, x1, labels) in enumerate(val_loader):

                        count=count + 1
            # print('testing subject:',count)

                        patch_predictions = []
                        patched_data = []
                        batch_features=[]
                        # x0 = expand_dims(x0, axis=1)
                        # x1 = expand_dims(x1, axis=1)
                        # x0=torch.from_numpy(x0)
                        # x1=torch.from_numpy(x1)
                        x0, x1, labels = x0.to(device).float(), x1.to(device).float(), labels.to(device).float()
              
                        _,z_dim, x_dim, y_dim = x0.shape
                        #_,xx, yy, zz = torch.where(x0>0)
                        x_max = x_dim
                        x_min = 0
                        y_max = y_dim
                        y_min = 0
                        z_max=  z_dim
                        z_min= 0
                      
                        patch_size_x = 64
                        patch_size_y = 64
                        patch_size_z = 64
                        
                   
                        stride_x = (patch_size_x // 2)  # Half of the patch size for overlapping
                        stride_y = (patch_size_y // 2)   # Half of the patch size for overlapping
                        stride_z = (patch_size_z // 2)   # Half of the patch size for overlapping
                        
                        num_patches_x = ((x_max - x_min) - patch_size_x) // stride_x + 1
                        num_patches_y = ((y_max - y_min) - patch_size_y) // stride_y + 1
                        num_patches_z = ((z_max - z_min) - patch_size_z) // stride_z + 1
                        # print(num_patches_x,num_patches_y,num_patches_z)
                       
                        i = 0
                        
                        for z in range(num_patches_z):
                            for x in range(num_patches_x):
                                for y in range(num_patches_y):
                                    patches1 = []
                                    patches2 = []
                                    patch_start_x = x * stride_x + x_min
                                    patch_end_x = patch_start_x + patch_size_x
                                    patch_start_y = y * stride_y + y_min
                                    patch_end_y = patch_start_y + patch_size_y
                                    patch_start_z = z * stride_z + z_min
                                    patch_end_z = patch_start_z + patch_size_z
                                    
                                    # Ignore patches that go beyond the image dimensions
                                    if (
                                        patch_end_x > x_max or
                                        patch_end_y > y_max or
                                        patch_end_z > z_max
                                    ):
                                        continue
                                    # if patch_start_x == 0 or patch_end_x == x_dim or patch_start_y == 0 or patch_end_y == y_dim or patch_start_z == 0 or patch_end_z == z_dim:
                                    #       continue
                                    for batch in range(x0.shape[0]):
                                         patch1 = x0[batch, patch_start_z:patch_end_z,patch_start_x:patch_end_x, patch_start_y:patch_end_y]
                                         patch2 = x1[batch,patch_start_z:patch_end_z,patch_start_x:patch_end_x, patch_start_y:patch_end_y]
                                         if torch.sum(patch1) == 0 or torch.sum(patch2) == 0:
                                                continue
                                         patches1.append(patch1)
                                         patches2.append(patch2)
                                    i=i+1
                                    patches1=torch.stack(patches1)
                                    patches2=torch.stack(patches2)
                                     # print('shape:',patches1.shape,patches2.shape)
                                     # plot3d(patches1[0])
                                     # plot3d(patches2[0])
                                    patch1 = torch.unsqueeze(patches1, dim=1)
                                    patch2 = torch.unsqueeze(patches2, dim=1)     # print('size:',patch1.shape)
                                    f1=model(patch1)
                                    f2=model(patch2)
                                    sg=nn.Sigmoid()
                                    dis1 = F.pairwise_distance(f1, f2)
                   
                                    patch_predictions.append(dis1)
                                
    
    
                         # print(patch_predictions)
                        patch_predictions=torch.stack(patch_predictions)
                        patch_predictions = torch.transpose(patch_predictions, 0, 1)
                        # print(patch_predictions)
                        c=c+1
                
                        topk_values, topk_indices = torch.topk(patch_predictions, k=3,dim=1)
                        
                        f_d=torch.median(topk_values,dim=1).values.cpu()
                       # f_d = torch.mean(topk_values,dim=1).cpu()

         # Extend the list with the CPU tensor values

                         
                       # print(f_d)
                        all_targets1.extend(labels.cpu().detach().numpy().tolist())
                        all_predictions1.extend(f_d.detach().numpy().tolist())
                       

    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc
    #val_roc_auc = roc_auc_score(all_targets1, all_predictions1,multi_class='ovr')
    fpr, tpr, th = metrics.roc_curve(all_targets1, all_predictions1)
    val_roc_auc = auc(fpr, tpr)
    print('val AUC:',val_roc_auc)    
    
    ##########f1##########      
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = th[optimal_idx]
    predicted_classes1 = []
    for score in all_predictions1:
        if score > optimal_threshold:
            predicted_classes1.append(1)
        else:
            predicted_classes1.append(0)
# Calculate F1-score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_targets1, predicted_classes1, average='weighted')
    print('val_f1:',f1)
    val_f1.append(f1)
    val_target.append(all_targets1)
    val_predicted.append(predicted_classes1)
    print('five-fold:',val_f1,sum(val_f1)/len(val_f1))
    # np.save('val_t.npy',val_target)
    # np.save('val_p.npy',val_predicted)
    # np.save('val_f1.npy',val_f1)
    

    # Calculate Youden's J statistic
    j_values = tpr - fpr
  #  print('chk_diff:',tpr,fpr,j_values)
    optimal_index = np.argmax(j_values)
  #  print('index:',optimal_index)
    
    optimal_threshold = th[optimal_index]
    optimal_threshold_class_0 = th[np.argmax(tpr - fpr)]
    optimal_threshold_class_1 = th[np.argmax(tpr + (1 - fpr))]
    print('threshold:',optimal_threshold_class_0,optimal_threshold_class_1)

    p=0
    g=0
    correct=0
    t=torch.from_numpy(th)
    #t=t.to(device)
    dis1=all_predictions1
    lb=all_targets1
    new=[]
    new_l=[]
    count=0
    count1=0
    max_acc=0
    for j in range(len(t)):
      #  print('for th:',t[j])
        for i in range(len(dis)):
        #     print(dis[i])
            if dis1[i]>=t[j]:

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


        accuracy=(correct/len(val_loader))*100
        if accuracy>max_acc:
            max_acc=accuracy
            best_th=t[j]
      #  print('overall:',accuracy)
    #    print('correctly classified-good:',g,'poor:',p)
        
        correct=0
        g=0
        p=0
  #  print('best_acc with best th:',max_acc,best_th)   
  #      
    ############################3d display###########################
    import torch
    from torch.utils.data import DataLoader
    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')

            self.X = X
            rows, cols, self.slices = X.shape
            self.ind = self.slices//2

            self.im = ax.imshow(self.X[:, :, self.ind],cmap= 'gray')
            self.update()

        def onscroll(self, event):
            print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.X[:, :, self.ind])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()
            
    def plot3d(image):
        original=image
        original = np.rot90(original, k=-1)
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, original)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
    
    
    
    
    def start_points(size, split_size, overlap=0):
        points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                if split_size == size:
                    break
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points
    
    
    ######################test overlapping patches##################
    
    import torch
    from torch.utils.data import DataLoader
    from monai.data import ArrayDataset
    from monai.transforms import Compose, SpatialCrop, ToTensor
    from monai.inferers import SimpleInferer
    # from monai.data.utils import compute_patch_indices
    #from monai.metrics import compute_accuracy
    
    

    test_dataloader = test_loader
    image_shape=(140,70,170)
    

    all_predictions2=[]
    all_predictions3=[]
    to_save_t=[]
    all_targets2=[]   
    model.eval()
    voted_classes = []
    with torch.no_grad():
        aggregated_predictions = []
        lb=[]
        for  i,(x0, x1, labels) in enumerate(test_dataloader):
               patch_predictions = []
               patched_data = []
               batch_features=[]
              
               x0, x1, labels = x0.to(device).float(), x1.to(device).float(), labels.to(device).float()
               #print(x0.shape)
               patch_size = [64,64,64]  # Size of the patches
               stride_size =[32,32,32]
             
               _,z_dim, x_dim, y_dim = x0.shape
               #_,xx, yy, zz = torch.where(x0>0)
               x_max = x_dim
               x_min = 0
               y_max = y_dim
               y_min = 0
               z_max=  z_dim
               z_min= 0
             
               patch_size_x = 64
               patch_size_y = 64
               patch_size_z  = 64
               
               stride_x = (patch_size_x // 2)  # Half of the patch size for overlapping
               stride_y = (patch_size_y // 2) # Half of the patch size for overlapping
               stride_z = (patch_size_z // 2)   # Half of the patch size for overlapping
               
               num_patches_x = ((x_max - x_min) - patch_size_x) // stride_x + 1
               num_patches_y = ((y_max - y_min) - patch_size_y) // stride_y + 1
               num_patches_z = ((z_max - z_min) - patch_size_z) // stride_z + 1
               # print(num_patches_x,num_patches_y,num_patches_z)
              
               i = 0
               
               for z in range(num_patches_z):
                   for x in range(num_patches_x):
                       for y in range(num_patches_y):
                           patches1 = []
                           patches2 = []
                           patch_start_x = x * stride_x + x_min
                           patch_end_x = patch_start_x + patch_size_x
                           patch_start_y = y * stride_y + y_min
                           patch_end_y = patch_start_y + patch_size_y
                           patch_start_z = z * stride_z + z_min
                           patch_end_z = patch_start_z + patch_size_z
                           
                           # Ignore patches that go beyond the image dimensions
                           if (
                               patch_end_x > x_max or
                               patch_end_y > y_max or
                               patch_end_z > z_max
                           ):
                               continue
                           # if patch_start_x == 0 or patch_end_x == x_dim or patch_start_y == 0 or patch_end_y == y_dim or patch_start_z == 0 or patch_end_z == z_dim:
                           #         continue
                           for batch in range(x0.shape[0]):
                                patch1 = x0[batch, patch_start_z:patch_end_z,patch_start_x:patch_end_x, patch_start_y:patch_end_y]
                                patch2 = x1[batch,patch_start_z:patch_end_z,patch_start_x:patch_end_x, patch_start_y:patch_end_y]
                                if torch.sum(patch1) == 0 or torch.sum(patch2) == 0:
                                      continue
                                patches1.append(patch1)
                                patches2.append(patch2)
                           i=i+1
                           patches1=torch.stack(patches1)
                           patches2=torch.stack(patches2)
                            # print('shape:',patches1.shape,patches2.shape)
                            # plot3d(patches1[0])
                            # plot3d(patches2[0])
                           patch1 = torch.unsqueeze(patches1, dim=1)
                           patch2 = torch.unsqueeze(patches2, dim=1)     # print('size:',patch1.shape)
                           f1=model(patch1)
                           f2=model(patch2)
                           sg=nn.Sigmoid()
                           dis1 = F.pairwise_distance(f1, f2)
                           patch_predictions.append(dis1)
                            
   
   
               # print(patch_predictions)
               patch_predictions=torch.stack(patch_predictions)
               patch_predictions = torch.transpose(patch_predictions, 0, 1)
               c=c+1
                #print(c,patch_predictions.shape)
               # print('before:',patch_predictions)
               # patch_predictions1=patch_predictions.view(-1)
                #print(c,patch_predictions.shape) 
               topk_values, topk_indices = torch.topk(patch_predictions, k=3,dim=1)
               
               f_d=torch.median(topk_values,dim=1).values.cpu()
               # f_d = torch.mean(topk_values,dim=1).cpu()
             #  print(topk_values)
#                thresholded_values = (topk_values >= optimal_threshold).int()
#                print(thresholded_values)
# # Perform majority voting
               
#                for i in range(thresholded_values.size(0)):
#                  counts = torch.bincount(thresholded_values[i])
#                  print(counts)
#                  majority_class = counts.argmax()
#                  print(majority_class)
#                  voted_classes.append(majority_class)
           
# Extend the list with the CPU tensor values

                
              # print(f_d)
               all_targets2.extend(labels.cpu().detach().numpy().tolist())
               all_predictions2.extend(f_d.detach().numpy().tolist())
               # all_predictions3.extend(voted.detach().numpy().tolist())
              

    # print(voted_classes)
    fpr, tpr, th = metrics.roc_curve(all_targets2, all_predictions2)
    test_roc_auc = auc(fpr, tpr)
    print('test AUC:',test_roc_auc) 
   
    predicted_classes2 = []
    for score in all_predictions2:
       if score > optimal_threshold:
          predicted_classes2.append(1)
       else:
        predicted_classes2.append(0)
# Calculate F1-score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_targets2, predicted_classes2, average='weighted')
    # f2=f1_score(all_targets2, voted_classes, average='weighted')
    print('f1 from test:',f1)
    
    # Classify samples based on the optimal thresholds
    class_0_predictions = [1 if p >= optimal_threshold_class_0 else 0 for p in all_predictions2]
    class_1_predictions = [1 if p >= optimal_threshold_class_1 else 0 for p in all_predictions2]
    
    # Calculate sensitivity (TPR) for each class
    sensitivity_class_0 = np.sum(np.logical_and(class_0_predictions == 0, all_targets2 == 0)) / np.sum(all_targets2 == 0)
    sensitivity_class_1 = np.sum(np.logical_and(class_1_predictions == 1, all_targets2 == 1)) / np.sum(all_targets2 == 1)
    
    print("Sensitivity (TPR) for Class 0:", sensitivity_class_0)
    print("Sensitivity (TPR) for Class 1:", sensitivity_class_1)

    final_f1.append(f1)
    final_target.append(all_targets2)
    final_predicted.append(predicted_classes2)
    print('five-fold:',final_f1,sum(final_f1)/len(final_f1),all_targets2,predicted_classes2)
    save_target = f"target{fold}.npy"  # Choose the desired save path and filename for each fold
    save_predicted = f"predicted{fold}.npy"
    # save_path_lb = f"fold{fold}_lb.npy"
    np.save(save_target,all_targets2)
    np.save(save_predicted,predicted_classes2)
  
    
  
