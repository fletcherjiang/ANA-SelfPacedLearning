import os
import cv2
import sys
import copy
from tqdm import tqdm
import torch
import time
from config import *  # Import everything from config.py
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.utils.data
import scipy.misc as misc
import torch.optim as optim
import sklearn.metrics as sm
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset
import torch.nn.functional as F

#自定义函数
from data_weighted_single import get_traindata as get_traindata_single, get_testdata as get_testdata_single
from data_weighted_multiple_filename import get_traindata as get_traindata_multiple, get_testdata as get_testdata_multiple
from data_weighted_filename import get_traindata, get_testdata

from utils import set_random_seed, Logger, inference, get_small, train, test, save, get_images, test_small, get_train_dataset, get_train_dataloader,save_results_to_json

import concurrent.futures

def cv_imread(file_path):
    # 避免中文路径imread无法使用，重写了函数
    # Open source version - reads standard image files without decryption
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

set_random_seed(42 , True)
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='HQnet-for-resnet-pre-train')
parser.add_argument('--lr',              default=LEARNING_RATE,     type=float, help='learning rate')
parser.add_argument('--epoch',           default=EPOCH,             type=int,   help='number of epochs')
parser.add_argument('--trainBatchSize',  default=BATCH_SIZE,        type=int,   help='training batch size')
parser.add_argument('--testBatchSize',   default=BATCH_SIZE,        type=int,   help='testing batch size')
parser.add_argument('--weightDecay',     default=WEIGHT_DECAY,      type=float, help='weight decay')
parser.add_argument('--pathModelParams', default=None, type=str, help='path of model parameters')
parser.add_argument('--saveModel',       action='store_true', help='save model parameters')
parser.add_argument('--loadModel',       action='store_true', help='load model parameters')
parser.add_argument('--weight_lr', default=LEARNING_RATE, type=float, help='learning rate of weights')
parser.add_argument('--initWeight', default='iw-ones', type=str, help='initilization method of weights')
parser.add_argument('--updateLR', default='ulr-ones', type=str, help='updating method of learning rates')
parser.add_argument('--trainingLabel', default='real', type=str, help='type of training labels')
parser.add_argument('--sampling', action='store_true', help='weighted sampling')
parser.add_argument('--granularity', default='label', type=str, help='granularity of weights')
args = parser.parse_args()
# No encryption in open source version
print('Running open source version without encryption')

if args.pathModelParams is None:
    PATH_MODEL_PARAMS  = '2_25_' +'iw_al_val_' + str(args.lr) + 'lr_' + str(args.weight_lr) + 'weight_' + args.initWeight + '_' + args.updateLR + '_trainingLabel_' + str(args.trainingLabel) + '_sampling_' + str(args.sampling) + '_granularity_' + args.granularity
else:
    PATH_MODEL_PARAMS = args.pathModelParams
    
if args.saveModel:
    os.makedirs(PATH_MODEL_PARAMS, exist_ok = True)

sys.stdout = Logger(f"{PATH_MODEL_PARAMS}/output.txt")
# ==================================================================
# 准备数据
# ==================================================================
normalize = transforms.Normalize(mean=[0.005, 0.190, 0.006],
                                  std=[0.008, 0.102, 0.008]) #计算得出

val_transforms = transforms.Compose([
                    transforms.Resize((N, N)), 
                    transforms.ToTensor(),
                    normalize
                  ]) 
val_dataset = get_testdata(root=DIR_TEST_IMAGES,
                            annFile=PATH_TEST_ANNFILE, 
                            transform=val_transforms,
                          split = 'val')

# val_indices = list(range(VAL_LIMIT)) if VAL_LIMIT is not None else list(range(len(val_dataset)))
# val_dataset = Subset(val_dataset, val_indices)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                            batch_size=args.testBatchSize,  
                            shuffle=False,
                            num_workers = 4)

print('Data Preparation : Finished')

resnet50 = torchvision.models.resnet50(pretrained=True) #使用预训练
resnet50.fc = nn.Linear(2048, NUM_CATEGORIES)

if GPU_IN_USE:
    resnet50.to(device)

print('Model Preparation : Finished')

# ==================================================================
# loss function
# ==================================================================
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
    def forward(self, input, target, weight):
        bce = F.binary_cross_entropy(input, target, reduction = 'none')        
        if args.granularity == 'sample':
            bce *= F.relu(weight.nan_to_num())[:, None]
        else:
            bce[target == 1] *= F.relu(weight[target == 1].nan_to_num())
        
        return bce.sum()
    
loss_function = WeightedBCELoss()


# ==================================================================
# Main Loop
# ==================================================================
t_start = time.time()
train_dataset = get_train_dataset()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                batch_size=args.trainBatchSize, 
                                num_workers = 4)

print('Initialize weights')
labels = torch.zeros((len(train_dataset), NUM_CATEGORIES), dtype = torch.float64)
label_filenames = np.array([''] * len(train_dataset)).astype('U256')
weight_idx = []
label_types = []


for batch_num, (_, label, idx, filenames) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing Train Data", unit="batch"):
    labels[idx] = label
    label_filenames[idx] = filenames

    for i, l1 in enumerate(label):
        break_flag = False
        for j, l2 in enumerate(label_types):
            if (l1 - l2).sum() == 0:
                break_flag = True
                weight_idx[j].append(idx[i])
                break
        if not break_flag:
            label_types.append(l1)
            weight_idx.append([idx[i]])

print('label_types', torch.stack(label_types), torch.stack(label_types).sum(dim = 1).max())

targets = labels.clone()

if args.updateLR == 'ulr-ones':
    learning_rates = torch.ones(len(label_types))
else:
    learning_rates = (torch.log(torch.stack(label_types).sum(dim=1)) / torch.log(torch.stack(label_types).sum(dim = 1).max())).cuda()
print('learning_rates', learning_rates)
weight_idx_flatten = torch.tensor([ww for w in weight_idx for ww in w])
group_idx = torch.argsort(weight_idx_flatten)


if args.initWeight == 'iw-ones':
    #Optimized version #1
    if args.granularity == 'sample':
        weights = [torch.ones(len(train_dataset))[ sum(list(map(len, weight_idx))[:i]) : sum(list(map(len, weight_idx))[:i]) + len(weight_idx[i])] for i in range(len(weight_idx))]
    else:
        weights = [labels[weight_idx_flatten][ sum(list(map(len, weight_idx))[:i]) : sum(list(map(len, weight_idx))[:i]) + len(weight_idx[i])] for i in range(len(weight_idx))]
elif args.initWeight == 'iw-data':
    #Optimized version #2
    if args.granularity == 'sample':
        iw = F.softmax(torch.tensor(list(map(len, weight_idx))) / len(train_dataset))
        weights = [torch.ones(len(weight_idx[i])) * iw[i] for i in range(len(weight_idx))]
    else:
        iw = labels.sum(dim = 0)
        iw /= iw.sum()
    
        iw = labels * iw
        iw[labels == 0] = float('-inf')
    
        iw = F.softmax(iw) #.float()
        
        weights = [(labels * iw)[weight_idx_flatten][ sum(list(map(len, weight_idx))[:i]) : sum(list(map(len, weight_idx))[:i]) + len(weight_idx[i])] for i in range(len(weight_idx))]



elif args.initWeight == 'iw-sample':
    #Optimized version #3
    iw = torch.zeros((len(train_dataset), NUM_CATEGORIES), dtype = torch.float64)
    for filename in np.unique(label_filenames):
        iw_sub = labels[label_filenames == filename].sum(dim = 0)
        iw_sub /= iw_sub.sum()
        iw[label_filenames == filename] = iw_sub
    #iw = iw.float()
    weights = [iw[weight_idx_flatten][ sum(list(map(len, weight_idx))[:i]) : sum(list(map(len, weight_idx))[:i]) + len(weight_idx[i])] for i in range(len(weight_idx))]
else:
    print('No valid initilization method for weights')
    exit()



print('Initialization Done', time.time() - t_start)

weight_lr = args.weight_lr
weight_params = []
for p, l in zip(weights, learning_rates):
    p.requires_grad = True
    def print_grad(grad):
        if torch.isnan(grad).sum() > 0:
            print(torch.isnan(grad).sum())
    p.register_hook(print_grad)
    weight_params.append({'params': p, 'lr': l * weight_lr})
optimizer = optim.Adam([{'params': resnet50.parameters()}] + weight_params, lr=args.lr, weight_decay=args.weightDecay, betas=MOMENTUM)

num_early_stop = 5
count_early_stop = 0
count_early_stop_single = 0
count_early_stop_multiple = 0

for current_epoch in range(1, args.epoch + 1):
    print('\n===> epoch: %d/%d' % (current_epoch, args.epoch))

    with torch.no_grad():
        train_loader = get_train_dataloader(resnet50, torch.cat(weights)[group_idx], labels, train_dataset, args)
    
    train_cp, train_cr, train_cf1, train_op, train_or, train_of1 = train(optimizer, train_loader, weights, args, resnet50, group_idx, loss_function,targets,device)
    with torch.no_grad():
        
        #'''
        if args.updateLR == 'ulr-adaptive':
            #Update weights of learning rates
            weight_idx = []
            label_types = []
            for i, l1 in enumerate((targets > 0).float()):
                break_flag = False
                for j, l2 in enumerate(label_types):
                    if (l1 - l2).sum() == 0:
                        break_flag = True
                        weight_idx[j].append(i)
                        break
                if not break_flag:                
                    label_types.append(l1)
                    weight_idx.append([i])
            print('label_types', torch.stack(label_types), torch.stack(label_types).sum(dim = 1).max())
            #weights = [torch.zeros(len(idx), NUM_CATEGORIES) for idx in weight_idx]
            weight_idx_flatten = torch.tensor([ww for w in weight_idx for ww in w])
            weights = [torch.cat(weights)[group_idx][weight_idx_flatten][ sum(list(map(len, weight_idx))[:i]) : sum(list(map(len, weight_idx))[:i]) + len(weight_idx[i])] for i in range(len(weight_idx))]

            learning_rates = (torch.log(torch.stack(label_types).sum(dim=1)) / torch.log(torch.stack(label_types).sum(dim = 1).max())).cuda()
            learning_rates[torch.isinf(learning_rates)] = 0
            learning_rates[torch.isnan(learning_rates)] = 0
            print('learning_rates', learning_rates)
            #weight_idx_flatten = torch.tensor([ww for w in weight_idx for ww in w])
            group_idx = torch.argsort(weight_idx_flatten)

            weight_params = []
            for p, l in zip(weights, learning_rates):
                p.requires_grad = True
                weight_params.append({'params': p, 'lr': l * weight_lr})
            optimizer = optim.Adam([{'params': resnet50.parameters()}] + weight_params, lr=args.lr, weight_decay=args.weightDecay, betas=MOMENTUM)
        #'''
        
        acc_total, f1_mi, f1_ma, mAP = test_small(val_loader,resnet50,device)

    if f1_ma > of1 and f1_mi > cf1:
        of1 = f1_ma
        cf1 = f1_mi    
        
    if f1_mi > best_mi:
        if args.saveModel:
            save(f'{PATH_MODEL_PARAMS}/{current_epoch}_f1mi.pt',resnet50)
        best_mi = f1_mi
        count_early_stop = 0
    if f1_ma > best_ma:
        if args.saveModel:
            save(f'{PATH_MODEL_PARAMS}/{current_epoch}_f1ma.pt',resnet50)
        best_ma = f1_ma
        count_early_stop = 0
        
    if acc_total > best_acc:
        if args.saveModel:
            save(f'{PATH_MODEL_PARAMS}/{current_epoch}_acc.pt',resnet50)
        best_acc = acc_total
        count_early_stop = 0
        
    if mAP > best_mAP:
        if args.saveModel:
            save(f'{PATH_MODEL_PARAMS}/{current_epoch}_mAP.pt',resnet50)
        best_mAP = mAP
        count_early_stop = 0
        
    count_early_stop += 1
    if count_early_stop >= num_early_stop:
        break
  
    val_results = {
    'acc_total': acc_total,
    'f1_mi': f1_mi,
    'f1_ma': f1_ma,
    'mAP': mAP,
    }

    print('===> BEST PERFORMANCE (OF1/CF1): %.3f / %.3f' % (of1, cf1))
    print('===> BEST PERFORMANCE (mi/ma): %.3f / %.3f' % (best_mi, best_ma))
    print('===> BEST PERFORMANCE (acc): %.3f' % (best_acc))
    print('===> BEST PERFORMANCE (mAP): %.3f' % (best_mAP))

    
    if count_early_stop == 1:
        with torch.no_grad():
            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_total, f1_mi, f1_ma, mAP = test(resnet50,args,device)
            print('All', 'Acc.', acc_total, 'F1_Mi', f1_mi, 'F1_Ma', f1_ma, 'mAP', mAP)

            test_results = {
                        'acc_total': acc_total,
                        'f1_mi': f1_mi,
                        'f1_ma': f1_ma,
                        'mAP': mAP,
                    }
            save_results_to_json(current_epoch, val_results, test_results, f"{PATH_MODEL_PARAMS}/results.json")

    t_end = time.time()
    t_use = int((t_end - t_start)/60)
    print(f'Use {t_use}/min ')

print(PATH_MODEL_PARAMS)
