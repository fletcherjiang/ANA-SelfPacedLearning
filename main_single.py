import os
import cv2
import sys
import copy
from tqdm import tqdm
import torch
import time
from config_single import *  # Import everything from config.py
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

from utils_single import set_random_seed, Logger, inference, get_small, train, test, save, get_images, test_single, get_train_dataset, get_train_dataset_single, get_train_dataloader,save_results_to_json

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
    PATH_MODEL_PARAMS  = '2.25_single_2_' +'iw_al_val_' + str(args.lr) + 'lr_' + str(args.weight_lr) + 'weight_' + args.initWeight + '_' + args.updateLR + '_trainingLabel_' + str(args.trainingLabel) + '_sampling_' + str(args.sampling) + '_granularity_' + args.granularity
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
val_dataset = get_testdata_single(root=DIR_TEST_IMAGES,
                            annFile=PATH_TEST_ANNFILE, 
                            transform=val_transforms,
                          split = 'val')

print(PATH_TEST_ANNFILE)
# val_indices = list(range(20)) if 20 is not None else list(range(len(val_dataset)))
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
        bce[target == 1] *= F.relu(weight[target == 1])
        return bce.sum()
    
loss_function = WeightedBCELoss()


# ==================================================================
# Main Loop
# ==================================================================
t_start = time.time()
# train_dataset = get_train_dataset()
train_dataset = get_train_dataset_single()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                batch_size=args.trainBatchSize, 
                                num_workers = 4)

print('Initialize weights')
weights = torch.zeros((len(train_dataset), NUM_CATEGORIES))
targets = torch.zeros((len(train_dataset), NUM_CATEGORIES), dtype = torch.float64)

for batch, (_, label, idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing Train Data", unit="batch"):
    tmp = weights[idx]
    tmp[label > 0] = 1
    weights[idx] = tmp
    targets[idx] = label
            
weights.requires_grad = True
print('Initialization Done', time.time() - t_start)

optimizer = optim.Adam([{'params': resnet50.parameters()}, {'params': weights}], lr=args.lr, weight_decay=args.weightDecay, betas=MOMENTUM)
# print('label_types', torch.stack(label_types), torch.stack(label_types).sum(dim = 1).max())

num_early_stop = 5
count_early_stop = 0
count_early_stop_single = 0
count_early_stop_multiple = 0

for current_epoch in range(1, args.epoch + 1):
    print('\n===> epoch: %d/%d' % (current_epoch, args.epoch))
    print('weights', weights[args.trainBatchSize:args.trainBatchSize * 2])

    if current_epoch % 2 == 1:
        with torch.no_grad():
            train_loader = get_train_dataloader(resnet50, weights, targets, train_dataset, args)
    
    train_cp, train_cr, train_cf1, train_op, train_or, train_of1 = train(optimizer, train_loader, weights, args, resnet50, loss_function, device)

    with torch.no_grad():
        
        acc_0 ,acc_1 ,acc_2 ,acc_3 ,acc_4 ,acc_5 ,acc_6 ,acc_7 ,acc_total, f1_mi, f1_ma, mAP = test_single(resnet50,args,device)
        
        # acc_total, f1_mi, f1_ma, mAP = test_single(val_loader,resnet50,device)

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
            acc_0 ,acc_1 ,acc_2 ,acc_3 ,acc_4 ,acc_5 ,acc_6 ,acc_7 ,acc_total_single, f1_mi_single, f1_ma_single, mAP_single = test_single(resnet50,args,device)
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



