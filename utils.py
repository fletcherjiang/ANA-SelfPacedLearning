# utils.py
import random
import numpy as np
import pandas as pd
import torch
import sys
import time
from tqdm import tqdm
import os
import sklearn.metrics as sm
from torchvision import transforms as transforms
from config import *  # Import everything from config.py
import torch.nn.functional as F
#自定义函数
from data_weighted_single import get_traindata as get_traindata_single, get_testdata as get_testdata_single
from data_weighted_multiple_filename import get_traindata as get_traindata_multiple, get_testdata as get_testdata_multiple
from data_weighted_filename import get_traindata, get_testdata
from torch.utils.data.sampler import WeightedRandomSampler
import concurrent.futures
import json
from PIL import Image
import cv2


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass




def inference(model, images):
    logits = model(images)
    probabilities = torch.sigmoid(logits)
    return logits, probabilities


def get_small(path):
    '''
    接受一个大图的路径，返回一个tensor
    
    shape like:[20,3,224,224]
    
    '''
    split_size = 448
    normalize = transforms.Normalize(mean=[0.005, 0.190, 0.006],
                                  std=[0.008, 0.102, 0.008]) #计算得出
    val_transforms = transforms.Compose([
                        transforms.Resize((224, 224)), 
                        transforms.ToTensor(),
                        normalize
                      ]) 
    
    # Read image without decryption (open source version)
    temp_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)[:,:,::-1]
    
    shape0 = temp_image.shape[0]
    shape1 = temp_image.shape[1]
    
    num0 = shape0//split_size
    num1 = shape1//split_size
    
    small_images = torch.zeros((num0*num1 , 3 , 224,224))
    i = 0
    for j in range(num0):
        for k in range(num1):
            
            small_image = temp_image[j*split_size:(j+1)*split_size,
                                     k*split_size:(k+1)*split_size,
                                     :]
            small_image = Image.fromarray(small_image.astype('uint8')).convert('RGB')
            small_image = val_transforms(small_image)
            small_images[i,:,:,:] = small_image
            i += 1
            
    return small_images


def train(optimizer, train_loader, weights, args, resnet50, group_idx, loss_function, targets, device):
    print('train:')
    t = time.time()
    resnet50.train()
    train_loss = 0. # sum of train loss up to current batch

    tmp = None
    print(f'len dataloader {len(train_loader)}')
    # for batch_num, (data, label, index, filenames) in enumerate(train_loader):
    for batch_num, (data, label, index, filenames) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress", unit="batch"):
        #print('Batch #', batch_num)
        if label.sum() == 0:
            continue
        #print('loader', time.time() - t)
        
        if tmp is None:
            tmp = index
        else:
            tmp = torch.cat((tmp, index))
        
        #weight = weights[index]
        #New version of accessing weights
        weight = []
        for i in index:
            offset = 0
            for w in weights:
                if group_idx[i] < offset + len(w):
                    weight.append(w[group_idx[i] - offset])
                    break
                offset += len(w)
        weight = torch.stack(weight)
        
        #避免没有标注的数据
        if GPU_IN_USE:
            data, label, weight = data.to(device), label.to(device), weight.to(device)
        
        target = label.float()
        if args.trainingLabel != 'real':
            if args.granularity == 'sample':
                #target = label * F.relu(weight.nan_to_num())[:, None]
                target = label * F.relu(weight.detach().nan_to_num())[:, None]
            else:   
                #target = label * F.relu(weight.nan_to_num())
                #out method
                target = label * F.relu(weight.detach().nan_to_num())
            target = (target - target.min(dim = 1, keepdim = True)[0]) / (target.max(dim = 1, keepdim = True)[0] - target.min(dim = 1, keepdim = True)[0])

            tmp_idx = torch.isnan(target)
            target[torch.isnan(target)] = 0

            if args.trainingLabel == 'pseudo':
                target = target.detach().bernoulli()

            targets[index] = target.detach().cpu()
            target = target.float()

        optimizer.zero_grad()

        # logits, probabilities
        output, probabilities = inference(resnet50,data)
        loss = loss_function(probabilities, target, weight)
        loss.backward()
        optimizer.step()
        train_loss += loss
    print('train done', time.time() - t)
    return 1,1,1,1,1,1

def test(resnet50,args,device):
    print('test:')
    t = time.time()
    resnet50.eval()
    

    dataframe = pd.read_csv('data_8_class_with_979val_979test_4605train.csv')
    dataframe = dataframe[dataframe['Split'] ==  'test']
    dataframe = dataframe.reset_index(drop = True)

    # Update this path to point to your own data directory
    path_origin = './data/images/'
    analysis_prob = []
    analysis_tar = []
    
    batch_imgs = None
    batch_paths = []

    # for i in range(len(dataframe)):
    for i in tqdm(range(len(dataframe)), desc="Testing Progress", unit="sample"):
        temp = dataframe.iloc[i]
        temp_path = path_origin + temp['path']
        temp_target = [int(j) for j in str(temp['TARGET']).split()]
        analysis_tar.append(temp_target)
        batch_paths.append(temp_path)
        
        if (i + 1) % args.testBatchSize == 0 or i == len(dataframe) - 1:
            #Multi-thread
            num_imgs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
                for temp_images in executor.map(get_small, batch_paths):
                    temp_images = temp_images.to(device)
                    num_imgs.append(len(temp_images))
                    if batch_imgs is None:
                        batch_imgs = temp_images
                    else:
                        batch_imgs = torch.cat((batch_imgs, temp_images))
            
            output, probabilities = inference(resnet50, batch_imgs)
            analysis_prob += [probabilities[sum(num_imgs[:i]) : sum(num_imgs[:i]) + num_imgs[i]] for i in range(len(num_imgs))]
            
            batch_imgs = None
            batch_paths = []
 
    ##prediction
    temp = np.zeros((979,8))
    for i in range(len(analysis_tar)):
        temp[i,:] = analysis_prob[i].max(dim=0)[0].detach().cpu().numpy()
    tt = temp>0.5
    
    ##计算真实标签
    ans = np.zeros((979,8))
    for i in range(len(analysis_tar)):
        cur_ans = analysis_tar[i] 
        _ = np.zeros((1,8))
        for j in cur_ans:
            _[0][j] = 1
        ans[i,:] = _
        
    targets = ans
    prediction_labels = tt
    
    mAP = sm.average_precision_score(targets, temp)
    acc_total = sm.accuracy_score(targets,prediction_labels)
    f1_mi = sm.f1_score(targets,prediction_labels,average='micro')
    f1_ma = sm.f1_score(targets,prediction_labels,average='macro')

    acc_0 = sm.accuracy_score(targets[:,0],prediction_labels[:,0])
    acc_1 = sm.accuracy_score(targets[:,1],prediction_labels[:,1])
    acc_2 = sm.accuracy_score(targets[:,2],prediction_labels[:,2])
    acc_3 = sm.accuracy_score(targets[:,3],prediction_labels[:,3])
    acc_4 = sm.accuracy_score(targets[:,4],prediction_labels[:,4])
    acc_5 = sm.accuracy_score(targets[:,5],prediction_labels[:,5])
    acc_6 = sm.accuracy_score(targets[:,6],prediction_labels[:,6])
    acc_7 = sm.accuracy_score(targets[:,7],prediction_labels[:,7])

    #print(sm.classification_report(targets,prediction_labels))
    print('test done', time.time() - t)
    return acc_0 ,acc_1 ,acc_2 ,acc_3 ,acc_4 ,acc_5 ,acc_6 ,acc_7 , acc_total,f1_mi,f1_ma, mAP


# ==================================================================
# Save Model
# ==================================================================
def save(tag,resnet50):
    torch.save(resnet50.state_dict(), tag)
    print('Checkpoint saved to {}'.format(tag))

# ==================================================================
# 准备数据
# ==================================================================
def get_images(path):
    
    ## 从single数据集中抽取数据
    ##返回单张图片tensor
    
    # Read image without decryption (open source version)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    normalize = transforms.Normalize(mean=[0.005, 0.190, 0.006],
                                  std=[0.008, 0.102, 0.008]) #计算得出
    N =224
    val_transforms = transforms.Compose([
                        transforms.Resize((N, N)), 
                        transforms.ToTensor(),
                        normalize
                      ]) 
    
    image = val_transforms(image)
    image = image.unsqueeze(0)
    return image

def test_small(val_loader,resnet50,device):
    print('val:', len(val_loader))
    t = time.time()
    resnet50.eval()


    prediction_labels = []
    targets = []
    prob = []
    # for batch_num, (data, target, _, _) in enumerate(val_loader):
    for batch_num, (data, target, _, _) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation Progress", unit="batch"):
        if target.sum() == 0:
            continue
        
        target = target.float() 
    
        if GPU_IN_USE:
            data, target = data.to(device), target.to(device) 

        
        output, probabilities = inference(resnet50, data)
        if len(prob) == 0:  # 检查 prob 是否为空
            prob = probabilities.detach().cpu().numpy()
        else:
            prob = np.vstack((prob, probabilities.detach().cpu().numpy()))


        prediction_label = np.zeros(probabilities.shape)
        prediction_label[probabilities.detach().cpu().numpy() > 0.5]= 1

        target = target.detach().cpu().numpy()

        if len(prediction_labels) == 0:
            prediction_labels = prediction_label
        else:
            prediction_labels = np.vstack((prediction_labels , prediction_label))
        
        if len(targets) == 0:
            targets = target
        else:
            targets = np.vstack((targets , target))


    
    mAP = sm.average_precision_score(targets, prob)
    acc_total = sm.accuracy_score(targets,prediction_labels)
    f1_mi = sm.f1_score(targets,prediction_labels,average='micro')
    f1_ma = sm.f1_score(targets,prediction_labels,average='macro')

    acc_0 = sm.accuracy_score(targets[:,0],prediction_labels[:,0])
    acc_1 = sm.accuracy_score(targets[:,1],prediction_labels[:,1])
    acc_2 = sm.accuracy_score(targets[:,2],prediction_labels[:,2])
    acc_3 = sm.accuracy_score(targets[:,3],prediction_labels[:,3])
    acc_4 = sm.accuracy_score(targets[:,4],prediction_labels[:,4])
    acc_5 = sm.accuracy_score(targets[:,5],prediction_labels[:,5])
    acc_6 = sm.accuracy_score(targets[:,6],prediction_labels[:,6])
    acc_7 = sm.accuracy_score(targets[:,7],prediction_labels[:,7])

    #print(sm.classification_report(targets,prediction_labels))
    print('validation done', time.time() - t)
    return acc_total, f1_mi, f1_ma, mAP


    
def get_train_dataset():
    path_csv = 'all_single_small_82240_19400_19330.csv'
    normalize = transforms.Normalize(mean=[0.005, 0.190, 0.006],
                                  std=[0.008, 0.102, 0.008]) #计算得出
    
    train_transforms = transforms.Compose([
                         transforms.RandomHorizontalFlip(p=0.5), 
                         transforms.Resize((N, N)),
                         transforms.ToTensor(),
                         normalize
                      ])

    train_dataset = get_traindata(root=DIR_TRAIN_IMAGES,
                                  annFile=path_csv, 
                                  transform=train_transforms,
                                  split = 'train')
    return train_dataset


def get_train_dataloader(model, weights, labels, train_dataset, args):
    if args.granularity == 'sample':
        sw = F.relu(weights.nan_to_num())
    else:
        #s_i= Max(ReLU)
        sw = (F.relu(weights.nan_to_num()) * labels).max(dim = 1)[0]
    
    sampler = None
    if args.sampling:
        # weights = weights.view(-1)  # 将权重展平为一维
        sampler = WeightedRandomSampler(sw, num_samples=len(weights), replacement=True)

    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, 
                                batch_size=args.trainBatchSize,
                                num_workers = 4,
                                sampler = sampler   )
    
    
    return train_loader





def save_results_to_json(epoch, val_results, test_results, file_path):
    # Create the dictionary to be saved in the specified format
    results = {
        "epoch": str(epoch),  # Ensure epoch is a string
        "val_results": val_results,
        "test_results": test_results
    }
    
    # Check if the file exists to determine whether to append or create a new file
    file_exists = os.path.isfile(file_path)
    
    # Open the file in append mode ('a'), create a new one if it doesn't exist
    with open(file_path, 'a' if file_exists else 'w') as f:
        if not file_exists:
            f.write("[")  # Start with the opening square bracket for an array
        json.dump(results, f, indent=4)  # Dump the results dictionary as a JSON string with indentation
        f.write("\n")  # Add a newline after each JSON object
        if not file_exists:
            f.write("]")  # Close the JSON array after the last entry
