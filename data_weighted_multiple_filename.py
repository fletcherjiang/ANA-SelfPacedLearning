#!/usr/bin/python
#coding=utf-8

import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import copy
import pandas as pd
import scipy.misc as misc
from PIL import Image
import cv2

class ANA(data.Dataset):



    def __init__(self,root,annFile,transform,split):

        self.num_classes = 8
        # self.split = split
        # self.idx_fold = config.data.params.idx_fold
        # self.num_fold = config.data.params.num_fold

        self.transform = transform
        self.imgs_dir = root
        self.split = split
        self.info_dir  =  annFile
        self.info = self.load_info() #通常是csv文件
        self.examples = self.load_examples() #list[(ID,PATH,[label])]
        self.size = len(self.examples)

    def load_examples(self):

        return [(row['path'],row['FILEPATH'], [int(i) for i in str(row['TARGET']).split(' ')] )
                for _,row in self.info.iterrows()]



    def load_info(self):
        info = pd.read_csv(self.info_dir)
        info = info[info['Split']==self.split].reset_index(drop = True)
        info = info[np.array(list(map(len, info['TARGET']))) > 1].reset_index(drop = True) #Read data with multiple labels

#         info = info[info['Split'] == self.split].reset_index(drop = True).loc[:20]
        def generate_filepath(v):
            return os.path.join(self.imgs_dir,v)
        info['FILEPATH'] = info['path'].transform(generate_filepath)

        return info


    def __getitem__(self,index):

        example = self.examples[index]
        filename = example[1]
        real_labels = example[2]  #like[3,6,7]
#         print(example[0])
#         print(real_labels)
        
        
#         label = int(real_labels[0])
        label = [0.0]*self.num_classes
        for i in real_labels:
            label[i] = 1.0

        label = np.array(label).astype(float)

        # Read image without decryption (open source version)
        image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image.astype('uint8')).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)



        # return {'image':image,
        #         'label':label,
        #         'key':example[0]}
        return image, label, index, filename.split('/')[-1][4:]



    def __len__(self):
        return self.size


def get_traindata(root, annFile, transform,split):

    return ANA(root, annFile, transform,split)

def get_testdata(root, annFile, transform,split):
    return ANA(root, annFile, transform,split)

