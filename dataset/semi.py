from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None, bbox_path=None, bbox_lbl_path=None, bbox_score_path=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.bbox_path=bbox_path
        self.bbox_lbl_path=bbox_lbl_path
        self.bbox_score_path=bbox_score_path
        

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')#(500, 333, 3)#(333, 500)-----PIL
        #print(np.array(img).shape, img.size)
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
        if(self.mode == 'train_u'):
            self.bbox_path = str(self.bbox_path)
            self.bbox_lbl_path = str(self.bbox_lbl_path)
            self.bbox_score_path = str(self.bbox_score_path)
            image_name = str(id.split(' ')[0].split('/')[-1])
            BBOX = np.load(os.path.join(self.bbox_path,image_name+'.npy'))
            LABELBBOX = np.load(os.path.join(self.bbox_lbl_path, image_name.split('/')[-1]+'.npy'))
            SCOREBBOX = np.load(os.path.join(self.bbox_score_path,image_name.split('/')[-1]+'.npy'))
            BB_all = np.zeros((19,np.array(mask).shape[0],np.array(mask).shape[1]), dtype=np.uint8)#(21, 500, 333)
            for i,BB_ in enumerate(BBOX):
                #print(i)
                LL_ = LABELBBOX[i]
                SC_ = int(SCOREBBOX[i]*100)
                if(SC_>=95):
                    if(LL_==9):
                        cv2.rectangle(BB_all[6,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )                      
                    if(LL_==1):
                        cv2.rectangle(BB_all[11,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==3):
                        cv2.rectangle(BB_all[13,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==7):
                        cv2.rectangle(BB_all[14,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==8):
                        cv2.rectangle(BB_all[15,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==6):
                        cv2.rectangle(BB_all[16,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==5):
                        cv2.rectangle(BB_all[17,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==4):
                        cv2.rectangle(BB_all[18,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==2):
                        cv2.rectangle(BB_all[12,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==10):
                        cv2.rectangle(BB_all[7,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
    #                 if(LL_==11):
    #                     cv2.rectangle(BB_all[5,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==12):
                        cv2.rectangle(BB_all[3,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
                    if(LL_==13):
                        cv2.rectangle(BB_all[4,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )
    #                 if(LL_==14):
    #                     cv2.rectangle(BB_all[9,:,:],(int(BB_[0]),int(BB_[1])),(int(BB_[2]) ,int(BB_[3]) ),[1],-1 )  
                    
            #BB_all=Image.fromarray(np.uint8(BB_all))
            BB_all = np.array(BB_all, dtype=np.uint8)
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id
        if self.mode == 'train_l':
            img, mask = resize(img, mask, (0.5, 2.0))
            ignore_value = 254 if self.mode == 'train_u' else 255
            img, mask = crop(img, mask, self.size, ignore_value)
            img, mask = hflip(img, mask, p=0.5)
        else:
            img, mask, BB_all = resize(img, mask, (0.5, 2.0), BBOX=BB_all)
            ignore_value = 254 if self.mode == 'train_u' else 255
            img, mask,BB_all = crop(img, mask, self.size, ignore_value, BB_all=BB_all)
            img, mask, BB_all = hflip(img, mask, p=0.5, BB_all=BB_all)
        if self.mode == 'train_l':
            return normalize(img, mask)


        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        final_BB_all=[]
        for _i_ in range(len(BB_all)):
            final_BB_all.append(np.array(BB_all[_i_], dtype=np.uint8))
        final_BB_all = np.stack(final_BB_all, axis=0)#(21, 321, 321)
        if(254 in list(np.unique(final_BB_all))):
            final_BB_all[final_BB_all==254]=255
        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2, final_BB_all

    def __len__(self):
        return len(self.ids)
