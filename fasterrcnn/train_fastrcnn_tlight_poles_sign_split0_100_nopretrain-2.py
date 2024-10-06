import os
import numpy as np
import torch
from PIL import Image
import json
import albumentations as A
import sys
sys.path.append('../')
#from engine import train_one_epoch, evaluate
import scipy.misc as m
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import shutil
#import transforms as T
import utils


def save_ckp(state, is_best, checkpoint_path, best_model_path,epoch):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path+str(epoch)+'.pth')
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path+str(epoch)+'.pth', best_model_path+str(epoch)+'.pth')
        
        
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
#folders train: train_183,train_1488,train_372,train_92
class CityScapesDataset(data.Dataset):
    def __init__(self,train=False):
        #self.transforms = transforms
        if(train):
            self.img_path = './train_183'
            self.annotation_path = './cityscapes_coco/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json'
        else:
            self.img_path = './val'
            self.annotation_path = './cityscapes_coco/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
        self.image_list = os.listdir(self.img_path)
        #self.mean = (73.15835921/255, 82.90891754/255, 72.39239876/255)
        f = open(self.annotation_path)
        data = json.load(f)
        self.traffic_lights={}
        with open('./BBOX_annotation/19.txt') as f:
            lines = f.readlines()
        for line in lines:
            #print(line)
            name_image_tl = line.strip().split("||")[0]
            bb_0_tl = int(float(line.strip().split("||")[1]))
            bb_1_tl = int(float(line.strip().split("||")[2]))
            bb_2_tl = int(float(line.strip().split("||")[3]))
            bb_3_tl = int(float(line.strip().split("||")[4]))
            if name_image_tl not in self.traffic_lights.keys():
                self.traffic_lights[name_image_tl]=[]
            self.traffic_lights[name_image_tl].append([bb_0_tl,bb_1_tl,bb_2_tl,bb_3_tl])
        self.traffic_signs={}
        with open('./BBOX_annotation/20.txt') as f:
            lines = f.readlines()
        for line in lines:
            #print(line)
            name_image_tl = line.strip().split("||")[0]
            bb_0_tl = int(float(line.strip().split("||")[1]))
            bb_1_tl = int(float(line.strip().split("||")[2]))
            bb_2_tl = int(float(line.strip().split("||")[3]))
            bb_3_tl = int(float(line.strip().split("||")[4]))
            if name_image_tl not in self.traffic_signs.keys():
                self.traffic_signs[name_image_tl]=[]
            self.traffic_signs[name_image_tl].append([bb_0_tl,bb_1_tl,bb_2_tl,bb_3_tl])
        self.poles={}
        with open('./BBOX_annotation/17.txt') as f:
            lines = f.readlines()
        for line in lines:
            #print(line)
            name_image_tl = line.strip().split("||")[0]
            bb_0_tl = int(float(line.strip().split("||")[1]))
            bb_1_tl = int(float(line.strip().split("||")[2]))
            bb_2_tl = int(float(line.strip().split("||")[3]))
            bb_3_tl = int(float(line.strip().split("||")[4]))
            if name_image_tl not in self.poles.keys():
                self.poles[name_image_tl]=[]
            self.poles[name_image_tl].append([bb_0_tl,bb_1_tl,bb_2_tl,bb_3_tl])
            #print(sasas)
        self.image_name_id_dict={}
        for image_info in data['images']:
            image_id = image_info['id']
            image_name = image_info['file_name'].split('/')[-1]
            self.image_name_id_dict[image_name]=image_id
        self.id_annotations={}
        for annotations in data['annotations']:
            image_id = annotations['image_id']
            category_id = annotations['category_id']
            bbox = annotations['bbox']
            if(image_id not in self.id_annotations.keys()):
                self.id_annotations[image_id]=[]
            self.id_annotations[image_id].append([category_id,bbox])
        #from 1024x2048 to resize 512x1024 to crop input_size (512x512)
        if(train):
            self.transform = A.Compose([
                A.Resize(512,1024,p=1),
                A.RandomCrop(width=512, height=512),
                A.HorizontalFlip(p=0.7),
                A.RandomBrightnessContrast(p=0.6),
            ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(512,1024,p=1),
#                 A.RandomCrop(width=512, height=512),
#                 A.HorizontalFlip(p=0.5),
#                 A.RandomBrightnessContrast(p=0.2),
            ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))  
        #self.image_name_id_dict.pop('bochum_000000_031152_leftImg8bit.png')
        #self.image_list.remove('bochum_000000_031152_leftImg8bit.png')

        #print(self.id_annotations)
    def __getitem__(self,idx):
        image_name = self.image_list[idx]
        #print("AAAAA")
        #print(image_name)
        image_path = os.path.join(self.img_path,image_name)
        #print(image_path)
        image_id = self.image_name_id_dict[image_name] 
        #print(image_id)
        image_annotations = self.id_annotations[image_id]
        img = Image.open(image_path).convert("RGB")
        labels=[]
        boxes=[]
        for annotation in image_annotations[3:]:
            #numpy y,x,deltay,deltax
            boxes.append([annotation[1][0],annotation[1][1],annotation[1][0]+annotation[1][2],annotation[1][1]+annotation[1][3]])
            labels.append(annotation[0])
        if image_name in self.traffic_lights.keys():
            for traffic_light_list in self.traffic_lights[image_name]:
                boxes.append([traffic_light_list[0],traffic_light_list[1],traffic_light_list[2],traffic_light_list[3]])
                labels.append(9)
        if image_name in self.traffic_signs.keys():
            for traffic_sign_list in self.traffic_signs[image_name]:
                boxes.append([traffic_sign_list[0],traffic_sign_list[1],traffic_sign_list[2],traffic_sign_list[3]])
                labels.append(10)  
        if image_name in self.poles.keys():
            for pole_list in self.poles[image_name]:
                boxes.append([pole_list[0],pole_list[1],pole_list[2],pole_list[3]])
                labels.append(11)   
        transformed = self.transform(image=np.asarray(img), bboxes=boxes, class_labels=labels)
        transformed_image = transformed['image']
        transformed_boxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']
        pil_image = Image.fromarray(transformed_image)
        
        boxes = torch.as_tensor(transformed_boxes,dtype=torch.float32)
        labels = torch.as_tensor(transformed_class_labels,dtype=torch.int64)
        image_id = torch.tensor([idx])
        is_crowd = torch.zeros(labels.shape, dtype=torch.int64)

        target={}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = is_crowd 
        img_tranform = T.Compose([T.ToTensor()])
        img = img_tranform(pil_image)

        return img,target,image_name
            

    def __len__(self):
        return len(self.image_list)

    
    
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 12
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


dataset = CityScapesDataset( train=True)
#dataset_test = CityScapesDataset(train=False)
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=3,
    collate_fn=utils.collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(params, lr=0.00005,weight_decay=0.0005)

num_epochs=3000
train_loss_min = 0.9
total_train_loss = []
checkpoint_path = './model_weight_traffic_light_split0_183_nopretrained_tlight_tsign_pole/chkpoint_'
best_model_path = './model_weight_traffic_light_split0_183_nopretrained_tlight_tsign_pole/bestmodel_'

for epoch in range(num_epochs):
    print(f'Epoch :{epoch + 1}')
    #start_time = time.time()
    train_loss = []
    model.train()
    for images, targets, image_ids in data_loader:
        if(targets[0]['labels'].shape[0]>0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            #print(loss_dict)

            losses = sum(loss for loss in loss_dict.values())
            train_loss.append(losses.item())      
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
    #train_loss/len(train_data_loader.dataset)
    epoch_train_loss = np.mean(train_loss)
    total_train_loss.append(epoch_train_loss)
    print(f'Epoch train loss is {epoch_train_loss}')
    
#     if lr_scheduler is not None:
#         lr_scheduler.step()
    
    # create checkpoint variable and add important data
    checkpoint = {
            'epoch': epoch + 1,
            'train_loss_min': epoch_train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    
    # save checkpoint
    #save_ckp(checkpoint, False, checkpoint_path, best_model_path,epoch)
    ## TODO: save the model if validation loss has decreased
    if epoch_train_loss <= train_loss_min:
            print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,epoch_train_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path,epoch)
            train_loss_min = epoch_train_loss