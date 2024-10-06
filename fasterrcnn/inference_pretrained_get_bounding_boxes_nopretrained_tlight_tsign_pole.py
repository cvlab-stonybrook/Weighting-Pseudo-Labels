import os
import numpy as np
import torch
from PIL import Image
import json
import albumentations as A
import sys
#sys.path.append('../')
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
import os
import imageio
import cv2

#PATH = './val'
CONF = 90
MAIN_FOLDER = './nopretrained_tlight_tsign_pole_results'
if not os.path.exists(MAIN_FOLDER):
    os.makedirs(MAIN_FOLDER)
PATH_SAVE_BOXES = MAIN_FOLDER+'/train_split0_183_new_result_images_bounding_box_'+str(CONF)
if not os.path.exists(PATH_SAVE_BOXES):
    os.makedirs(PATH_SAVE_BOXES)
PATH_SAVE_LABELS = MAIN_FOLDER+'/train_split0_183_new_result_images_labels_'+str(CONF)
if not os.path.exists(PATH_SAVE_LABELS):
    os.makedirs(PATH_SAVE_LABELS)
PATH_SAVE_SCORES = MAIN_FOLDER+'/train_split0_183_new_result_images_scores_'+str(CONF)
if not os.path.exists(PATH_SAVE_SCORES):
    os.makedirs(PATH_SAVE_SCORES)
PATH_SAVE_IMAGES = MAIN_FOLDER+'/train_split0_183_new_result_images_unlabeled_images_'+str(CONF)
if not os.path.exists(PATH_SAVE_IMAGES):
    os.makedirs(PATH_SAVE_IMAGES) 
#model_weight_traffic_light_split0_183_nopretrained_tlight_tsign_pole
model_weight_path = './model_weight_traffic_light_split0_183_nopretrained_tlight_tsign_pole/bestmodel_2363.pth'
#PATH_SAVE = './result_images'
def load_ckp(checkpoint_fpath, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    #print(checkpoint['state_dict'].keys())
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['train_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model,  checkpoint['epoch'], valid_loss_min.item()
    
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 12
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model, epoch_num,loss = load_ckp(model_weight_path,model)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(params, lr=0.000005,weight_decay=0.0005)
model, epoch_num,loss = load_ckp(model_weight_path,model)
def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


PATH = '/nfs/bigtensor/add_disk0/ironman/data/cityscapes/leftImg8bit/train'
list_images = recursive_glob(rootdir=PATH, suffix=".png")#recursive_glob(rootdir = PATH,suffix=".png",train=True,label= False)

# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
id_class_dict={1:"person",2:"rider",3:"car",4:"bicycle",5:"motorcycle",6:"train",7:"truck",8:"bus",9:"traffic_light",10:"traffic_signs",11:"poles"}
#id_class_dict={1:"person",2:"rider",3:"car",4:"bicycle",5:"motorcycle",6:"train",7:"truck",8:"bus"}
#id_color_dict = {1:(255,0,0),2:(255,255,255),3:(0,255,0),4:(0,0,255),5:(255,255,0),6:(255,0,255),7:(0,255,255),8:(153,0,76)}
for image_n in list_images:
    image_name = image_n.split('/')[-1]
    if("checkpoint" not in image_name):
        model.eval()
        image_path = image_n
        image = imageio.imread(image_path)
        trans = T.ToTensor()
        #print("check")
        #print(image.shape)
         #print(dsds)
        model_input = trans(image).cuda()
        model_output = model([model_input])
        #print(model_input.shape)

        #print(dsds)
        temp_sample = np.ascontiguousarray(model_input.permute(1,2,0).cpu().numpy())*255
        boxes = model_output[0]['boxes'].data.cpu().numpy()
        scores = model_output[0]['scores'].data.cpu().numpy()  
        labels = model_output[0]['labels'].data.cpu().numpy()
        np.save(PATH_SAVE_BOXES+'/'+image_name, boxes)
        np.save(PATH_SAVE_LABELS+'/'+image_name,labels)
        np.save(PATH_SAVE_SCORES+'/'+image_name,scores)
        count=0
        for i,box in enumerate(boxes):
            #print(box)
            sample = temp_sample.copy()
            confidence= int(scores[i]*100)
            if(confidence>CONF):
                label_box_id = int(labels[i])

                PATH_SAVE_INDEPTH_CLASS = os.path.join(PATH_SAVE_IMAGES,str(label_box_id))
                if not os.path.exists(PATH_SAVE_INDEPTH_CLASS):
                    os.makedirs(PATH_SAVE_INDEPTH_CLASS)    
    
                cv2.rectangle(sample,(int(box[0]),int(box[1])),(int(box[2]) ,int(box[3]) ),[255,0,0],4 )
                cv2.putText(sample, id_class_dict[int(labels[i])]+'_'+str(confidence), (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12),2)
                cv2.imwrite(PATH_SAVE_INDEPTH_CLASS+'/'+image_name+'_'+str(count)+'.jpg',sample)
                count+=1