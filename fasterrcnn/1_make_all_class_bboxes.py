#####first remove the test directory from the image folder and then add then back once all the code is run.
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
import pickle
CLASS_CONSIDER_LIST = [20,19,17,21,12,13,11,8,22]
PATH_IMAGES = './data/cityscapes/leftImg8bit/'
PATH_LABELS = './data/cityscapes/gtFine/'
final_dict={}
for root_dir,sub_dirs,file_img in os.walk(PATH_IMAGES):
    for sub_dir in sub_dirs:
        PATH_SPLIT_FOLDER = os.path.join(PATH_IMAGES,sub_dir)
        for _r_,_d_,_f_ in os.walk(PATH_SPLIT_FOLDER):
            for city in _d_:
                PATH_CITY_FOLDER = os.path.join(PATH_SPLIT_FOLDER,city)
                for _,_,image_name_list in os.walk(PATH_CITY_FOLDER):
                    for image_name in image_name_list:
                        if('checkpoint' not in image_name):
                            PATH_IMAGE = os.path.join(PATH_CITY_FOLDER,image_name)
                            mask_name = image_name.replace('leftImg8bit','gtFine_instanceIds')
                            MASK_PATH = PATH_CITY_FOLDER.replace( 'leftImg8bit','gtFine')
                            MASK_PATH = os.path.join(MASK_PATH,mask_name)
                            img = cv2.imread(PATH_IMAGE)
                            img = Image.open(PATH_IMAGE).convert("RGB")
                            img = np.asarray(img)
                            inst = cv2.imread(MASK_PATH, cv2.IMREAD_ANYDEPTH)
                            for CONS_CLASS in CLASS_CONSIDER_LIST:
                                SAVE_IMAGE_DIRECTORY = str(CONS_CLASS)
                                if not os.path.exists(SAVE_IMAGE_DIRECTORY):
                                    os.makedirs(SAVE_IMAGE_DIRECTORY)
                                img_consider = img.copy()
                                mask_consider = np.zeros(inst.shape, np.uint8)
                                for id_ in np.unique(inst):
                                    if(id_<1000):
                                        class_ = id_
                                    else:
                                        class_ = id_//1000
                                    if(class_ == CONS_CLASS):
                                        #img_consider[inst==id_]=[255,0,0]
                                        mask_consider[inst==id_]=1
                                        output = cv2.connectedComponentsWithStats(mask_consider, 8, cv2.CV_32S)
                                        (numLabels, labels, stats, centroids) = output
                                        flag=0#to check if there are at all elements above area 90
                                        #loop over the number of unique connected component labels
                                        for i in range(1, numLabels):
                                            # extract the connected component statistics and centroid for
                                            # the current label
                                            x = stats[i, cv2.CC_STAT_LEFT]
                                            y = stats[i, cv2.CC_STAT_TOP]
                                            w = stats[i, cv2.CC_STAT_WIDTH]
                                            h = stats[i, cv2.CC_STAT_HEIGHT]
                                            area = stats[i, cv2.CC_STAT_AREA]
                                            centroid = centroids[i]
                                            crop_img = img_consider[y:y+h,x:x+w,:]   
                                            crop_mask = mask_consider[y:y+h,x:x+w]
                                            if(np.sum(crop_mask)>90):
                                                flag=1
                                                first = x
                                                second = y
                                                third = x+w
                                                fourth = y+h
                                                cv2.rectangle(img_consider, (first, second), (third, fourth), (255,0,0), 4)
                                                if(image_name not in final_dict.keys()):
                                                    final_dict[image_name]={}
                                                if(CONS_CLASS not in final_dict[image_name].keys()):
                                                    final_dict[image_name][CONS_CLASS]=[]
                                                final_dict[image_name][CONS_CLASS].append([first,second,third,fourth])                                                    
                                        if(flag==1):
                                            plt.imsave(SAVE_IMAGE_DIRECTORY+'/'+image_name,img_consider)
            break
    break

with open('all_bboxes.pkl', 'wb') as f:
    pickle.dump(final_dict, f)