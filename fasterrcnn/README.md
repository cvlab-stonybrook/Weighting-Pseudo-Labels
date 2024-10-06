## Training Detector
We have the code to first convert the semantic segmentation dataset to bounding boxes to train the detector

### Generating the bounding boxes
python 1_make_all_class_bboxes.py

2_putboxcoorinfile.ipynb

#### Training the detector
python train_fastrcnn_tlight_poles_sign_split0_100_nopretrain-2.py

#### Geting the detector training detections
python inference_pretrained_get_bounding_boxes_nopretrained_tlight_tsign_pole.py
