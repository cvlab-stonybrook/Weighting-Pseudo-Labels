# Weighting Pseudo-Labels via High-Activation Feature Index Similarity and Object Detection for Semi-Supervised Segmentation [ECCV 2024]


## Contact

If you have any questions, please email Prantik Howlader at phowlader@cs.stonybrook.edu.

### Dataset

- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)

Please modify your dataset path in configuration files.
```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
    
```
Generating bounding boxes for training segmentation network. check fasterrcnn for instructions

## Usage

### UniMatch + Ours

```bash
# use torch.distributed.launch
sh scripts/train.sh
```
## Citation

If you find this project useful, please consider citing:

```bibtex
@article{howlader2024weighting,
  title={Weighting Pseudo-Labels via High-Activation Feature Index Similarity and Object Detection for Semi-Supervised Segmentation},
  author={Howlader, Prantik and Le, Hieu and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2407.12630},
  year={2024}
}
```
