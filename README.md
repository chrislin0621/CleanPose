# [ICCV2025] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation
### Abstract
In the effort to achieve robust and generalizable category-level object pose estimation, recent methods primarily focus on learning fundamental representations from data. However, the inherent biases within the data are often overlooked: the repeated training samples and similar environments may mislead the models to over-rely on specific patterns, hindering models' performance on novel instances. In this paper, we present CleanPose, a novel method that mitigates the data biases to enhance category-level pose estimation by integrating causal learning and knowledge distillation. By incorporating key causal variables (structural information and hidden confounders) into causal modeling, we propose the causal inference module based on front-door adjustment, which promotes unbiased estimation by reducing potential spurious correlations. Additionally, to further confront the data bias at the feature level, we devise a residual-based knowledge distillation approach to transfer unbiased semantic knowledge from 3D foundation model, providing comprehensive causal supervision. Extensive experiments across multiple benchmarks (REAL275, CAMERA25 and HouseCat6D) hightlight the superiority of proposed CleanPose over state-of-the-art methods.

![](/main_v3.png)

[[paper](https://arxiv.org/pdf/2502.01312)]

## Environment Settings
The code has been tested with

- python 3.9
- torch 1.12
- cuda 12.4

Some dependencies:
```
pip install gorilla-core==0.2.5.3
pip install opencv-python

cd model/pointnet2
python setup.py install
```

## Data Preparation
### NOCS dataset
- Download and preprocess the dataset following [DPDN](https://github.com/JiehongLin/Self-DPDN)
- Download and unzip the segmentation results [here](http://home.ustc.edu.cn/~llinxiao/segmentation_results.zip)

Put them under ```PROJ_DIR/data```and the final file structure is as follows:
```
data
├── camera
│   ├── train
│   ├── val
│   ├── train_list_all.txt
│   ├── train_list.txt
│   ├── val_list_all.txt
├── real
│   ├── train
│   ├── test
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── test_list_all.txt
├── segmentation_results
│   ├── CAMERA25
│   └── REAL275
├── camera_full_depths
├── gts
└── obj_models
```
### HouseCat6D
Download and unzip the dataset from [HouseCat6D](https://sites.google.com/view/housecat6d) and the final file structure is as follows:
```
housecat6d
├── scene**
├── val_scene*
├── test_scene*
└── obj_models_small_size_final
```

## Other Preparation
### Confounder queue generation
You can generate the queue list with following command or utilize the pre-extracted pkl file from [this link](https://drive.google.com/drive/folders/15D9kkISuEP1z6yBZhBItdp4N26wDYcJB?usp=drive_link).
```
python queue_extraction.py
```
### ULIP model weights
Download the pretrained weights of PointBERT and Pointnet2_ssg from [ULIP Hugging Face](https://huggingface.co/datasets/SFXX/ulip) or [this link](https://drive.google.com/drive/folders/1yQhaP7AWtgu5NOW1GTVO23ytcGLt_wPW?usp=drive_link). And the pretrained weights file structure is as follows:
```
model
├── pointbert
│   ├── pretrained_model
│       └── pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt
├── pointnet2
├── pointnet2_ulip
│   ├── pretrained_model
│       └── pretrained_models_ckpt_zero-sho_classification_checkpoint_pointnet2_ssg.pt
```

## Train
### Training on NOCS
```
python train.py --config config/REAL/camera_real.yaml
```
### Training on HouseCat6D
```
python train_housecat6d.py --config config/HouseCat6D/housecat6d.yaml
```
## Evaluate 
- Evaluate on NOCS:
```
python test.py --config config/REAL/camera_real.yaml --test_epoch 30
```
- Evaluate on HouseCat6D:
```
python test_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test_epoch 150
```
## Results
You can download our training logs, detailed metrics for each category and checkpoints [here]().
### REAL275 testset:

| IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| 83.3 | 81.2 | 62.7 | 61.7| 67.6 | 78.3 | 86.3 |

### CAMERA25 testset:
| IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| 94.8 | 94.3 | 92.5 | 80.3 | 84.2 | 87.7 | 92.7 |

### HouseCat6D testset:
| IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| 89.2 | 79.8 | 53.9 | 22.4 | 24.1 | 51.6 | 56.5 |

## Visualization
For visualization, please run
```
python visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
```
## Acknowledgements
Our implementation leverages the code from [DPDN](https://github.com/JiehongLin/Self-DPDN), [AG-Pose](https://github.com/Leeiieeo/AG-Pose) and [GOAT](https://github.com/CrystalSixone/VLN-GOAT). Thank them for their excellent works!
## Citation
If our work is useful to you, please consider citing our paper using the following BibTeX entry.
```
@inproceedings{lin2025cleanpose,
  title={Cleanpose: Category-level object pose estimation via causal learning and knowledge distillation},
  author={Lin, Xiao and Peng, Yun and Wang, Liuyi and Zhong, Xianyou and Zhu, Minghao and Yang, Jingwei and Feng, Yi and Liu, Chengju and Chen, Qijun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
## License
Our code is released under MIT License (see LICENSE file for details).

