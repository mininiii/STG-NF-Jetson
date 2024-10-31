# Normalizing Flows for Human Pose Anomaly Detection [ICCV 2023]
[![arXiv](https://img.shields.io/badge/arXiv-<2211.10946>-<COLOR>.svg)](https://arxiv.org/abs/2211.10946)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/normalizing-flows-for-human-pose-anomaly/anomaly-detection-on-shanghaitech)](https://paperswithcode.com/sota/anomaly-detection-on-shanghaitech?p=normalizing-flows-for-human-pose-anomaly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/normalizing-flows-for-human-pose-anomaly/anomaly-detection-on-ubnormal)](https://paperswithcode.com/sota/anomaly-detection-on-ubnormal?p=normalizing-flows-for-human-pose-anomaly)



The official PyTorch implementation of the paper [**"Normalizing Flows for Human Pose Anomaly Detection"**](https://arxiv.org/abs/2211.10946).


![Framework_Overview](data/arch.png)

## Citation
If you find this useful, please cite this work as follows:
```
@InProceedings{Hirschorn_2023_ICCV,
    author    = {Hirschorn, Or and Avidan, Shai},
    title     = {Normalizing Flows for Human Pose Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13545-13554}
}
```

## Getting Started

This code was tested on `Ubuntu 20.04.4 LTS` and requires:
* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### Setup Conda Environment:
```
git clone https://github.com/orhir/STG-NF
cd STG-NF

# Conda environment setup
conda env create -f environment.yml
conda activate STG-NF
```

### Directory Structure
```
.
├── checkpoints
├── data
│   ├── ShanghaiTech
│   │   ├── gt
│   │   │   └── test_frame_mask
│   │   └── pose
│   │       ├── test
│   │       └── train
│   └── UBnormal
│       ├── gt
│       ├── pose
│       │   ├── abnormal_train
│       │   ├── test
│       │   ├── train
│       │   └── validation
│       └── videos
├── models
│   └── STG_NF
└── utils

```

### Data Directory
Data folder, including extracted poses and GT, can be downloaded using the [link](https://drive.google.com/file/d/1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt/view?usp=sharing).

The data directory holds pose graphs and ground truth vectors for the datasets.
A path for the directory may be configured using the arguments:

    --vid_path_train
    --vid_path_test
    --pose_path_train
    --pose_path_train_abnormal
    --pose_path_test

### Custom Dataset
We provide a script for creating JSON files in the accepted format using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).
Please download into pretrained_models folders [fast_421_res152_256x192.pth](https://drive.google.com/open?id=1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9)
It is also recommended to use the YOLOX-X detector, which can be downloaded from the AlphaPose repository.
Use the flag --video for video folder, otherwise assumes a folder of JPG/PNG images for each video.
    python gen_data.py --alphapose_dir /path/to/AlphaPoseFloder/ --dir /input/dir/ --outdir /output/dir/ [--video]

필요한 모델들 미리 다 다운받아야함. (위의 링크 참고)

## Pose extracting/tracking:
```
python3 gen_data.py --dir {데이터셋 폴더 경로} --outdir {결과 저장 경로} --alphapose_dir {AlphaPose clone해온 폴더 경로} --video
```
- AlphaPose 클론 후 실행 전에
  ```python setup.py build develop``` 필수

## ground truth 생성:
```
python3 gen_gt.py --train_label {train 라벨 폴더 경로} --test_label {test 라벨 폴더 경로}
```

## Training/Testing
Training and Evaluating is run using:
```
python train_eval.py --dataset [ShanghaiTech\UBnormal\AIHub]
```

Evaluation of our pretrained model can be done using:

ShanghaiTech/ShanghaiTech-HR:
```
python train_eval.py --dataset [ShanghaiTech/ShanghaiTech-HR] --checkpoint checkpoints/ShanghaiTech_85_9.tar
```
Unsupervised UBnormal
```
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar 
```
Supervised UBnormal
```
python train_eval.py --dataset UBnormal --seg_len 16 --R 10 --checkpoint checkpoints/UBnormal_supervised_79_2.tar
```
AIHub
```
python3 train_eval.py --dataset AIHub --checkpoint /data/exp_dir/AIHub/Apr11_1229/Apr11_1230__checkpoint.pth.tar
```

## Acknowledgments
Our code is based on code from:
- [Graph Embedded Pose Clustering for Anomaly Detection](https://github.com/amirmk89/gepc)
- [Glow](https://github.com/y0ast/Glow-PyTorch)

## License
This code is distributed under a [Creative Commons LICENSE](LICENSE).

Note that our code depends on other libraries and uses datasets that each have their own respective licenses that must also be followed.
