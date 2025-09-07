## GMSNet: Lightweight-Accurate Matching via Hierarchical Feature Optimization

## GMSNet implementation：
  
  Pytorch implementation of GMSNet


## Installation
You can create a new environment with the following command if you're using conda as your virtual environment:
```bash
git clone https://github.com/wjf/GMSNet.git
cd GMSNet
conda create -n GMSNet python=3.9
conda activate GMSNet
pip install -r requirements.txt
```


## Training:
To train GMSNet as described in the paper, you will need MegaDepth dataset and COCO_20k subset of COCO2017 dataset. As mentioned in the paper *[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/abs/2404.19174)*, you can obtain the full COCO2017 training data at https://cocodataset.org/.
However, we [make available](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) a subset of COCO for convenience. We simply selected a subset of 20k images according to image resolution. Please check COCO [terms of use](https://cocodataset.org/#termsofuse) before using the data.

To reproduce the training setup from the paper, please follow the steps:
1. Download [COCO_20k](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) containing a subset of COCO2017;
2. Download MegaDepth dataset. You can follow [LoFTR instructions](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets), we use the same standard as LoFTR. Then put the megadepth indices inside the MegaDepth root folder following the standard below:
```bash
{megadepth_root_path}/train_data/megadepth_indices #indices
{megadepth_root_path}/MegaDepth_v1 #images & depth maps & poses
```
3. Finally you can call training
```bash
python -m modules.training.train --training_type GMSNet_default --megadepth_root_path <path_to>/MegaDepth --synthetic_root_path <path_to>/coco_20k --ckpt_save_path /path/to/ckpts --num_workers 8
```


## Evaluation：
All evaluation code are in *evaluation*, you can download **MegaDepth** and **ScanNet** test dataset following [LoFTR](https://github.com/zju3dv/LoFTR/tree/master).
 
**Download and process MegaDepth-1500**  
We provide download link to [megadepth_test_1500](https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc)
```bash
tar xvf <path to megadepth_test_1500.tar>

cd <GMSNet>/data

ln -s <path to megadepth_test_1500> ./megadepth_test_1500
```
Then, you can call the mega1500 eval script, it should take a couple of minutes:
```bash
python -m evaluation.megadepth1500 --dataset-dir </data/Mega1500> --matcher GMSNet --ransac-thr 2.5
```


**Download and process ScanNet-1500**  
We provide download link to [scannet_test_1500](https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc)
```bash
tar xvf <path to scannet_test_1500.tar>

cd <GMSNet>/data

ln -s <path to scannet_test_1500> ./scannet_test_1500
```
Then, you can call the ScanNet1500 eval script, it should take a couple of minutes:
```bash
python -m evaluation.scannet1500 --scannet_path </data/ScanNet1500> --output </data/ScanNet1500/output>
python -m evaluation.scannet1500 --scannet_path </data/ScanNet1500> --output </data/ScanNet1500/output> --show
```


<!--
## Citing GMSNet
If you find the GMSNet code useful, please consider citing
```
@article{,
  title={GMSNet: Lightweight-Accurate Matching via Hierarchical Feature Optimization},
  author={},
  journal={},
  year={2025},
  publisher={IEEE}
}
```
-->

