# MoCaE: Mixture of Calibrated Experts Significantly Improves Accuracy in Object Detection

[![TMLR](https://img.shields.io/badge/arXiv-2309.14976-b31b1b.svg)](https://openreview.net/pdf?id=fJEsas1z8J) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocae-mixture-of-calibrated-experts/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=mocae-mixture-of-calibrated-experts) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocae-mixture-of-calibrated-experts/oriented-object-detection-on-dota-1-0)](https://paperswithcode.com/sota/oriented-object-detection-on-dota-1-0?p=mocae-mixture-of-calibrated-experts) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocae-mixture-of-calibrated-experts/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=mocae-mixture-of-calibrated-experts)


The official implementation of "MoCaE: Mixture of Calibrated Experts Significantly Improves Accuracy in Object Detection". This work is accepted to TMLR.

> [**MoCaE: Mixture of Calibrated Experts Significantly Improves Accuracy in Object Detection**](https://arxiv.org/abs/2309.14976)            
> Kemal Oksuz, Selim Kuzucu, Tom Joy, Puneet K. Dokania

## How to Cite

Please cite the paper if you benefit from our paper or the repository:
```
@article{
oksuz2024mocae,
title={MoCaE: Mixture of Calibrated Experts Significantly Improves Object Detection},
author={Kemal Oksuz and Selim Kuzucu and Tom Joy and Puneet K. Dokania},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=fJEsas1z8J},
note={}
}
```

## Introduction

Combining the strengths of many existing predictors to obtain a Mixture of Experts which is superior to its individual components is an effective way to improve the performance without having to develop new architectures or train a model from scratch. With this repository, we aim to provide the means to construct an effective Mixture of Experts of object detectors through calibration.

Using this repository, you can 

- Reproduce single model performances for object detection,
- Reproduce a Deep Ensemble (RS R-CNN x 5 with 43.4 AP), Vanilla MoE (with 43.4 AP) and MoCaE (with 45.5 AP),
- Obtain calibrators for object detectors, and
- Reproduce our DOTA result with MoCaE which is currently the state-of-the-art.

Coming Soon:
- Obtain calibrators for rotated object detectors
  
## Specification of Dependencies and Preparation

- Please see [get_started.md](docs/en/get_started.md) for requirements and installation of mmdetection.
- Please download [this zip file](https://drive.google.com/file/d/1wtr6lcB8n3yeeuYrONe6nInkTMo2DYWy/view?usp=sharing) and place it under the root. This provides inference results of the detectors on COCO mini-test and DOTA test set, the trained calibrators and annotations for both settings (object detection and rotated object detection).
  
## Reproducing Object Detection Results in Table 3

- For single models, please run the following command for RS R-CNN, ATSS and PAA respectively:

```
python tools/test_ensemble.py configs/calibration/single_models/rs_r50_fpn_straug_3x_minicoco.py --eval bbox
python tools/test_ensemble.py configs/calibration/single_models/atss_r50_fpn_straug_3x_minicoco.py --eval bbox
python tools/test_ensemble.py configs/calibration/single_models/paa_r50_fpn_straug_3x_minicoco.py --eval bbox
```
and obtain the following results:

('bbox_mAP', 0.424), ('bbox_mAP_50', 0.621), ('bbox_mAP_75', 0.463), ('bbox_mAP_s', 0.268), ('bbox_mAP_m', 0.463), ('bbox_mAP_l', 0.569)

('bbox_mAP', 0.431), ('bbox_mAP_50', 0.615), ('bbox_mAP_75', 0.471), ('bbox_mAP_s', 0.279), ('bbox_mAP_m', 0.475), ('bbox_mAP_l', 0.543)

('bbox_mAP', 0.433), ('bbox_mAP_50', 0.608), ('bbox_mAP_75', 0.471), ('bbox_mAP_s', 0.27.0), ('bbox_mAP_m', 0.47.0), ('bbox_mAP_l', 0.576)

- To obtain MoCaE of these 3 Detectors, please run the following command,
```
python tools/test_ensemble.py configs/calibration/rs_atss_paa_r50_fpn_straug_3x_minicoco_calibrated_refiningnms.py --eval bbox 
```
and obtain the following result:

('bbox_mAP', 0.455), ('bbox_mAP_50', 0.632), ('bbox_mAP_75', 0.5), ('bbox_mAP_s', 0.297), ('bbox_mAP_m', 0.498), ('bbox_mAP_l', 0.593)

- To obtain Vanilla MoE of RS R-CNN, ATSS, PAA (without calibration), please run the following command,
```
python tools/test_ensemble.py configs/calibration/rs_atss_paa_r50_fpn_straug_3x_minicoco_uncalibrated.py --eval bbox 
```
and obtain the following result:

('bbox_mAP', 0.434), ('bbox_mAP_50', 0.625), ('bbox_mAP_75', 0.471), ('bbox_mAP_s', 0.273), ('bbox_mAP_m', 0.473), ('bbox_mAP_l', 0.58.1)

- Finally, to obtain an Example Deep Ensemble (RS R-CNN x 5), we share one of the deep ensembles to keep the size of the calibration.zip lower as a deep ensemble requires several models. To reproduce, RS R-CNN x 5, please run the following command,
```
python tools/test_ensemble.py configs/calibration/rs_rcnn5_r50_fpn_mstrain_3x_minicoco.py --eval bbox
```
and obtain the following result:

('bbox_mAP', 0.434), ('bbox_mAP_50', 0.63), ('bbox_mAP_75', 0.477), ('bbox_mAP_s', 0.28), ('bbox_mAP_m', 0.475), ('bbox_mAP_l', 0.57)

## Calibrating Object Detectors 

Please run the following command,

```
python tools/analysis_tools/model_calibration.py model_name
```
where model_name can be any directory name in the calibration directory that you downloaded and unzipped. As an example, if you run 
```
python tools/analysis_tools/model_calibration.py rs_rcnn
```
you will find the calibrator under calibration/rs_rcnn/calibrators directory and get the following results:

uncalibrated test set error:

```
ECE =  0.3645087488831893
ACE = 0.2929792825104798
MCE = 0.45415657187482583
```

calibrated test set error:

```
ECE = 0.03191429508521918
ACE = 0.08949849505398023
MCE = 0.3724112771554677
```

## Reproducing Rotated Object Detection Results in Table 9

- We provide inference results of the [LSKNet](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html) obtained from the [official LSKNet GitHub repository](https://github.com/zcablii/LSKNet) in ``` calibration/lsk/final_detections/test/Task1 ```. You can submit compress this directory and submit to the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html) and obtain $\mathrm{AP}_{50} = 81.85$.

- We also provide inference results of the [RTMDet](https://arxiv.org/abs/2212.07784) obtained from the [mmrotate](https://github.com/open-mmlab/mmrotate/tree/1.x/configs/rotated_rtmdet) in ``` calibration/rotated_rtm/final_detections/test/Task1 ```. Similarly, you can submit compress this directory and submit to the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html) and obtain $\mathrm{AP}_{50} = 81.32$.


- To generate the **Vanilla MoE** detections for the test set, please run:
```
python tools/mocae_rotated_bounding_box.py --calibrate False
```
Then, zip all of the generated .txt files under ``` ./work_dirs/vanilla_moe/Task1/ ``` and submit the generated zip file to the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html) to obtain $\mathrm{AP}_{50} = 80.60$.

- To generate the **state-of-the-art MoCaE** detections for the test set, please run:
```
python tools/mocae_rotated_bounding_box.py --calibrate True
```
Then, zip all of the generated .txt files under ``` ./work_dirs/mocae/Task1/ ``` and submit the generated zip file to the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html) to obtain $\mathrm{AP}_{50} = 82.62$.


### Calibrating Rotated Object Detectors 

This is not yet supported. However, we provide the validation set inference files for both of the detectors as well as the list of validation images. We would appreciate a PR for this feature.
