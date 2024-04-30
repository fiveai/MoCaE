# MoCaE: Mixture of Calibrated Experts Significantly Improves Accuracy in Object Detection

The official implementation of "MoCaE: Mixture of Calibrated Experts Significantly Improves Accuracy in Object Detection".

> [**MoCaE: Mixture of Calibrated Experts Significantly Improves Accuracy in Object Detection**](https://arxiv.org/abs/2309.14976)            
> Kemal Oksuz, Selim Kuzucu, Tom Joy, Puneet K. Dokania


## Introduction

Combining the strengths of many existing predictors to obtain a Mixture of Experts which is superior to its individual components is an effective way to improve the performance without having to develop new architectures or train a model from scratch. With this repository, we aim to provide the means to construct an effective Mixture of Experts of object detectors through calibration.

Using this repository, you can 

- Reproduce single model performances
- Reproduce a Deep Ensemble (RS R-CNN x 5 with 43.4 AP), Vanilla MoE (with 43.4 AP) and our MoCaE (with 45.5 AP)
- Obtain calibrators 

## Specification of Dependencies and Preparation

- Please see [get_started.md](docs/en/get_started.md) for requirements and installation of mmdetection.
- Please download [this zip file](https://drive.google.com/file/d/10KizA1LWH8xdHKz5qDUmRL81wgMJx3wG/view?usp=sharing) and place it under the root. This provides inference results of the detectors on COCO mini-test.
- Please download the relevant annotations from this [Google Drive link](https://drive.google.com/drive/u/0/folders/1tYKERgVQhx0UkAfGRl7sEHFnNpYNOkBy)

## 1. Reproducing Single Models (Table 3 and Table 4)

### RS R-CNN
Please run the following command,

```
python tools/test_ensemble.py configs/calibration/single_models/rs_r50_fpn_straug_3x_minicoco.py --eval bbox
```
and obtain the following results:

('bbox_mAP', 0.424), ('bbox_mAP_50', 0.621), ('bbox_mAP_75', 0.462), ('bbox_mAP_s', 0.268), ('bbox_mAP_m', 0.463), ('bbox_mAP_l', 0.569)

### ATSS

Please run the following command,

```
python tools/test_ensemble.py configs/calibration/single_models/atss_r50_fpn_straug_3x_minicoco.py --eval bbox
```
and obtain the following results:

('bbox_mAP', 0.431), ('bbox_mAP_50', 0.615), ('bbox_mAP_75', 0.471), ('bbox_mAP_s', 0.278), ('bbox_mAP_m', 0.475), ('bbox_mAP_l', 0.542)

### PAA

Please run the following command,

```
python tools/test_ensemble.py configs/calibration/single_models/paa_r50_fpn_straug_3x_minicoco.py --eval bbox
```
and obtain the following results:

('bbox_mAP', 0.432), ('bbox_mAP_50', 0.608), ('bbox_mAP_75', 0.471), ('bbox_mAP_s', 0.27), ('bbox_mAP_m', 0.47), ('bbox_mAP_l', 0.576)

## 2. Reproducing the Results in Table 4

### Our MoCaE of 3 Detectors (Ours)

Please run the following command,

```
python tools/test_ensemble.py configs/calibration/rs_atss_paa_r50_fpn_straug_3x_minicoco_calibrated_refiningnms.py --eval bbox 
```
and obtain the following results:

('bbox_mAP', 0.455), ('bbox_mAP_50', 0.632), ('bbox_mAP_75', 0.5), ('bbox_mAP_s', 0.297), ('bbox_mAP_m', 0.497), ('bbox_mAP_l', 0.593)

### Vanilla MoE of RS R-CNN, ATSS, PAA (without calibration)
Please run the following command,

```
python tools/test_ensemble.py configs/calibration/rs_atss_paa_r50_fpn_straug_3x_minicoco_uncalibrated.py --eval bbox 
```
and obtain the following results:

('bbox_mAP', 0.434), ('bbox_mAP_50', 0.625), ('bbox_mAP_75', 0.471), ('bbox_mAP_s', 0.273), ('bbox_mAP_m', 0.473), ('bbox_mAP_l', 0.58)

### An Example Deep Ensemble (RS R-CNN x 5)
We share one of the deep ensembles to keep the size of the calibration.zip lower as a deep ensemble requires several models. To reproduce, RS R-CNN x 5, please run the following command,

```
python tools/test_ensemble.py configs/calibration/rs_rcnn5_r50_fpn_mstrain_3x_minicoco.py --eval bbox
```
and obtain the following results:

('bbox_mAP', 0.434), ('bbox_mAP_50', 0.63), ('bbox_mAP_75', 0.477), ('bbox_mAP_s', 0.28), ('bbox_mAP_m', 0.475), ('bbox_mAP_l', 0.57)

## 3. Calibrating the Detectors 

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

ACE= 0.2929792825104798

MCE= 0.45415657187482583
```

calibrated test set error:

```
ECE =  0.03191429508521918

ACE= 0.08949849505398023

MCE= 0.3724112771554677
```

Here, we obtain the calibrators with 500 images following how we obtain MoEs. Hence the results for calibrated test error very slightly differ from Table 7 reporting LaECE using 2.5K images for calibration. As an example, LaECE after calibration here is 3.19 instead of 3.15 in Table 7. Besides, the uncalibrated test error remains the same as 36.45.

## How to Cite

Please cite the paper if you benefit from our paper or the repository:
```
@misc{oksuz2024mocae,
      title={MoCaE: Mixture of Calibrated Experts Significantly Improves Object Detection}, 
      author={Kemal Oksuz and Selim Kuzucu and Tom Joy and Puneet K. Dokania},
      year={2024},
      eprint={2309.14976},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
