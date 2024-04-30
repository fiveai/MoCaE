from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import argparse

import numpy as np
from pycocotools.coco import COCO
import os
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
import pickle
import random
from operator import itemgetter 
import json

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

set_all_seeds(0)

def assign_post(ann_dict, det_bboxes, det_score, det_label, dataset_classes, min_iou=0.5, max_iou=0.7):
    num_classes = len(dataset_classes)
    ious = np.zeros([det_bboxes.shape[0]])
    ## Assign
    for k, v in ann_dict.items():
        # Convert to numpy and reshape
        gt_boxes = np.array(v).reshape(-1, 4)

        # Convert to TL, BR representation
        gt_boxes[:, 2] += gt_boxes[:, 0]
        gt_boxes[:, 3] += gt_boxes[:, 1]

        rel_idx = (det_label==k).nonzero()[0]

        ious_cl = (bbox_overlaps(torch.from_numpy(gt_boxes), torch.from_numpy(det_bboxes[rel_idx]))).numpy()

        ious[rel_idx] = np.max(ious_cl, axis=0)

    return ious

def get_ann(cocoGt, ann_ids, dataset_classes):
    anns = cocoGt.loadAnns(ann_ids)
    ann_dict = {}
    for ann in anns:
        key = dataset_classes.index(ann['category_id'])
        if key not in ann_dict:
            ann_dict[key] = list(ann['bbox'])
        else:
            ann_dict[key].extend(ann['bbox'])
    return ann_dict


def create_calibration_dataset(cocoGt, model_detections, filename, dataset_classes, num_images=500):
    all_detections = []
    num_classes = len(dataset_classes)
    breakpoint()
    
    if num_images > 0 and num_images < 2500:
        idx = np.random.choice(range(len(cocoGt.dataset['images'])), size=num_images, replace=False)
        print('sampled image indices:', idx)
        images = itemgetter(*idx)(cocoGt.dataset['images'])
    else:
        print('using all val set images')
        images = cocoGt.dataset['images']


    f = open(model_detections)
    final_dets = json.load(f)
    print('detections are loaded')


    for i, img in enumerate(images):
        if 'counter' in img:
            counter = img['counter']
        else:
            counter = i

        # Get detections for this image
        detections = [det for det in final_dets if det['image_id']==img['id']]
        
        det_score = np.array([det['score'] for det in detections])
        det_bboxes = np.array([det['bbox'] for det in detections])
        det_label = np.array([dataset_classes.index(det['category_id']) for det in detections])

        if det_bboxes.ndim < 2:
            continue

        # Convert to TL, BR representation
        det_bboxes[:, 2] += det_bboxes[:, 0]
        det_bboxes[:, 3] += det_bboxes[:, 1]

        # Get ground truth bounding boxes
        # ann_ids = cocoGt.getAnnIds(imgIds=img['id'])

        # Get ground truth bounding boxes
        ann_ids = cocoGt.getAnnIds(imgIds=img['id'], iscrowd=False)

        ann_dict = get_ann(cocoGt, ann_ids, dataset_classes)
        
        ious = assign_post(ann_dict, det_bboxes, det_score, det_label, dataset_classes)

        detections = np.concatenate((np.expand_dims(det_score, axis=1), np.expand_dims(ious, axis=1), np.expand_dims(det_label, axis=1)), axis=1)


        all_detections.append(detections)

    dets = np.vstack(all_detections)

    np.save(filename, dets)

    return dets


def calibration_error(predicted_ious, det_ious, bin_count=25, num_cl=80):
    bins = np.linspace(0., 1., bin_count + 1)
    errors = np.zeros([num_cl, bin_count])
    weights_per_bin = np.zeros([num_cl, bin_count])

    total_cls_iou = np.zeros([num_cl])

    for cl in range(num_cl):      
        rel_idx = (det_ious[:, 2]==cl).nonzero()[0]
        predicted_ious_cls = predicted_ious[rel_idx]
        det_ious_cls = det_ious[rel_idx, 1]

        total_det = len(predicted_ious_cls)
        total_cls_iou[cl] = total_det

        for i in range(bin_count):
            # Find detections in this bin
            bin_idxs = np.logical_and(bins[i] <= predicted_ious_cls, predicted_ious_cls < bins[i + 1])
            bin_pred_ious_cls = predicted_ious_cls[bin_idxs]
            bin_det_ious_cls = det_ious_cls[bin_idxs]

            num_det = len(bin_pred_ious_cls)

            if num_det == 0:
                errors[cl, i] = np.nan
                weights_per_bin[cl, i] = 0
            else:
                # Average of Scores in this bin
                mean_pred = bin_pred_ious_cls.mean()
                mean_det = bin_det_ious_cls.mean()

                errors[cl, i] = np.abs(mean_pred - mean_det)

                # Weight of the bin
                weights_per_bin[cl, i] = num_det / total_det

    ECE_OD = np.nanmean(np.nansum(weights_per_bin * errors, axis=1))
    ACE_OD = np.nanmean(np.nanmean(errors, axis=1))
    MCE_OD = np.nanmean(np.nanmax(errors, axis=1))
    print('ECE = ', ECE_OD)
    print('ACE=', ACE_OD)
    print('MCE=', MCE_OD)

def get_calibration_data(cocoGt, model_detections, filename_val, dataset_classes, num_images=-1):
    if not os.path.exists(filename_val):
        print('Creating dataset...')
        dets = create_calibration_dataset(cocoGt, model_detections, filename_val, dataset_classes, num_images)
    else:
        print('Reading dataset...')
        dets = np.load(filename_val)
    return dets


def train_calibrator(coco, dets, dataset_classes, calibration_file, type, class_agnostic):
    calibrator = dict()
    if class_agnostic:
        det_scores = dets[:, 0].reshape(-1, 1)
        det_ious = dets[:, 1].reshape(-1)
        if type == 'IR':
            shared_calibrator = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip').fit(det_scores, det_ious)
        elif type == 'LR':
            shared_calibrator = LinearRegression().fit(det_scores, det_ious)

        for cls in range(len(dataset_classes)):
            calibrator[cls] = shared_calibrator

    else:
        for cls in range(len(dataset_classes)):
            idx = (dets[:,2] == cls).nonzero()[0] 
            det_scores = dets[idx, 0].reshape(-1, 1)
            det_ious = dets[idx, 1].reshape(-1)
            

            if type == 'IR':
                calibrator[cls] = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip').fit(det_scores.reshape(-1, 1), det_ious)
            elif type == 'LR':
                calibrator[cls] = LinearRegression().fit(det_scores.reshape(-1, 1), det_ious)

    with open(calibration_file, 'wb') as f:
         pickle.dump(calibrator, f)

    return calibrator


def predict_prob(calibrator, dets, dataset_classes):
    predicted_ious = np.zeros(dets.shape[0])
    for cls in range(len(dataset_classes)):
        idx = (dets[:, 2] == cls).nonzero()[0]
        if len(idx) == 0:
            continue
        det_scores = dets[idx, 0]
        predicted_ious[idx] = np.clip(calibrator[cls].predict(det_scores.reshape(-1, 1)), 0, 1)
    return predicted_ious


def get_calibrator(val_file, calibration_file, model_detections, calibration_type,
                   class_agnostic=False, num_images=-1):
    # Get Validation Dataset
    cocoGt = COCO(val_file)
    dataset_classes = list(cocoGt.cats.keys())
    dets = get_calibration_data(cocoGt, model_detections, filename_val, dataset_classes, num_images)

    # Learn Calibration Model
    print('Fitting calibrator...')
    calibrator = train_calibrator(cocoGt, dets, dataset_classes, calibration_file, calibration_type,
                                  class_agnostic)

    return calibrator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('model_name', help='Model Name to Calibrate)')
    args = parser.parse_args()

    model_name = args.model_name

    calibration_type = 'IR'
    class_agnostic_calibration = True
    num_images = 500

    val_file = 'calibration/data/calibration_val2017.json'
    test_file = 'calibration/data/calibration_test2017.json'
    model_detections = "calibration/" + model_name + "/final_detections/val.bbox.json"
    model_detections_test = "calibration/" + model_name + "/final_detections/val.bbox.json"
    filename_val = "calibration/" + model_name + "/final_detections/" + 'all_val500.npy'
    filename_test = "calibration/" + model_name + "/final_detections/" + 'all_test.npy'

    if class_agnostic_calibration:
        calibration_file = "calibration/" + model_name + "/calibrators/" + calibration_type + '_class_agnostic_finaldets500.pkl'
    else:
        calibration_file = "calibration/" + model_name + "/calibrators/" + calibration_type + '_class_wise_finaldets500.pkl'


    calibrator = get_calibrator(val_file, calibration_file, model_detections, calibration_type, class_agnostic=class_agnostic_calibration,
                                num_images=num_images)

    # Get Test Dataset
    cocoGt = COCO(test_file)
    dataset_classes = list(cocoGt.cats.keys())
    num_classes = len(dataset_classes)

    dets = get_calibration_data(cocoGt, model_detections_test, filename_test, dataset_classes)
    # Uncalibrated Test Error
    print("uncalibrated test set error:")
    # Measure Error
    calibration_error(dets[:, 0], dets, num_cl=num_classes)

    # Get calibrated probabilities on test set
    predicted_ious_test = predict_prob(calibrator, dets, dataset_classes)

    # Measure Error
    print("calibrated test set error:")
    calibration_error(predicted_ious_test, dets, num_cl=num_classes)
