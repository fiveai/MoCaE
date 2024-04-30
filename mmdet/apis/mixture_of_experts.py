# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import prepare_mask_results, multiclass_nms, bbox2result
import pycocotools.mask as mask_util
import numpy as np
import pickle, os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import json

def find_det_idx(det_bboxes, det_labels, bboxes, scores, num_dets, detector_idx=True):
    det_idx = np.zeros(det_bboxes.shape[0]) - 1
    for i, (box, label) in enumerate(zip(det_bboxes, det_labels)):
        # Find box
        box_idxs = (torch.isclose(bboxes, box[:4]).sum(axis=1) > 3).nonzero()
        flag = False

        for box_idx in box_idxs:
            if torch.allclose(box[-1], scores[box_idx.item(), label]):
                flag = True
                if detector_idx:
                    for j, num_det in enumerate(num_dets.cumsum()):
                        if box_idx < num_det:
                            det_idx[i] = j
                            break
                else:
                    det_idx[i] = box_idx.item()

            if flag:
                break
    return det_idx


def score_voting(det_bboxes, det_labels, mlvl_bboxes,
                 mlvl_nms_scores, score_thr, sigma=0.025, num_classes=80):
    """Implementation of score voting method works on each remaining boxes
    after NMS procedure.

    Args:
        det_bboxes (Tensor): Remaining boxes after NMS procedure,
            with shape (k, 5), each dimension means
            (x1, y1, x2, y2, score).
        det_labels (Tensor): The label of remaining boxes, with shape
            (k, 1),Labels are 0-based.
        mlvl_bboxes (Tensor): All boxes before the NMS procedure,
            with shape (num_anchors,4).
        mlvl_nms_scores (Tensor): The scores of all boxes which is used
            in the NMS procedure, with shape (num_anchors, num_class)
        score_thr (float): The score threshold of bboxes.

    Returns:
        tuple: Usually returns a tuple containing voting results.

            - det_bboxes_voted (Tensor): Remaining boxes after
                score voting procedure, with shape (k, 5), each
                dimension means (x1, y1, x2, y2, score).
            - det_labels_voted (Tensor): Label of remaining bboxes
                after voting, with shape (num_anchors,).
    """
    candidate_mask = mlvl_nms_scores > score_thr
    candidate_mask_nonzeros = candidate_mask.nonzero(as_tuple=False)
    candidate_inds = candidate_mask_nonzeros[:, 0]
    candidate_labels = candidate_mask_nonzeros[:, 1]
    candidate_bboxes = mlvl_bboxes[candidate_inds]
    candidate_scores = mlvl_nms_scores[candidate_mask]
    det_bboxes_voted = []
    det_labels_voted = []
    for cls in range(num_classes):
        candidate_cls_mask = candidate_labels == cls
        if not candidate_cls_mask.any():
            continue
        candidate_cls_scores = candidate_scores[candidate_cls_mask]
        candidate_cls_bboxes = candidate_bboxes[candidate_cls_mask]
        det_cls_mask = det_labels == cls
        det_cls_bboxes = det_bboxes[det_cls_mask].view(
            -1, det_bboxes.size(-1))
        det_candidate_ious = bbox_overlaps(det_cls_bboxes[:, :4],
                                           candidate_cls_bboxes)
        for det_ind in range(len(det_cls_bboxes)):
            single_det_ious = det_candidate_ious[det_ind]
            pos_ious_mask = single_det_ious > 0.01
            pos_ious = single_det_ious[pos_ious_mask]
            pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
            pos_scores = candidate_cls_scores[pos_ious_mask]
            pis = (torch.exp(-(1 - pos_ious)**2 / sigma) *
                   pos_scores)[:, None]
            voted_box = torch.sum(
                pis * pos_bboxes, dim=0) / torch.sum(
                    pis, dim=0)
            voted_score = det_cls_bboxes[det_ind][-1:][None, :]
            det_bboxes_voted.append(
                torch.cat((voted_box[None, :], voted_score), dim=1))
            det_labels_voted.append(cls)

    det_bboxes_voted = torch.cat(det_bboxes_voted, dim=0)
    det_labels_voted = det_labels.new_tensor(det_labels_voted)
    return det_bboxes_voted, det_labels_voted

def predict_prob(calibrator, dets, num_classes):
    predicted_ious = np.zeros(dets.shape[0])
    for cls in range(num_classes):
        idx = (dets[:, 2] == cls).nonzero()[0]
        if len(idx) == 0:
            continue
        det_scores = dets[idx, 0]
        predicted_ious[idx] = np.clip(calibrator[cls].predict(det_scores.reshape(-1, 1)), 0, 1)
    return predicted_ious

def preprocess_detections(detections):
    num_detectors = len(detections)
    final_dets = [dict() for x in range(num_detectors)]

    for det in range(num_detectors):
        for pred in detections[det]: 
            key = pred['image_id']
            if key not in final_dets[det]:
                final_dets[det][key] = list()
            final_dets[det][key].append(pred)


    return final_dets

    


def mixture_of_experts(
                    data_loader,
                    cfg,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.50,
                    dim=5,
                    online=False):
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))


    # Model Details
    model_names = cfg['model_names']
    ensemble_detections = cfg['ensemble_detections']

    score_vote = cfg['score_vote']
    sigma = 0.04

    max_per_img = cfg['model']['test_cfg']['rcnn']['max_per_img']
    score_thr = cfg['model']['test_cfg']['rcnn']['score_thr']
    nms_cfg = cfg['model']['test_cfg']['rcnn']['nms']

    # Calibration Details
    calibration_type = cfg['calibration_type']
    calibrator = []
    for model_name in model_names:
        calibration_file = "calibration/" + model_name + "/calibrators/IR_class_agnostic_finaldets500.pkl"
        if os.path.exists(calibration_file):
            with open(calibration_file, 'rb') as f:
                calibrator.append(pickle.load(f))

    cocoGt = COCO(cfg['data']['test']['ann_file'])
    final_dets = []
    dataset_classes = list(cocoGt.cats.keys())
    num_classes = len(dataset_classes)

    for j, dir in enumerate(ensemble_detections):
        print('Reading final detections ', str(j))
        f = open(dir)
        final_dets.append(json.load(f))
        f.close()

    final_dets = preprocess_detections(final_dets)

    print('Final detections have been read')



    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_list = []
            num_dets = np.zeros(len(model_names))
            img_id = cocoGt.dataset['images'][i]['id']
            for j, dir in enumerate(ensemble_detections):   
                if j == 0:
                    
                    #detections = [det for det in final_dets[j] if det['image_id']==img_id]
                    if img_id in final_dets[j]:
                        detections = final_dets[j][img_id]
                    else:
                        bboxes = torch.zeros(0, 4).cuda().type(torch.cuda.FloatTensor)
                        scores = torch.zeros(0, num_classes+1).cuda().type(torch.cuda.FloatTensor)                            
                        continue    
                    bboxes = np.array([det['bbox'] for det in detections])
                    scores_ = np.array([det['score'] for det in detections])

                    
                    cats = [det['category_id'] for det in detections]
                    for i, cat in enumerate(cats):
                        if cat not in dataset_classes:
                            scores_[i] = 0.
                            detections[i]['category_id'] = 1
                            
                    det_labels = np.array([dataset_classes.index(det['category_id']) for det in detections])

                    if bboxes.ndim < 2:
                        bboxes = torch.zeros(0, 4).cuda().type(torch.cuda.FloatTensor)
                        scores = torch.zeros(0, num_classes+1).cuda().type(torch.cuda.FloatTensor)
                        if len(ensemble_detections) == 1:
                            det_bboxes = torch.zeros(0, 5).cuda().type(torch.cuda.FloatTensor)
                            det_labels = torch.zeros(0).cuda().type(torch.cuda.FloatTensor)
                        continue

                    # Convert to TL, BR representation
                    bboxes[:, 2] += bboxes[:, 0]
                    bboxes[:, 3] += bboxes[:, 1]

                    if calibration_type == 'IR' or calibration_type == 'LR':
                        score_ = np.zeros([bboxes.shape[0], 3])
                        score_[:,0] = scores_
                        score_[:,2] = det_labels
                        scores_ = predict_prob(calibrator[j], score_, num_classes)

                    num_dets[j] = bboxes.shape[0]

                    if len(ensemble_detections) > 1:
                        bboxes = torch.from_numpy(bboxes).cuda().type(torch.cuda.FloatTensor)
                        scores = torch.zeros(bboxes.shape[0], num_classes+1).cuda().type(torch.cuda.FloatTensor)
                        scores[range(bboxes.shape[0]), det_labels] = torch.from_numpy(scores_).cuda().type(torch.cuda.FloatTensor)
                    else: # Test single model, no need for NMS
                        det_bboxes = torch.from_numpy(bboxes).cuda().type(torch.cuda.FloatTensor)
                        scores = torch.from_numpy(scores_).cuda().type(torch.cuda.FloatTensor)
                        det_bboxes = torch.concat((det_bboxes, scores.unsqueeze(1)), dim=1)
                        det_labels = torch.from_numpy(det_labels).cuda().type(torch.cuda.FloatTensor)

                else:
                    #detections = [det for det in final_dets[j] if det['image_id']==img_id]
                    if img_id in final_dets[j]:
                        detections = final_dets[j][img_id]
                    else:
                        continue
                    bboxes_ = np.array([det['bbox'] for det in detections])
                    scores_ = np.array([det['score'] for det in detections])
                    det_labels = np.array([dataset_classes.index(det['category_id']) for det in detections])

                    if bboxes_.ndim < 2:
                        continue

                    # Convert to TL, BR representation
                    bboxes_[:, 2] += bboxes_[:, 0]
                    bboxes_[:, 3] += bboxes_[:, 1]

                    if calibration_type == 'IR' or calibration_type == 'LR':
                        score_ = np.zeros([bboxes_.shape[0], 3])
                        score_[:,0] = scores_
                        score_[:,2] = det_labels
                        scores_ = predict_prob(calibrator[j], score_, num_classes)
                    
                    num_dets[j] = bboxes_.shape[0]

                    temp_bboxes = torch.from_numpy(bboxes_).cuda().type(torch.cuda.FloatTensor)
                    temp_scores = torch.zeros(bboxes_.shape[0], num_classes+1).cuda().type(torch.cuda.FloatTensor)
                    temp_scores[range(bboxes_.shape[0]), det_labels] = torch.from_numpy(scores_).cuda().type(torch.cuda.FloatTensor)

                    scores = torch.cat((scores, temp_scores))
        
                    bboxes = torch.cat((bboxes, temp_bboxes))

            if len(ensemble_detections) > 1:
                det_bboxes, det_labels, nms_time = multiclass_nms(bboxes, scores, score_thr, nms_cfg, max_per_img)
                if score_vote:
                    det_bboxes, det_labels = score_voting(det_bboxes, det_labels, bboxes, scores, score_thr, sigma, num_classes=num_classes)

            raw_result = (det_bboxes, det_labels)
            
            result_list.append(raw_result)

            result = [
                bbox2result(det_bboxes, det_labels, num_classes, dim)
                for det_bboxes, det_labels in result_list
            ]

        batch_size = len(result)

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
