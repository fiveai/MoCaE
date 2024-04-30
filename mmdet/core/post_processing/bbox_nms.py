# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms
from torchvision.ops.boxes import box_iou

from mmdet.core.bbox.iou_calculators import bbox_overlaps
import time
import numpy as np
def nms_op(bboxes, scores, idxs, iou_threshold, adaptive=False):
    '''

    thr = np.array([0.95164016, 0.62330024, 0.81254435, 0.78536419, 0.37861322, 0.30134115,
     0.68653207, 0.40848095, 0.49185754, 0.39846809, 0.16170525, 0.0873497,
     0.19686127, 0.59451186, 0.96589883, 0.40336517, 0.44888876, 0.68014123,
     0.61055271, 0.75141321, 0.57796502, 0.32502559, 0.78221739, 0.59335823,
     0.35517336, 0.69973779, 0.62187284, 0.73819503, 0.67374957, 0.32707269,
     0.67898807, 0.0282159 , 0.40659039, 0.45000432, 0.72906236, 0.02772125,
     0.28705251, 0.5647891 , 0.22654974, 0.65972738, 0.65611309, 0.51032197,
     0.65728038, 0.66076773, 0.55568704, 0.69174313, 0.48390494, 0.37010244,
     0.53167621, 0.43796463, 0.58900912, 0.71088026, 0.6147678 , 0.49900978,
     0.55800844, 0.44449992, 0.72516549, 0.46277947, 0.89407245, 0.32546595,
     0.51999035, 0.50062418, 0.9337909 , 0.74126504, 0.05983433, 0.42885012,
     0.24040819, 0.37056739, 0.        , 0.42495436, 0.        , 0.35137375,
     0.34987263, 0.84814419, 0.06225386, 0.2971844 , 0.47848104, 0.52552091,
     0.        , 0.71448834]) * 0.70

    thr = np.clip(thr, a_min=0, a_max=1)
    '''

    order = torch.argsort(-scores)
    indices = torch.arange(bboxes.shape[0]).cuda()
    keep = torch.ones_like(indices, dtype=torch.bool)
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > iou_threshold)
            #overlapped = torch.nonzero(iou > thr[idxs[order[i]]])
            keep[overlapped + i + 1] = 0
    return order[keep]

def batched_nms_(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    keep = nms_op(boxes_for_nms, scores, idxs, nms_cfg_['iou_threshold'])
    boxes = boxes[keep]
    scores = scores[keep]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False,
                   uncertainty=None,
                   nms_time=None,
                   cuda_nms=True):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr

    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        if score_factors.dim() < 2:
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if uncertainty is not None:
        if 'cls' in uncertainty.keys():
            for k, v in uncertainty['cls'].items():
                uncertainty['cls'][k] = v.view(-1, 1).expand(multi_scores.size(0), num_classes).reshape(-1)
        if 'loc' in uncertainty.keys():
            for k, v in uncertainty['loc'].items():
                uncertainty['loc'][k] = v.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        if uncertainty is not None:
            if 'cls' in uncertainty.keys():
                for k, v in uncertainty['cls'].items():
                    uncertainty['cls'][k] = v[inds]
            if 'loc' in uncertainty.keys():
                for k, v in uncertainty['loc'].items():
                    uncertainty['loc'][k] = v[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels, nms_time

    if cuda_nms and nms_cfg is not None:
        start_time = time.perf_counter()
        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    elif nms_cfg is not None:
        start_time = time.perf_counter()
        dets, keep = batched_nms_(bboxes, scores, labels, nms_cfg)
    else:
        keep = torch.argsort(scores, descending=True)
        dets = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)
        dets = dets[keep]


    #nms_time += time.perf_counter() - start_time


    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if uncertainty is not None:
        if 'cls' in uncertainty.keys():
            for k, v in uncertainty['cls'].items():
                dets = torch.cat([dets, v[keep].unsqueeze(dim=1)], dim=1)
        if 'loc' in uncertainty.keys():
            for k, v in uncertainty['loc'].items():
                dets = torch.cat([dets, v[keep].unsqueeze(dim=1)], dim=1)

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep], nms_time


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (dets, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Dets are boxes with scores.
            Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
