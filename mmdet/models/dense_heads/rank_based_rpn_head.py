# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmdet.core import vectorize_labels, bbox_overlaps
import numpy as np

from ..builder import HEADS
from .rpn_head import RPNHead

from mmdet.models.losses import ranking_losses

@HEADS.register_module()
class RankBasedRPNHead(RPNHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605
    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 head_weight=0.20,
                 **kwargs):
        super(RankBasedRPNHead, self).__init__(
            in_channels, **kwargs)
        self.head_weight = head_weight
        self.loss_rank = ranking_losses.RankSort()

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        all_labels=[]
        all_label_weights=[]
        all_cls_scores=[]
        all_bbox_targets=[]
        all_bbox_weights=[]
        all_bbox_preds=[]
        for labels, label_weights, cls_score, bbox_targets, bbox_weights, bbox_pred in zip(labels_list, label_weights_list,cls_scores, bbox_targets_list, bbox_weights_list, bbox_preds):
            all_labels.append(labels.reshape(-1))
            all_label_weights.append(label_weights.reshape(-1))
            all_cls_scores.append(cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))
            
            all_bbox_targets.append(bbox_targets.reshape(-1, 4))
            all_bbox_weights.append(bbox_weights.reshape(-1, 4))
            all_bbox_preds.append(bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4))

        cls_labels = torch.cat(all_labels)
        all_scores=torch.cat(all_cls_scores)
        pos_idx = (cls_labels < self.num_classes)
        #flatten_anchors = torch.cat([torch.cat(item, 0) for item in anchor_list])
        if pos_idx.sum() > 0:
            # regression loss
            pos_pred = self.delta2bbox(torch.cat(all_bbox_preds)[pos_idx])
            pos_target = self.delta2bbox(torch.cat(all_bbox_targets)[pos_idx])
            loss_bbox = self.loss_bbox(pos_pred, pos_target)

            # flat_labels = self.flatten_labels(cls_labels, torch.cat(all_label_weights))
            flat_labels = vectorize_labels(cls_labels, self.num_classes, torch.cat(all_label_weights))
            flat_preds = all_scores.reshape(-1)
            pos_weights = all_scores.detach().sigmoid().max(dim=1)[0][pos_idx]

            bbox_avg_factor = torch.sum(pos_weights)
            if bbox_avg_factor < 1e-10:
                bbox_avg_factor = 1

            loss_bbox = torch.sum(pos_weights*loss_bbox)/bbox_avg_factor

            IoU_targets = bbox_overlaps(pos_pred.detach(), pos_target, is_aligned=True)
            flat_labels[flat_labels==1] = torch.clamp(IoU_targets + 1e-8, max=1)
            ranking_loss, sorting_loss = self.loss_rank.apply(flat_preds, flat_labels)

            self.SB_weight = (ranking_loss+sorting_loss).detach()/float(loss_bbox.item())
            loss_bbox *= self.SB_weight

            return dict(loss_rpn_rank=self.head_weight*ranking_loss, loss_rpn_sort=self.head_weight*sorting_loss, loss_rpn_bbox=self.head_weight*loss_bbox)

        else:
            losses_bbox=torch.cat(all_bbox_preds).sum()*0+1
            ranking_loss = all_scores.sum()*0+1
            sorting_loss = all_scores.sum()*0+1
            return dict(loss_rpn_rank=self.head_weight*ranking_loss, loss_rpn_sort=self.head_weight*sorting_loss, loss_rpn_bbox=self.head_weight*losses_bbox)

    def delta2bbox(self, deltas, means=[0., 0., 0., 0.], stds=[0.1, 0.1, 0.2, 0.2], max_shape=None, wh_ratio_clip=16/1000):

        wx, wy, ww, wh = stds
        dx = deltas[:, 0] * wx
        dy = deltas[:, 1] * wy
        dw = deltas[:, 2] * ww
        dh = deltas[:, 3] * wh
        
        max_ratio = np.abs(np.log(wh_ratio_clip))

        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = torch.exp(dw)
        pred_h = torch.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

