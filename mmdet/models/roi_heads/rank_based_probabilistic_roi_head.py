import torch
import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .rank_based_standard_roi_head import RankBasedStandardRoIHead

@HEADS.register_module()
class RankBasedProbabilisticRoIHead(RankBasedStandardRoIHead):

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred, bbox_var = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_var=bbox_var, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox, bbox_weights = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], bbox_results['bbox_var'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(bbox_weights=bbox_weights)
        return bbox_results
