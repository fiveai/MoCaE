# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import distributions

from mmcv.cnn import ConvModule

from mmdet.core import vectorize_labels, bbox_overlaps, multiclass_nms, Uncertainty
from mmcv.runner import force_fp32
from mmdet.models.losses import ranking_losses

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
import torch.nn.functional as F
import numpy as np
import os
import pdb
import pickle


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # filename = self.features_folder + str(self.log_file_counter) + ".npy"
        # np.save(filename, x.cpu().numpy())


        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        #filename = self.logits_folder + str(self.log_file_counter) + ".npy"
        #np.save(filename, cls_score.cpu().numpy())
        #self.log_file_counter += 1

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class ProbShared2FCBBoxHead(Shared2FCBBoxHead):

    def __init__(self,
                 loss_var=dict(type='NLLLoss', loss_weight=1.0),
                 cov_type='diagonal',
                 var_predictor_cfg=dict(type='Linear'),
                 with_var=True,
                 *args,
                 **kwargs):
        super(ProbShared2FCBBoxHead, self).__init__(*args, **kwargs)

        self.cov_type = cov_type

        if self.cov_type == 'diagonal':
            self.single_out_dim_var = 4
        elif self.cov_type == 'full':
            self.single_out_dim_var = 10

        out_dim_var = (self.single_out_dim_var if self.reg_class_agnostic else
                       self.single_out_dim_var * self.num_classes)

        self.var_predictor_cfg = var_predictor_cfg
        self.with_var = with_var

        self.bbox_cov_num_samples = 1000

        self.fc_var = build_linear_layer(
            self.var_predictor_cfg,
            in_features=self.reg_last_dim,
            out_features=out_dim_var)

        self.loss_var = build_loss(loss_var)

        self.init_cfg += [
            dict(
                type='Normal', std=0.0001, override=dict(name='fc_var'))
        ]

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_var = self.fc_var(x_reg) if self.with_var else None
        return cls_score, bbox_pred, bbox_var

    def covariance_output_to_cholesky(self, pred_bbox_cov):
        """
        Taken from probdet repository

        Transforms output to covariance cholesky decomposition.
        Args:
            pred_bbox_cov (kx4 or kx10): Output covariance matrix elements.

        Returns:
            predicted_cov_cholesky (kx4x4): cholesky factor matrix
        """
        # Embed diagonal variance
        diag_vars = torch.sqrt(torch.exp(pred_bbox_cov[:, 0:4]))
        predicted_cov_cholesky = torch.diag_embed(diag_vars)

        if pred_bbox_cov.shape[1] > 4:
            tril_indices = torch.tril_indices(row=4, col=4, offset=-1)
            predicted_cov_cholesky[:, tril_indices[0],
            tril_indices[1]] = pred_bbox_cov[:, 4:]

        return predicted_cov_cholesky

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_var'))
    def loss(self,
             cls_score,
             bbox_pred,
             bbox_var,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):

        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)

            # There is a bug here and nan targets may be obtained very rarely,
            # need to debug properly, currently ignoring the corresponding predictions.
            nan_gt = torch.isnan(bbox_targets)
            if nan_gt.any():
                valid_target = torch.logical_not(torch.max(nan_gt, dim=1)[0])
                pos_inds = torch.logical_and(valid_target, pos_inds)

            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # NOT SUPPORTED FOR PROBABILISTIC FASTER R-CNN

                    # When the regression loss (e.g. `IoULoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]

                    pos_bbox_var = bbox_var.view(
                        bbox_var.size(0), self.single_out_dim_var)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]

                    pos_bbox_var = bbox_var.view(
                        bbox_var.size(0), -1,
                        self.single_out_dim_var)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]

                if type(self.loss_var).__name__ == 'NLLLoss':  # NLL Loss
                    if self.cov_type == 'diagonal':
                        losses['loss_bbox'] = self.loss_bbox(
                            pos_bbox_pred,
                            bbox_targets[pos_inds.type(torch.bool)],
                            bbox_weights[pos_inds.type(torch.bool)],
                            # avg_factor=bbox_targets.size(0),
                            reduction_override=reduction_override)

                        pos_bbox_var = torch.clamp(pos_bbox_var, -7, 7)

                        losses['loss_var'] = self.loss_var(
                            pos_bbox_var,
                            weight=bbox_weights[pos_inds.type(torch.bool)],
                            avg_factor=bbox_targets.size(0),
                            reduction_override=reduction_override)
                        losses['loss_bbox'] *= (0.50 * torch.exp(-pos_bbox_var))

                    elif self.cov_type == 'full':
                        # Multivariate Gaussian Negative Log Likelihood loss using pytorch
                        # distributions.multivariate_normal.log_prob()
                        forecaster_cholesky = self.covariance_output_to_cholesky(pos_bbox_var)

                        multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                            pos_bbox_pred, scale_tril=forecaster_cholesky)

                        losses['loss_bbox'] = - \
                            multivariate_normal_dists.log_prob(bbox_targets[pos_inds.type(torch.bool)])
                    losses['loss_bbox'] = losses['loss_bbox'].mean()

                elif type(self.loss_var).__name__ == 'L1Loss' or type(
                        self.loss_var).__name__ == 'SmoothL1Loss':  # Energy Score
                    forecaster_cholesky = self.covariance_output_to_cholesky(pos_bbox_var)

                    # Define per-anchor Distributions
                    multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                        pos_bbox_pred, scale_tril=forecaster_cholesky)
                    # Define Monte-Carlo Samples
                    distributions_samples = multivariate_normal_dists.rsample((self.bbox_cov_num_samples + 1,))

                    distributions_samples_1 = distributions_samples[0:self.bbox_cov_num_samples, :, :]
                    distributions_samples_2 = distributions_samples[1:self.bbox_cov_num_samples + 1, :, :]

                    # Compute energy score
                    losses['loss_var'] = -self.loss_var(
                        distributions_samples_1,
                        distributions_samples_2) / self.bbox_cov_num_samples

                    gt_proposals_delta_samples = torch.repeat_interleave(
                        bbox_targets[pos_inds.type(torch.bool)].unsqueeze(0), self.bbox_cov_num_samples, dim=0)

                    losses['loss_bbox'] = 2.0 * self.loss_bbox(
                        distributions_samples_1,
                        gt_proposals_delta_samples) / self.bbox_cov_num_samples  # First term

                    losses['loss_var'] /= bbox_targets.size(0)
                    losses['loss_bbox'] /= bbox_targets.size(0)

            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
                if self.cov_type == 'diagonal':
                    losses['loss_var'] = bbox_var[pos_inds].sum()

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_var'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   bbox_var,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if bbox_var is not None:
            # Convert log_variance to variance
            bbox_var = torch.exp(bbox_var).view(bbox_var.size(0), -1, self.single_out_dim_var)

        if "uncertainty" in cfg:
            uncertainty = Uncertainty.get_uncertainties(Uncertainty, cfg.uncertainty, logits=cls_score,
                                                        covariance=bbox_var)
        else:
            uncertainty = None

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, uncertainty=uncertainty)

            return det_bboxes, det_labels


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RankBasedShared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RankBasedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        self.loss_rank = ranking_losses.RankSort()
        self.log_folder = "calibration/rs_rcnn/detections/val/"
        calibration_file = "calibration/rs_rcnn/calibrators/IR_class_agnostic500.pkl"

        if os.path.exists(calibration_file):
            with open(calibration_file, 'rb') as f:
                self.calibrator = pickle.load(f)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        flat_labels = vectorize_labels(labels, self.num_classes, label_weights)
        flat_preds = cls_score.reshape(-1)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                pos_target = bbox_targets[pos_inds]
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds, labels[pos_inds]]

                loss_bbox = self.loss_bbox(pos_bbox_pred, pos_target)

                bbox_weights = cls_score.detach().sigmoid().max(dim=1)[0][pos_inds]

                IoU_targets = bbox_overlaps(pos_bbox_pred.detach(), pos_target, is_aligned=True)
                flat_labels[flat_labels == 1] = torch.clamp(IoU_targets + 1e-8, max=1)

                ranking_loss, sorting_loss = self.loss_rank.apply(flat_preds, flat_labels)

                bbox_avg_factor = torch.sum(bbox_weights)
                if bbox_avg_factor < 1e-10:
                    bbox_avg_factor = 1

                losses_bbox = torch.sum(bbox_weights * loss_bbox) / bbox_avg_factor
                self.SB_weight = (ranking_loss + sorting_loss).detach() / float(losses_bbox.item())
                losses_bbox *= self.SB_weight
                return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss,
                            loss_roi_bbox=losses_bbox), bbox_weights

            else:
                losses_bbox = bbox_pred.sum() * 0 + 1
                ranking_loss = cls_score.sum() * 0 + 1
                sorting_loss = cls_score.sum() * 0 + 1
                return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss,
                            loss_roi_bbox=losses_bbox), bbox_weights

        else:
            losses_bbox = bbox_pred.sum() * 0 + 1
            ranking_loss = cls_score.sum() * 0 + 1
            sorting_loss = cls_score.sum() * 0 + 1
            return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss, loss_roi_bbox=losses_bbox), bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        scores = cls_score.sigmoid() if cls_score is not None else None

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if "uncertainty" in cfg:
            uncertainty = Uncertainty.get_uncertainties(Uncertainty, cfg.uncertainty, logits=cls_score)
        else:
            uncertainty = None

        '''

        
        filename = self.log_folder + str(self.log_file_counter) + ".npy"
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "a") as f:
            x = np.concatenate((scores[:, :-1].cpu().numpy(), bboxes.cpu().numpy()), axis=1)
            np.save(filename, x)
        self.log_file_counter += 1
        '''
        

        #s_ = self.calibrator[0].predict(scores[:, :-1].cpu().numpy().reshape(-1, 1))
        #scores[:, :-1] = torch.from_numpy(s_.reshape(-1, self.num_classes)).cuda().type(torch.cuda.FloatTensor)



        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, nms_time = multiclass_nms(bboxes, scores,
                                                              cfg.score_thr, cfg.nms,
                                                              cfg.max_per_img, uncertainty=uncertainty,
                                                              nms_time=self.nms_time)
            self.nms_time = nms_time
            #print(self.nms_time)

            return det_bboxes, det_labels


@HEADS.register_module()
class ProbRankBasedShared2FCBBoxHead(RankBasedShared2FCBBoxHead):

    def __init__(self,
                 with_var=True,
                 loss_var=dict(type='NLLLoss', loss_weight=1.0),
                 var_predictor_cfg=dict(type='Linear'),
                 *args,
                 **kwargs):
        super(ProbRankBasedShared2FCBBoxHead, self).__init__(*args, **kwargs)

        # Currently only support for diagonal covariance
        self.single_out_dim_var = (1 if self.reg_decoded_bbox else 4)

        out_dim_var = (self.single_out_dim_var if self.reg_class_agnostic else
                       self.single_out_dim_var * self.num_classes)
        self.var_predictor_cfg = var_predictor_cfg
        self.with_var = with_var

        self.fc_var = build_linear_layer(
            self.var_predictor_cfg,
            in_features=self.reg_last_dim,
            out_features=out_dim_var)

        self.loss_var = build_loss(loss_var)

        self.init_cfg += [
            dict(
                type='Normal', std=0.0001, override=dict(name='fc_var'))
        ]

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_var = self.fc_var(x_reg) if self.with_var else None
        return cls_score, bbox_pred, bbox_var

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_var'))
    def loss(self,
             cls_score,
             bbox_pred,
             bbox_var,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        flat_labels = vectorize_labels(labels, self.num_classes, label_weights)
        flat_preds = cls_score.reshape(-1)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.

            # There is a bug here and nan targets may be obtained very rarely,
            # need to debug properly, currently ignoring the corresponding predictions.
            nan_gt = torch.isnan(bbox_targets)
            if nan_gt.any():
                valid_target = torch.logical_not(torch.max(nan_gt, dim=1)[0])
                pos_inds = torch.logical_and(valid_target, pos_inds)

            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                pos_target = bbox_targets[pos_inds]
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]

                    pos_bbox_var = bbox_var.view(
                        bbox_var.size(0), self.single_out_dim_var)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]

                    pos_bbox_var = bbox_var.view(
                        bbox_var.size(0), -1,
                        self.single_out_dim_var)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]

                loss_bbox = self.loss_bbox(pos_bbox_pred, pos_target)

                bbox_weights = cls_score.detach().sigmoid().max(dim=1)[0][pos_inds]

                IoU_targets = bbox_overlaps(pos_bbox_pred.detach(), pos_target, is_aligned=True)
                flat_labels[flat_labels == 1] = torch.clamp(IoU_targets + 1e-8, max=1)

                ranking_loss, sorting_loss = self.loss_rank.apply(flat_preds, flat_labels)

                bbox_avg_factor = torch.sum(bbox_weights)
                if bbox_avg_factor < 1e-10:
                    bbox_avg_factor = 1

                losses_bbox = (bbox_weights * loss_bbox) / bbox_avg_factor
                pos_bbox_var = torch.clamp(pos_bbox_var, -7, 7)

                losses_var = self.loss_var(pos_bbox_var, avg_factor=pos_bbox_var.size(0),
                                           reduction_override=reduction_override)

                if self.reg_decoded_bbox:
                    losses_bbox *= (0.50 * torch.exp(-pos_bbox_var)).squeeze(dim=1)
                else:
                    losses_bbox *= (0.50 * torch.exp(-pos_bbox_var))

                losses_bbox = torch.sum(losses_bbox)

                self.SB_weight = (ranking_loss + sorting_loss).detach() / float(losses_bbox.item())
                losses_bbox *= self.SB_weight

                return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss, loss_roi_bbox=losses_bbox,
                            loss_var=losses_var), bbox_weights

            else:
                losses_bbox = bbox_pred.sum() * 0 + 1
                ranking_loss = cls_score.sum() * 0 + 1
                sorting_loss = cls_score.sum() * 0 + 1
                losses_var = bbox_var.sum() * 0 + 1
                return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss, loss_roi_bbox=losses_bbox,
                            loss_var=losses_var), bbox_weights

        else:
            losses_bbox = bbox_pred.sum() * 0 + 1
            ranking_loss = cls_score.sum() * 0 + 1
            sorting_loss = cls_score.sum() * 0 + 1
            losses_var = bbox_var.sum() * 0 + 1
            return dict(loss_roi_rank=ranking_loss, loss_roi_sort=sorting_loss, loss_roi_bbox=losses_bbox,
                        loss_var=losses_var), bbox_weights
