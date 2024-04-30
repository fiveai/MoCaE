_base_ = 'faster_rcnn_r50_fpn_straug_3x_coco.py'

# model settings
model = dict(
    roi_head=dict(
        type='ProbabilisticRoIHead',
        bbox_head=dict(
            type='ProbShared2FCBBoxHead',
            cov_type='full',
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0, reduction='sum'),
            loss_var=dict(type='SmoothL1Loss', loss_weight=1.0, reduction='sum'))))