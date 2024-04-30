_base_ = 'rs_faster_rcnn_r50_fpn_straug_3x_coco.py'

model = dict(
    roi_head=dict(
        type='RankBasedProbabilisticRoIHead',
        bbox_head=dict(
            type='ProbRankBasedShared2FCBBoxHead',
            loss_bbox=dict(type='IoULoss', reduction='none'),
            loss_var=dict(type='NLLLoss', loss_weight=1.0, reduction='mean'))))