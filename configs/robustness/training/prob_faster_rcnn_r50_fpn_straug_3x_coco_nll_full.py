_base_ = 'prob_faster_rcnn_r50_fpn_straug_3x_coco_nll_diagonal.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            cov_type='full')))