_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

model_names = ['rs_rcnn', 'atss', 'paa']

data = dict(test=dict(ann_file='calibration/data/calibration_test2017.json'))

# model settings
model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.00,
            nms=dict(type='nms', iou_threshold=0.65),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

score_vote = False