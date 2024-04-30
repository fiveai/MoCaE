_base_ = 'atss_r50_fpn_straug_3x_coco.py'
model = dict(
    type='ATSS',
    bbox_head=dict(
        type='ProbATSSHead',
        loss_bbox=dict(type='IoULoss', loss_weight=2.0, reduction='none'),
        loss_var=dict(type='NLLLoss', loss_weight=1.0, reduction='none')))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

log_config= dict(interval=1)