_base_ = '../training/faster_rcnn_r50_fpn_straug_3x_nuimages.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='RemoveObjects', fill='constant'),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(samples_per_gpu=2,
              ann_file='data/robustod/annotations/av_od_val.json',
              img_prefix='data/nuimages/',
              pipeline=test_pipeline,
              test_mode=False,
              filter_empty_gt=False),
    test_time_modifications=dict(corruptions='benchmark',
                                 severities=[0])  # 0: the standard test w/o corruptions
    # >0: corruptions
    # -1: OOD mix up
)

# Uncertainties

# cls_type = {entropy, ds, avg_entropy, max_class_entropy}
# detection scores are kept in all cases
# loc_type = {spectral_entropy, determinant, trace}
model = dict(
    test_cfg=dict(
        rcnn=dict(uncertainty=dict(cls_type=['entropy', 'ds']))))
