_base_ = '../training/deformable_detr_r50_16x2_50e_coco.py'

# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(
        ann_file='data/objects365/annotations/zhiyuan_objv2_trainval_robust_od_ood.json',
        img_prefix='data/',
        samples_per_gpu=2),
    test_time_modifications=dict(corruptions = 'benchmark',
                           severities = [0]) # only 0 implies the test w/o corruptions
        )

# Uncertainties

# cls_type = {none, entropy, ds, avg_entropy, max_class_entropy}
# detection scores are kept in all cases
# loc_type = {none, spectral_entropy, determinant, trace}
model = dict(
    test_cfg=dict(
        uncertainty=dict(cls_type=['entropy', 'ds', 'avg_entropy', 'max_class_entropy'])))