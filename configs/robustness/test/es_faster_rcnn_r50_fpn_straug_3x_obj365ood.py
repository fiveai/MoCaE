_base_ = '../training/prob_faster_rcnn_r50_fpn_straug_3x_coco_es_diagonal_l1.py'


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

# cls_type = {entropy, ds, avg_entropy, max_class_entropy}
# detection scores are kept in all cases
# loc_type = {spectral_entropy, determinant, trace}
model = dict(
    test_cfg=dict(
        rcnn=dict(uncertainty=dict(cls_type=['entropy', 'ds'],
                                   loc_type=['determinant', 'spectral_entropy', 'trace']))))