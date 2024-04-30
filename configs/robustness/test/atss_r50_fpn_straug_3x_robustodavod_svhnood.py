_base_ = '../training/atss_r50_fpn_straug_3x_nuimages.py'

# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(
        type='CocoDataset',
        ann_file='data/svhn/svhn_ann_robust_od.json',
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
