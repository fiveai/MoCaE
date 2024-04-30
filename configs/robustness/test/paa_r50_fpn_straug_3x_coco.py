_base_ = '../../paa/paa_r50_fpn_mstrain_3x_coco.py'

model = dict(bbox_head=dict(score_voting=False))

# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(samples_per_gpu=2),
    test_time_modifications=dict(corruptions='benchmark',
                                 severities=[0, 1, 3, 5])  # 0: the standard test w/o corruptions
                                                           # >0: corruptions
                                                           # -1: OOD mix up
)

# Uncertainties

# cls_type = {none, entropy, ds, avg_entropy, max_class_entropy}
# detection scores are kept in all cases
# loc_type = {none, spectral_entropy, determinant, trace}
model = dict(
    test_cfg=dict(
        uncertainty=dict(cls_type=['entropy', 'ds', 'avg_entropy', 'max_class_entropy'])))

