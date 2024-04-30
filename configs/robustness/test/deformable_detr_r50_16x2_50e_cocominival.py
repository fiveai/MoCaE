_base_ = '../training/deformable_detr_r50_16x2_50e_coco.py'

# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(ann_file='data/coco/annotations/instances_robustness_val2017.json',
        samples_per_gpu=1),
    test_time_modifications=dict(corruptions='benchmark',
                                 severities=[0])  # 0: the standard test w/o corruptions
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
