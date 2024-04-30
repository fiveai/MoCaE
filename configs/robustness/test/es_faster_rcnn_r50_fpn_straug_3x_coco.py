_base_ = '../training/prob_faster_rcnn_r50_fpn_straug_3x_coco_es_diagonal_l1.py'

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

# cls_type = {entropy, ds, avg_entropy, max_class_entropy}
# detection scores are kept in all cases
# loc_type = {spectral_entropy, determinant, trace}
model = dict(
    test_cfg=dict(
        rcnn=dict(uncertainty=dict(cls_type=['entropy', 'ds'],
                                   loc_type=['determinant', 'spectral_entropy', 'trace']))))
