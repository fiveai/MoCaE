_base_ = '../training/faster_rcnn_r50_fpn_straug_3x_coco.py'

# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(ann_file='data/coco/annotations/instances_robustness_test2017.json',
        samples_per_gpu=1),
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
