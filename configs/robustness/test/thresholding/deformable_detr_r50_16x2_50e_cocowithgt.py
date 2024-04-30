_base_ = '../deformable_detr_r50_16x2_50e_coco.py'

# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(
        # In-distribution
        ann_file='data/coco/annotations/instances_withgt_val2017.json',
        img_prefix='data/coco/val2017/',
        samples_per_gpu=2),
    test_time_modifications=dict(corruptions='benchmark',
                                 severities=[0])  # 0: the standard test w/o corruptions
                                                           # >0: corruptions
                                                           # -1: OOD mix up
)
