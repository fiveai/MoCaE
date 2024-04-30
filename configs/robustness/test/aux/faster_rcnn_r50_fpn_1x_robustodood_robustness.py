_base_ = '../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'


# Datasets
data = dict(
    workers_per_gpu=2,
    test=dict(
        # In-distribution
        ann_file= 'data/coco/annotations/general_od_ood_test.json',
        img_prefix='data/',
        samples_per_gpu=2),
    syn_cov_sh = dict(corruptions = 'benchmark',
                  severities = [0, 0, 0, 0, 0])
        )

# Entropy Type
model = dict(
    test_cfg=dict(
        rcnn=dict(
            uncertainty=dict(type = 'entropy'))))