_base_ = '../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'


# Datasets
data = dict(
    val=dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/'
        ),
    test=dict(
        # In-distribution
        ann_file= 'data/inaturalist/annotations/generalod_ood_inat_val.json',
        img_prefix='data/inaturalist/images/')

        # Natural Covariate Shift 
        # ann_file='data/OpenImages_OOD/coco_classes/COCO-Format/val_coco_format.json',
        # img_prefix='data/OpenImages_OOD/coco_classes/images/',

        # Out of Distribution 
        # ann_file='data/OpenImages_OOD/ood_classes/COCO-Format/val_coco_format.json',
        # img_prefix='data/OpenImages_OOD/ood_classes/images/')
        )


# Synthetic Covariate Shift
'''
Supported corruption keywords

'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
'brightness', 'contrast', 'elastic_transform', 'pixelate',
'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
'saturate'

'all', 'benchmark', 'noise', 'blur', 'weather', 'digital', 'holdout', 
'None', 'random_benchmark'
'''

syn_cov_sh = dict(corruptions = 'random_benchmark',
                  corr_per_image_per_sev = [1, 2, 0, 2, 0, 2],
                  severities = [0, 1, 2, 3, 4, 5])

# Entropy Type
model = dict(
    test_cfg=dict(
        rcnn=dict(
            uncertainty=dict(type = 'ds'))))