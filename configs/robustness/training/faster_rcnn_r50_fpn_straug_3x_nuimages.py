_base_ = 'faster_rcnn_r50_fpn_straug_3x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3)))

data = dict(
    train=dict(type='RobustODAV',
    	ann_file='data/robustod/annotations/av_od_train.json',
    	img_prefix='data/nuimages/'),
    val=dict(type='RobustODAV',
    	ann_file='data/robustod/annotations/av_od_val.json',
    	img_prefix='data/nuimages/'),
    test=dict(type='RobustODAV',
    	ann_file='data/nuimages/annotations/nuimages_v1.0-val.json',
    	img_prefix='data/nuimages/'))