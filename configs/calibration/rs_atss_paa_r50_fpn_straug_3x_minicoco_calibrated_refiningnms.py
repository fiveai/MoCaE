_base_ = 'base_ensemble.py'

ensemble_detections = ["calibration/rs_rcnn/final_detections/minitest.bbox.json",
						"calibration/atss/final_detections/minitest.bbox.json",
						"calibration/paa/final_detections/minitest.bbox.json"]

model_names = ['rs_rcnn', 'atss', 'paa']

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.00,
            nms=dict(type='soft_nms'),
            max_per_img=100)))

score_vote = True

calibration_type = 'IR'