_base_ = 'base_ensemble.py'

ensemble_detections = ["calibration/rs_rcnn/final_detections/minitest.bbox.json",
						"calibration/atss/final_detections/minitest.bbox.json",
						"calibration/paa/final_detections/minitest.bbox.json"]

model_names = ['rs_rcnn', 'atss', 'paa']

calibration_type = 'IR'