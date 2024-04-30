_base_ = 'base_ensemble.py'

ensemble_detections = ["calibration/rs_rcnn/final_detections/minitest.bbox.json",
                        "calibration/rs_rcnn_2/final_detections/minitest.bbox.json",
                        "calibration/rs_rcnn_3/final_detections/minitest.bbox.json",
                        "calibration/rs_rcnn_4/final_detections/minitest.bbox.json",
                        "calibration/rs_rcnn_5/final_detections/minitest.bbox.json"]

model_names = ['rs_rcnn', 'rs_rcnn_2', 'rs_rcnn_3', 'rs_rcnn_4', 'rs_rcnn_5']

calibration_type = 'IR'
