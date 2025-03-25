import os
from mmcv.ops import nms_quadri
import numpy as np
import torch 
import pickle 
import argparse
import ast

def obtain_moe_for_rotated(calibrate, iou_thr):
    model_names = ['rotated_rtmdet', 'lsk']
    from_scratch = True
    det_paths = []
    
    if calibrate:
        target_path = 'calibration/rotated_mocae/Task1/'
    else:
        target_path = 'calibration/rotated_vanilla_moe/Task1/'
        
    image_files = 'dota_test_images.npy'
        
    for model_name in model_names:
        det_paths.append('calibration/'+model_name+'/final_detections/Task1/')

    with open('calibration/data/'+image_files) as file:
        for line in file: 
            images = [l.strip('\'').strip(' \'') for l in line.split(',')]

    # Find all the txt files in an example results folder
    inputs = []
    for file in os.listdir(det_paths[0]):
        if file.endswith(".txt"):
            inputs.append(file)

    if calibrate:
        calibrator = []

        # Obtain the pre-fitted calibrators for each model
        for model_name in model_names:
            print(model_name)
            calibration_file = "calibration/" + model_name + "/calibrators/IR_class_agnostic_finaldets_ms.pkl"

            if os.path.exists(calibration_file):
                with open(calibration_file, 'rb') as f:
                    calibrator.append(pickle.load(f))
            else:
                print('no calibrator, need to implement identity')

    # Process all subtasks to obtain the combinations
    for fname in inputs:
        out_file_path = os.path.join(target_path, fname)
    
        if not from_scratch and os.path.exists(out_file):
            print('Already exists ', fname)
            continue
        out_file = open(out_file_path, 'w')
        print('Processing ', fname)
        all_detections = []
        num_dets = np.zeros([len(det_paths)+1])
        for j, det_path in enumerate(det_paths):
            # Using readlines()
            file = open(os.path.join(det_path, fname), 'r')
            Lines = file.readlines()
            num_dets[j+1] = len(Lines)
            file.close()
            all_detections.extend(Lines)

        img_ids = []
        bboxes = []
        scores = []

        for line in all_detections:
            split = line.split()

            # Get image id
            img_ids.append(split[0])

            # get score
            scores.append(float(split[1]))

            # Get bbox
            bboxes.append([float(split[2]), float(split[3]), float(split[4]), float(split[5]),
                                    float(split[6]), float(split[7]), float(split[8]), float(split[9])])
        all_img_ids = np.array(img_ids)
        all_scores = np.array(scores)
        all_bboxes = np.array(bboxes)

        if calibrate:
            num_dets = num_dets.cumsum()
            for j in range(len(det_paths)):
                init_idx = int(num_dets[j])
                end_idx = int(num_dets[j+1])
                temp = np.clip(calibrator[j][0].predict(all_scores[init_idx:end_idx].reshape(-1, 1)), 0, 1)
                all_scores[init_idx:end_idx] = temp

        for img in images:
            idx = (all_img_ids == img).nonzero()[0]
            if len(idx) == 0:
                continue
            temp_scores = all_scores[idx]
            temp_bboxes = all_bboxes[idx]

            nms_dets, idx = nms_quadri(torch.from_numpy(temp_bboxes).float(), torch.from_numpy(temp_scores).float(), iou_thr)

            qboxes, scores = torch.split(nms_dets, (8, 1), dim=-1)

            for qbox, score in zip(qboxes, scores):
                txt_element = [img, str(round(float(score), 2))
                           ] + [f'{p:.2f}' for p in qbox]
                out_file.writelines(' '.join(txt_element) + '\n')
        out_file.close()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate', default=True, type=ast.literal_eval, help='Whether to calibrate before combining')
    parser.add_argument('--iou_thr', help='IoU threshold for post-processing', default=0.35)
    
    args = parser.parse_args()

    calibrate = args.calibrate
    iou_thr = args.iou_thr
    
    obtain_moe_for_rotated(calibrate, iou_thr)
