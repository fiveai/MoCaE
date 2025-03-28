import json
import os
import math


class_map = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter']


with open('/data/dota/val/DOTA_1.0.json', 'r') as file: # your path to the file
    data = json.load(file)   
image_id_mapping = {image['file_name'].split('.')[0]: image['id'] for image in data['images']}

coco_format = []
# your path to the directory containing the .txt files, which can be obtained from the mmrotate by using --format-only --eval-options
# and here rmove "Task1_" in the .txt filename
directory = '/data/val_results/rtmdet_results' 
for file_name in os.listdir(directory):
    if file_name.endswith('.txt'):
        file_path = os.path.join(directory, file_name)
        base_name = os.path.splitext(file_name)[0]
        # print(file_path)
        with open(file_path, 'r') as file:
            # print(file)
            for line in file:
                parts = line.strip().split()
                # print(parts[0])
                image_id = image_id_mapping[parts[0]]
                score = float(parts[1])
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[2:10])
                xmin = min(x1, x2, x3, x4)
                xmax = max(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                ymax = max(y1, y2, y3, y4)
                width = xmax - xmin
                height = ymax - ymin
                coco_dict = {
                    "image_id": int(image_id),
                    "score": score,
                    # "bbox": [xmin, ymin, width, height],
                    "bbox": [x1, y1, x2, y2, x3, y3, x4, y4],
                    "category_id": class_map.index(base_name) + 1
                }
                coco_format.append(coco_dict)
           
det_directory = 'mocae/calibration/rtmdet/obb_final_detections/val.bbox.json'  # your path to save the output json file  
with open(det_directory, 'w') as output_file:
    json.dump(coco_format, output_file)
