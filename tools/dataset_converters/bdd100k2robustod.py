import os
import json
from os.path import exists

from tqdm import tqdm
import mmdet.datasets.cityscapes as city
from pycocotools.coco import COCO
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps

NUIMAGES_TO_BDD = {'pedestrian': ['pedestrian', "other person"],
                   'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'train', "trailer", "other vehicle"],
                   'bicycle': 'bicycle'}

FIX_CLASSES = {'rider': ['bicycle', 'motorcycle']}


def reverse_dict(NUIMAGES_TO_BDD):
    BDD_TO_NUIMAGES = {}
    for k, v in NUIMAGES_TO_BDD.items():
        if type(v) is list:
            for v_ in v:
                BDD_TO_NUIMAGES[v_] = k
        else:
            BDD_TO_NUIMAGES[v] = k
    return BDD_TO_NUIMAGES


def get_bboxes(anns, BDD100K_ID_TO_NAME, classes):
    boxes = list()
    for ann in anns:
        ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
        if ann_cat_name in classes:
            box = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
            boxes.append(box)

    return torch.from_numpy(np.array(boxes))


def coco2robustod(data_path, limit=-1):
    # 1. Read Data
    bdd100k = city.CocoDataset(data_path, [], test_mode=True)
    nuimages_path = 'data/nuimages/annotations/nuimages_v1.0-val.json'
    nuimages = COCO(nuimages_path)

    # 2. Initialize out data
    out_data = {'info': 'Natural Covariate Shift Split', 'licenses': ['please check out bdd100k dataset license'],
                'images': list(), 'annotations': list(), 'categories': nuimages.dataset['categories']}

    # 2. Initialize out data
    stats = {'weather': dict(), 'weather': dict(), 'timeofday': dict(), 'scene': dict()}

    BDD100K_TO_NUIMAGES = reverse_dict(NUIMAGES_TO_BDD)
    BDD100K_ID_TO_NAME = {}
    for cat in bdd100k.coco.dataset['categories']:
        BDD100K_ID_TO_NAME[cat['id']] = cat['name']

    NUIMAGES_NAME_TO_ID = {}
    for cat in nuimages.dataset['categories']:
        NUIMAGES_NAME_TO_ID[cat['name']] = cat['id']

    counter = 0
    no_gt_counter = 0
    min_iou_ctr = 0

    for i in range(len(bdd100k.coco.dataset['images'])):
        # [{'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': '', 'filename': 'images/v1/patch8/objects365_v1_00420917.jpg'}]
        img_id = bdd100k.coco.dataset['images'][i]['id']
        ann_ids = bdd100k.coco.get_ann_ids(img_ids=[img_id])
        anns = bdd100k.coco.load_anns(ann_ids)
        # Ex. ann: {'id': 51, 'iscrowd': 0, 'isfake': 0, 'area': 1584.6324365356109, 'isreflected': 0, 'bbox': [491.3955078011, 88.1856689664, 35.65588379410002, 44.442382796800004], 'image_id': 420917, 'category_id': 84}

        #if limit > 0 and bdd100k.coco.dataset['images'][i]['weather'] == 'clear':
        #    continue

        num_id, num_amb, num_fix = 0, 0, 0
        fix_ann = []
        for ann in anns:
            ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
            if ann_cat_name in BDD100K_TO_NUIMAGES.keys():
                num_id += 1
            elif ann_cat_name in FIX_CLASSES.keys():
                num_fix += 1
                fix_ann.append(ann)

        if num_fix > 0:
            rider_boxes = get_bboxes(fix_ann, BDD100K_ID_TO_NAME, classes=[*FIX_CLASSES])
            riding_boxes = get_bboxes(anns, BDD100K_ID_TO_NAME, classes=FIX_CLASSES['rider'])

            # There is a rider but there is not what he/she rides
            if riding_boxes.size(0) == 0:
                continue

            # Assign riders and riding
            ious = bbox_overlaps(rider_boxes, riding_boxes).numpy()
            riders, riding = linear_sum_assignment(-ious)
            RIDERS = {}
            for j in range(len(riders)):
                RIDERS[riding[j]] = riders[j]

            # Ignore if the iou between a rider and riding is below some threhold.
            # This is because there can be wrong assignment, different
            # riding items such as prams
            min_iou = 1.0
            for k, v in RIDERS.items():
                if ious[v, k] < min_iou:
                    min_iou = ious[v, k]

            if min_iou < 0.10:
                min_iou_ctr += 1
                continue

            # Combine rider and riding and assign to riding
            riding_ctr = 0
            for ann in anns:
                ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
                if ann_cat_name in FIX_CLASSES['rider']:
                    if riding_ctr in riding:
                        tl_x = min(rider_boxes[RIDERS[riding_ctr]][0], riding_boxes[riding_ctr][0])
                        tl_y = min(rider_boxes[RIDERS[riding_ctr]][1], riding_boxes[riding_ctr][1])
                        br_x = max(rider_boxes[RIDERS[riding_ctr]][2], riding_boxes[riding_ctr][2])
                        br_y = max(rider_boxes[RIDERS[riding_ctr]][3], riding_boxes[riding_ctr][3])
                        ann['bbox'] = [tl_x.item(), tl_y.item(), br_x.item() - tl_x.item(), br_y.item() - tl_y.item()]
                    riding_ctr += 1

        # Now generate dataset with corrected boxes
        if num_id > 0:
            # Add to Natural Covariate Shift
            counter += 1
            out_data['images'].append(bdd100k.coco.dataset['images'][i])
            for stat in stats.keys():
                if bdd100k.coco.dataset['images'][i][stat] in stats[stat].keys():
                    stats[stat][bdd100k.coco.dataset['images'][i][stat]] += 1
                else:
                    stats[stat][bdd100k.coco.dataset['images'][i][stat]] = 1

            for ann in anns:
                ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
                if ann_cat_name in BDD100K_TO_NUIMAGES.keys():
                    coco_ann_cat_name = BDD100K_TO_NUIMAGES[ann_cat_name]
                    ann['category_id'] = NUIMAGES_NAME_TO_ID[coco_ann_cat_name]
                    out_data['annotations'].append(ann)

            if limit > 0 and counter == limit:
                break
        else:
            no_gt_counter += 1

    out_file_path = data_path[:-5] + '_robust_od.json'

    for stat in stats.keys():
        import matplotlib.pyplot as plt
        plt.bar(list(stats[stat].keys()), stats[stat].values(), color='g')
        #plt.show()
    #with open(out_file_path, 'w') as outfile:
    #    json.dump(out_data, outfile)

    return out_file_path


def bdd2coco_detection(labeled_images, save_dir):
    attr_dict = {"categories":
        [
            {"supercategory": "none", "id": 1, "name": "pedestrian"},
            {"supercategory": "none", "id": 2, "name": "car"},
            {"supercategory": "none", "id": 3, "name": "rider"},
            {"supercategory": "none", "id": 4, "name": "bus"},
            {"supercategory": "none", "id": 5, "name": "truck"},
            {"supercategory": "none", "id": 6, "name": "bicycle"},
            {"supercategory": "none", "id": 7, "name": "motorcycle"},
            {"supercategory": "none", "id": 8, "name": "traffic light"},
            {"supercategory": "none", "id": 9, "name": "traffic sign"},
            {"supercategory": "none", "id": 10, "name": "train"},
            {"supercategory": "none", "id": 11, "name": "other person"},
            {"supercategory": "none", "id": 12, "name": "other vehicle"},
            {"supercategory": "none", "id": 13, "name": "trailer"}
        ]}

    id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    images = list()
    annotations = list()
    ignore_categories = set()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name']

        image['height'] = 720
        image['width'] = 1280

        image['weather'] = i['attributes']['weather']
        image['timeofday'] = i['attributes']['timeofday']
        image['scene'] = i['attributes']['scene']

        image['id'] = counter

        empty_image = True

        tmp = 0
        if 'labels' not in i.keys():
            continue

        for l in i['labels']:
            annotation = dict()
            if l['category'] in id_dict.keys():
                tmp = 1
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = l['box2d']['x1']
                y1 = l['box2d']['y1']
                x2 = l['box2d']['x2']
                y2 = l['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = id_dict[l['category']]
                annotation['ignore'] = 0
                annotation['id'] = l['id']
                annotations.append(annotation)
            else:
                ignore_categories.add(l['category'])

        if empty_image:
            print('empty image!')
            continue
        if tmp == 1:
            images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    #with open(save_dir, "w") as file:
    #    json.dump(attr_dict, file)


def convert():
    bdd_dir = 'data/bdd100k/'
    src_val_dir = bdd_dir + 'annotations/det_val.json'
    src_train_dir = bdd_dir + 'annotations/det_train.json'

    dst_val_dir = bdd_dir + 'annotations/det_val_coco.json'
    dst_train_dir = bdd_dir + 'annotations/det_train_coco.json'

    out_file_paths = []

    # create BDD training set detections in COCO format
    if not exists(dst_train_dir):
        print('Loading training set...')
        with open(src_train_dir) as f:
            train_labels = json.load(f)
        print('Converting training set to COCO format. See:' + dst_train_dir)
        bdd2coco_detection(train_labels, dst_train_dir)
    print('Mapping BDD100K classes to ID classes for training set. See:' + dst_train_dir[:-5] + '_robust_od.json')
    out_file_paths.append(coco2robustod(dst_train_dir, 35000))

    # create BDD validation set detections in COCO format
    if exists(dst_val_dir):
        print('Converting validation set to COCO format. See:' + dst_val_dir)
        with open(src_val_dir) as f:
            val_labels = json.load(f)
        print('Converting validation set...')
        bdd2coco_detection(val_labels, dst_val_dir)
    print('Mapping BDD100K classes to ID classes for validation set. See:' + dst_val_dir[:-5] + '_robust_od.json')
    out_file_paths.append(coco2robustod(dst_val_dir))


    return out_file_paths
