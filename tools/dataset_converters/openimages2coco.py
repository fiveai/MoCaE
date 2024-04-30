import argparse
import csv
import cv2
import json
import os

from tqdm import tqdm

OPEN_IMAGES_TO_COCO = {'Person': 'person',
                       'Bicycle': 'bicycle',
                       'Car': 'car',
                       'Motorcycle': 'motorcycle',
                       'Airplane': 'airplane',
                       'Bus': 'bus',
                       'Train': 'train',
                       'Truck': 'truck',
                       'Boat': 'boat',
                       'Traffic light': 'traffic light',
                       'Fire hydrant': 'fire hydrant',
                       'Stop sign': 'stop sign',
                       'Parking meter': 'parking meter',
                       'Bench': 'bench',
                       'Bird': 'bird',
                       'Cat': 'cat',
                       'Dog': 'dog',
                       'Horse': 'horse',
                       'Sheep': 'sheep',
                       'Elephant': 'cow',
                       'Cattle': 'elephant',
                       'Bear': 'bear',
                       'Zebra': 'zebra',
                       'Giraffe': 'giraffe',
                       'Backpack': 'backpack',
                       'Umbrella': 'umbrella',
                       'Handbag': 'handbag',
                       'Tie': 'tie',
                       'Suitcase': 'suitcase',
                       'Flying disc': 'frisbee',
                       'Ski': 'skis',
                       'Snowboard': 'snowboard',
                       'Ball': 'sports ball',
                       'Kite': 'kite',
                       'Baseball bat': 'baseball bat',
                       'Baseball glove': 'baseball glove',
                       'Skateboard': 'skateboard',
                       'Surfboard': 'surfboard',
                       'Tennis racket': 'tennis racket',
                       'Bottle': 'bottle',
                       'Wine glass': 'wine glass',
                       'Coffee cup': 'cup',
                       'Fork': 'fork',
                       'Knife': 'knife',
                       'Spoon': 'spoon',
                       'Bowl': 'bowl',
                       'Banana': 'banana',
                       'Apple': 'apple',
                       'Sandwich': 'sandwich',
                       'Orange': 'orange',
                       'Broccoli': 'broccoli',
                       'Carrot': 'carrot',
                       'Hot dog': 'hot dog',
                       'Pizza': 'pizza',
                       'Doughnut': 'donut',
                       'Cake': 'cake',
                       'Chair': 'chair',
                       'Couch': 'couch',
                       'Houseplant': 'potted plant',
                       'Bed': 'bed',
                       'Table': 'dining table',
                       'Toilet': 'toilet',
                       'Television': 'tv',
                       'Laptop': 'laptop',
                       'Computer mouse': 'mouse',
                       'Remote control': 'remote',
                       'Computer keyboard': 'keyboard',
                       'Mobile phone': 'cell phone',
                       'Microwave oven': 'microwave',
                       'Oven': 'oven',
                       'Toaster': 'toaster',
                       'Sink': 'sink',
                       'Refrigerator': 'refrigerator',
                       'Book': 'book',
                       'Clock': 'clock',
                       'Vase': 'vase',
                       'Scissors': 'scissors',
                       'Teddy bear': 'teddy bear',
                       'Hair dryer': 'hair drier',
                       'Toothbrush': 'toothbrush'}


REMOVED_OOD_IMAGE_LIST = ['008838299bd147a4', '0096d83bed765696', '01ddedf26d3a6ef8', '032ea1e00897db72', '083f9fad0020b4f7', '084de6ad77c6fb0d', '093b3a277c0c4e4d', '0b720d3af8d92d6a', '1336e27052a51c9c', '17aec9329f538b94', '1aa50eaf4065a0ff', '1bd15490f7867026', '1c37405cc6c6dabe', '1c7e56d1f5a0a6a5', '1dfdf5beddae3f2b', '1e798dbf737b77b9', '1e8f0eddb38f4e89', '1ed32f0601a75cb8', '20030adfad998c72', '2097bdb2f8ef7860', '20d115b597f302b0', '266184acd6d0c59b', '2788df1e65718feb', '292e33ad70ca79e6', '2c26acebe1014492', '2d53431e4d33f93e', '3474688349a0ae7c', '350a12e1ce9ce267', '360239377194565e', '36e33129bef15dda', '39d61edaf10e5031', '3b127697f105b565', '3ba4cb0a5184527d', '3c5c86bbd830d7d7', '3c6fc8cd07130086', '3df7bbb849ab732d', '3fee67408ae70284', '423054422c8d04a9', '430b66cea3815884', '44ef8bb9e9c91ae2', '450d5b11ac09e36b', '4707d3632e2f664a', '472b401c8b970106', '48519c2e46d774d1', '494e7346bf4825b0', '496a97aae1d83b1a', '4a40365a23fe5d94', '4b5d85c3c6d244d9', '4eaf5475ddbd976c', '4faad76a341011e3', '5172de2fa1e30628', '51d2c658a64fefe5', '53d9de07d6a4b6b5', '54cd76cfe18735bf', '566423e1b091f806', '57722abc66efd5a8', '59115ad88e974bed', '5e9547c0908f7488', '5f7f7ab5546bbd8e', '60a3a338c2f9c835', '60d421c79decaeee', '614b45004e8278da', '63c93fafaf9d1eb7', '647d6e75aa5a14ae', '675387578c948ec5', '6ac9aee1c9372045', '6b4cdbf933c62f09', '6daed282f0f13165', '6f2bd5e2f8674337', '766b6f6e03f2cadf', '7dace5a8f4cdd464', '7fe7a9a0078e9028', '81b280849432f867', '832e7d4d793cfe7a', '84b4b4e8579f9f96', '84cdd35e28d706da', '88ba6afc3c3ba201', '8b3a044770d9603b', '8c12c2011969c85a', '8fe7bab9f21e8044', '8ff589e6dcac313a', '900b0e9e4efebc44', '90fd0817ea9eec1b', '915ff919c966b71c', '92f02f779568e31b', '967c9f738cd6688c', '9681a7609ec7f319', '97dd5b9ecd8a37dc', '992c02971efd667e', '99e2b9f1a1d4aaa6', '9c7d59173f379725', '9d7120dd4661d3bd', '9e16f1fe2dbf8e67', '9f8077085cbe79ee']

def main(args):
    dataset_dir = args.dataset_dir

    if args.output_dir is None:
        output_dir = os.path.expanduser(
            os.path.join(dataset_dir, 'COCO-Format'))
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Get category mapping from openimages symbol to openimages names.
    with open(os.path.expanduser(os.path.join(dataset_dir, 'class-descriptions-boxable.csv')), 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        openimages_class_mapping_dict = dict()
        for row in csv_f:
            openimages_class_mapping_dict.update({row[0]: row[1]})

    # Get mapping from openimages names to coco names
    open_images_to_coco_dict = OPEN_IMAGES_TO_COCO

    # Get annotation csv path and image directories
    annotations_csv_path = os.path.expanduser(
        os.path.join(dataset_dir, 'train-annotations-bbox.csv'))
    image_dir = os.path.expanduser(os.path.join(dataset_dir, 'images'))
    id_list = [image[:-4] for image in os.listdir(image_dir)]

    # Begin processing annotations
    with open(annotations_csv_path, 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)

        processed_ids = []
        images_list = []
        annotations_list = []
        count = 0
        with tqdm(total=len(id_list)) as pbar:
            for i, row in enumerate(csv_f):
                image_id = row[0]
                if image_id in id_list:
                    image = cv2.imread(
                        os.path.join(
                            image_dir,
                            image_id) + '.jpg')
                    width = image.shape[1]
                    height = image.shape[0]

                    category_symbol = row[2]
                    category_name = openimages_class_mapping_dict[category_symbol]

                    if image_id in REMOVED_OOD_IMAGE_LIST:
                        continue

                    if category_name in list(open_images_to_coco_dict.keys()):
                        mapped_category = open_images_to_coco_dict[category_name]
                        category_id = list(
                            open_images_to_coco_dict.values()).index(mapped_category) + 1

                        x_min = float(row[4]) * width
                        x_max = float(row[5]) * width

                        y_min = float(row[6]) * height
                        y_max = float(row[7]) * height

                        is_occluded = int(row[8])
                        is_truncated = int(row[9])

                        bbox_coco = [
                            x_min,
                            y_min,
                            x_max - x_min,
                            y_max - y_min]

                        annotations_list.append({'image_id': image_id,
                                                 'id': count,
                                                 'category_id': category_id,
                                                 'bbox': bbox_coco,
                                                 'area': bbox_coco[2] * bbox_coco[3],
                                                 'iscrowd': 0,
                                                 'is_truncated': is_truncated,
                                                 'is_occluded': is_occluded})
                        count += 1
                    else:
                        category_id = 81

                        x_min = float(row[4]) * width
                        x_max = float(row[5]) * width

                        y_min = float(row[6]) * height
                        y_max = float(row[7]) * height

                        is_occluded = int(row[8])
                        is_truncated = int(row[9])

                        bbox_coco = [
                            x_min,
                            y_min,
                            x_max - x_min,
                            y_max - y_min]

                        annotations_list.append({'image_id': image_id,
                                                 'id': count,
                                                 'category_id': category_id,
                                                 'bbox': bbox_coco,
                                                 'area': bbox_coco[2] * bbox_coco[3],
                                                 'iscrowd': 0,
                                                 'is_truncated': is_truncated,
                                                 'is_occluded': is_occluded})
                        count += 1

                    if image_id not in processed_ids:
                        pbar.update(1)
                        images_list.append({'id': image_id,
                                            'width': width,
                                            'height': height,
                                            'file_name': image_id + '.jpg',
                                            'license': 1})
                        processed_ids.append(image_id)
                    

                else:
                    continue

    licenses = [{'id': 1,
                 'name': 'none',
                 'url': 'none'}]

    categories = [
        {"supercategory": "person", "id": 1, "name": "person"},
        {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
        {"supercategory": "vehicle", "id": 3, "name": "car"},
        {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
        {"supercategory": "vehicle", "id": 5, "name": "airplane"},
        {"supercategory": "vehicle", "id": 6, "name": "bus"},
        {"supercategory": "vehicle", "id": 7, "name": "train"},
        {"supercategory": "vehicle", "id": 8, "name": "truck"},
        {"supercategory": "vehicle", "id": 9, "name": "boat"},
        {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
        {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
        {"supercategory": "outdoor", "id": 12, "name": "stop sign"},
        {"supercategory": "outdoor", "id": 13, "name": "parking meter"},
        {"supercategory": "outdoor", "id": 14, "name": "bench"},
        {"supercategory": "animal", "id": 15, "name": "bird"},
        {"supercategory": "animal", "id": 16, "name": "cat"},
        {"supercategory": "animal", "id": 17, "name": "dog"},
        {"supercategory": "animal", "id": 18, "name": "horse"},
        {"supercategory": "animal", "id": 19, "name": "sheep"},
        {"supercategory": "animal", "id": 20, "name": "cow"},
        {"supercategory": "animal", "id": 21, "name": "elephant"},
        {"supercategory": "animal", "id": 22, "name": "bear"},
        {"supercategory": "animal", "id": 23, "name": "zebra"},
        {"supercategory": "animal", "id": 24, "name": "giraffe"},
        {"supercategory": "accessory", "id": 25, "name": "backpack"},
        {"supercategory": "accessory", "id": 26, "name": "umbrella"},
        {"supercategory": "accessory", "id": 27, "name": "handbag"},
        {"supercategory": "accessory", "id": 28, "name": "tie"},
        {"supercategory": "accessory", "id": 29, "name": "suitcase"},
        {"supercategory": "sports", "id": 30, "name": "frisbee"},
        {"supercategory": "sports", "id": 31, "name": "skis"},
        {"supercategory": "sports", "id": 32, "name": "snowboard"},
        {"supercategory": "sports", "id": 33, "name": "sports ball"},
        {"supercategory": "sports", "id": 34, "name": "kite"},
        {"supercategory": "sports", "id": 35, "name": "baseball bat"},
        {"supercategory": "sports", "id": 36, "name": "baseball glove"},
        {"supercategory": "sports", "id": 37, "name": "skateboard"},
        {"supercategory": "sports", "id": 38, "name": "surfboard"},
        {"supercategory": "sports", "id": 39, "name": "tennis racket"},
        {"supercategory": "kitchen", "id": 40, "name": "bottle"},
        {"supercategory": "kitchen", "id": 41, "name": "wine glass"},
        {"supercategory": "kitchen", "id": 42, "name": "cup"},
        {"supercategory": "kitchen", "id": 43, "name": "fork"},
        {"supercategory": "kitchen", "id": 44, "name": "knife"},
        {"supercategory": "kitchen", "id": 45, "name": "spoon"},
        {"supercategory": "kitchen", "id": 46, "name": "bowl"},
        {"supercategory": "food", "id": 47, "name": "banana"},
        {"supercategory": "food", "id": 48, "name": "apple"},
        {"supercategory": "food", "id": 49, "name": "sandwich"},
        {"supercategory": "food", "id": 50, "name": "orange"},
        {"supercategory": "food", "id": 51, "name": "broccoli"},
        {"supercategory": "food", "id": 52, "name": "carrot"},
        {"supercategory": "food", "id": 53, "name": "hot dog"},
        {"supercategory": "food", "id": 54, "name": "pizza"},
        {"supercategory": "food", "id": 55, "name": "donut"},
        {"supercategory": "food", "id": 56, "name": "cake"},
        {"supercategory": "furniture", "id": 57, "name": "chair"},
        {"supercategory": "furniture", "id": 58, "name": "couch"},
        {"supercategory": "furniture", "id": 59, "name": "potted plant"},
        {"supercategory": "furniture", "id": 60, "name": "bed"},
        {"supercategory": "furniture", "id": 61, "name": "dining table"},
        {"supercategory": "furniture", "id": 62, "name": "toilet"},
        {"supercategory": "electronic", "id": 63, "name": "tv"},
        {"supercategory": "electronic", "id": 64, "name": "laptop"},
        {"supercategory": "electronic", "id": 65, "name": "mouse"},
        {"supercategory": "electronic", "id": 66, "name": "remote"},
        {"supercategory": "electronic", "id": 67, "name": "keyboard"},
        {"supercategory": "electronic", "id": 68, "name": "cell phone"},
        {"supercategory": "appliance", "id": 69, "name": "microwave"},
        {"supercategory": "appliance", "id": 70, "name": "oven"},
        {"supercategory": "appliance", "id": 71, "name": "toaster"},
        {"supercategory": "appliance", "id": 72, "name": "sink"},
        {"supercategory": "appliance", "id": 73, "name": "refrigerator"},
        {"supercategory": "indoor", "id": 74, "name": "book"},
        {"supercategory": "indoor", "id": 75, "name": "clock"},
        {"supercategory": "indoor", "id": 76, "name": "vase"},
        {"supercategory": "indoor", "id": 77, "name": "scissors"},
        {"supercategory": "indoor", "id": 78, "name": "teddy bear"},
        {"supercategory": "indoor", "id": 79, "name": "hair drier"},
        {"supercategory": "indoor", "id": 80, "name": "toothbrush"},
        {"supercategory": "indoor", "id": 81, "name": "OOD"}]

    json_dict_val = {'info': {'year': 2020},
                     'licenses': licenses,
                     'categories': categories,
                     'images': images_list,
                     'annotations': annotations_list}

    val_file_name = os.path.join(output_dir, 'val_coco_format.json')
    with open(val_file_name, 'w') as outfile:
        json.dump(json_dict_val, outfile)


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=str,
        help='OpenImages dataset directory')

    parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        help='converted dataset write directory')

    args = parser.parse_args()
    main(args)