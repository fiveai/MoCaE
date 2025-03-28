# please change CocoConvert.py to dotadevkit.py in your virtual environment

# --------------------------------------------------------
# Modified by Ashwin Nair
# Written by Jian Ding for DOTA_Devkit
# --------------------------------------------------------

# common path: /anaconda3/envs/obbdetection/lib/python3.x/site-packages/dotadevkit/ops/CocoConvert.py

import cv2
import json

from dotadevkit.misc.dota_utils import dota_classes, parse_dota_poly2
from pathlib import Path


def DOTA2COCO(srcpath, destfile, version="1.0"):
    imageparent = srcpath / "images"
    labelparent = srcpath / "labelTxt"
    assert version in ["1.0", "1.5", "2.0"]

    if version == "1.5":
        dota_classes.append("container-crane")

    if version == "2.0":
        dota_classes.extend(["container-crane", "airport", "helipad"])

    data_dict = {}
    info = {
        "contributor": "Captain Group, Wuhan University",
        "data_created": "2018",
        "description": f"DOTA dataset version {version}",
        "url": "https://captain-whu.github.io/DOTA/dataset.html",
        "version": version,
        "year": 2018,
    }
    data_dict["info"] = info
    data_dict["images"] = []
    data_dict["categories"] = []
    data_dict["annotations"] = []

    for idex, name in enumerate(dota_classes):
        single_cat = {"id": idex + 1, "name": name, "supercategory": name}
        data_dict["categories"].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, "w") as f_out:
        filenames = [lbl for lbl in labelparent.iterdir()]
        for file in filenames:
            basename = file.stem

            imagepath = imageparent / (basename + ".png")
            img = cv2.imread(str(imagepath))
            if img is None:
              print("can not load!")
            else:
              height, width, c = img.shape

            single_image = {}
            single_image["file_name"] = basename + ".png"
            single_image["id"] = image_id
            single_image["width"] = width
            single_image["height"] = height
            data_dict["images"].append(single_image)

            # annotations
            objects = parse_dota_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj["area"] = obj["area"]
                single_obj["category_id"] = dota_classes.index(obj["name"]) + 1
                single_obj["segmentation"] = []
                single_obj["segmentation"].append(obj["poly"])
                single_obj["iscrowd"] = 0
                x1, y1, x2, y2, x3, y3, x4, y4 = (obj["poly"][0], obj["poly"][1], obj["poly"][2], obj["poly"][3], obj["poly"][4], obj["poly"][5], obj["poly"][6], obj["poly"][7])
                xmin, ymin, xmax, ymax = (
                   min(obj["poly"][0::2]),
                   min(obj["poly"][1::2]),
                   max(obj["poly"][0::2]),
                   max(obj["poly"][1::2]),
                )

                width, height = xmax - xmin, ymax - ymin
                # single_obj["bbox"] = xmin, ymin, width, height
                single_obj["bbox"] = x1, y1, x2, y2, x3, y3, x4, y4 # modify here
                single_obj["image_id"] = image_id
                data_dict["annotations"].append(single_obj)
                single_obj["id"] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


if __name__ == "__main__":
    out_dir = Path("home/ashwin/Desktop/Projects/dotadevkit/example_split")
    DOTA2COCO(out_dir, out_dir / "DOTA_val.json", version="1.0")
