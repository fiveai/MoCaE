# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset

from .api_wrappers import COCO, COCOeval

@DATASETS.register_module()
class RobustODAV(CocoDataset):
     CLASSES = ('pedestrian', 'vehicle', 'bicycle')
     PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142)]