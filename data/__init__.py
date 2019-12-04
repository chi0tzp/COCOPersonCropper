from .coco import COCOPerson, COCOAnnotationTransform
from .augmentations import PersonAugmentor
from .collation import detection_collate
import numpy as np
import cv2


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img, keypoints=None):
        return base_transform(img, self.size, self.mean), keypoints
