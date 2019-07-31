import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


class Person:
    def __init__(self, bbox, keypoints):
        self.bbox = bbox
        self.keypoints = keypoints


class COCOAnnotationTransform(object):
    def __init__(self):
        pass

    def __call__(self, target):
        persons = []
        for obj in target:
            if 'bbox' in obj and 'keypoints' in obj:
                # Get person's bounding box
                person_bbox = obj['bbox']

                # Get person's bounding box width and height and compute scaling array
                bbox_w = person_bbox[2]
                bbox_h = person_bbox[3]

                # Get person's keypoints (body landmarks)
                kpts = np.array(obj['keypoints'])
                kpts_x = kpts[0::3]
                kpts_y = kpts[1::3]
                kpts_v = kpts[2::3]
                assert len(kpts_x) == len(kpts_y) == len(kpts_v)
                person_keypoints = []
                for kpt_idx in range(len(kpts_v)):
                    if kpts_v[kpt_idx] != 0:
                        person_keypoints.append([(kpts_x[kpt_idx] - person_bbox[0]) / bbox_w,
                                                 (kpts_y[kpt_idx] - person_bbox[1]) / bbox_h,
                                                 kpt_idx])
                # If at least one keypoint (body landmark) is given for this person, add them to persons list
                if len(person_keypoints) > 0:
                    current_person = Person(bbox=np.array(person_bbox), keypoints=person_keypoints)
                    persons.append(current_person)

        return persons


class COCOPerson(data.Dataset):
    def __init__(self, root, year=2017, split='train', dim=300, transform=None,
                 target_transform=COCOAnnotationTransform()):
        sys.path.append(osp.join(root, "PythonAPI"))
        from pycocotools.coco import COCO
        self.year = year
        self.split = split
        self.dim = dim
        self.root = osp.join(root, 'images', '{}{}'.format(self.split, self.year))
        self.coco = COCO(osp.join(root, 'annotations', 'person_keypoints_{}{}.json'.format(self.split, self.year)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(osp.join(self.root, path))
        img_h, img_w, _ = img.shape

        persons = self.target_transform(target)
        persons_ndarray = np.zeros((len(persons), 3, self.dim, self.dim))
        keypoints_list = []
        person_cnt = 0
        for p in persons:
            # Crop person from image
            p_bbox_x = int(p.bbox[0])
            p_bbox_y = int(p.bbox[1])
            p_bbox_w = int(p.bbox[2])
            p_bbox_h = int(p.bbox[3])
            person_img = img[p_bbox_y:p_bbox_y + p_bbox_h, p_bbox_x:p_bbox_x + p_bbox_w]
            p_keypoints = p.keypoints
            person_img, keypoints = self.transform(img=person_img, keypoints=p_keypoints)
            persons_ndarray[person_cnt, :, :, :] = person_img.transpose(2, 0, 1)
            keypoints_list.append(p_keypoints)
            person_cnt += 1

        if persons_ndarray.shape[0] > 0:
            persons_tensor = torch.from_numpy(persons_ndarray)
            persons_keypoints = keypoints_list
        else:
            persons_tensor = torch.Tensor()
            persons_keypoints = []

        return persons_tensor, persons_keypoints
