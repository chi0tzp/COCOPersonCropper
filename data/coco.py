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
    """Transforms a COCO annotation into a Tensor of bbox coords and label index."""
    def __init__(self):
        pass

    def __call__(self, target):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
        Returns:
            persons (list): list of Person class instances

        """
        persons = []
        for obj in target:
            if 'bbox' in obj and 'keypoints' in obj:
                # Get person's bounding box
                person_bbox = obj['bbox']

                # Get person's bounding box width and height and compute scaling array
                bbox_w = person_bbox[2]
                bbox_h = person_bbox[3]

                # Transform person's bounding box in the following format [P_x, P_y, Q_x, Q_y], where
                # (P_x, P_y) is the upper left and (Q_x, Q_y) is the bottom right poinds
                # person_bbox[2] += person_bbox[0]
                # person_bbox[3] += person_bbox[1]

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
            # REVIEW: debugging
            # if len(persons) > 0:
            #     return persons

        return persons

        # scale = np.array([width, height, width, height])
        # res = []
        # for obj in target:
        #     # Get bounding box
        #     bbox_w = bbox_h = 0
        #     if 'bbox' in obj:
        #         print("*** Found person bbox!")
        #         bbox = obj['bbox']
        #         bbox_w = bbox[2]
        #         bbox_h = bbox[3]
        #         bbox[2] += bbox[0]
        #         bbox[3] += bbox[1]
        #         bbox_list = list(np.array(bbox) / scale)
        #         # Set label idx corresponding to class "person" (see data/config.py)
        #         label_idx = cfg['person_id']
        #         bbox_list.append(label_idx)
        #         res += [bbox_list]  # res += [xmin, ymin, xmax, ymax, label_idx]
        #
        #         # Get keypoints as objects
        #         if 'keypoints' in obj:
        #             print("\t*** Found keypoints!")
        #             kpts = np.array(obj['keypoints'])
        #             # print("Found keypoints!")
        #             kpts_x = kpts[0::3]
        #             kpts_y = kpts[1::3]
        #             kpts_v = kpts[2::3]
        #             assert len(kpts_x) == len(kpts_y) == len(kpts_v)
        #             for kpt_idx in range(len(kpts_v)):
        #                 # print("kpts_v[kpt_idx] = ", kpts_v[kpt_idx])
        #                 # Check if
        #                 #   i) the current keypoint belongs to PERSON_KEYPOINTS_MAP.keys()
        #                 #   ii) the `kpt-idx`-th keypoint is visible
        #                 # Then, set keypoint as an object and create a bounding box around it.
        #                 if (kpt_idx in cfg["person_keypoints_map"].keys()) and (kpts_v[kpt_idx] != 0):
        #                     person_bbox_ref = np.mean([bbox_w, bbox_h])
        #                     if kpt_idx < 5:
        #                         a = 0.075
        #                     else:
        #                         a = 0.2
        #
        #                     kpt_bbox = [kpts_x[kpt_idx] - int(a * person_bbox_ref / 2),
        #                                 kpts_y[kpt_idx] - int(a * person_bbox_ref / 2),
        #                                 kpts_x[kpt_idx] + int(a * person_bbox_ref / 2),
        #                                 kpts_y[kpt_idx] + int(a * person_bbox_ref / 2)]
        #                     kpt_list = list(np.array(kpt_bbox) / scale)
        #                     label_idx = cfg["person_keypoints_map"][kpt_idx]
        #                     kpt_list.append(label_idx)
        #                     res += [kpt_list]
        #         else:
        #             print("no keypoints problem!")
        #     else:
        #         print("no bbox problem!")
        # return res


class COCOPerson(data.Dataset):
    """`MS COCO dataset for human pose estimation.
    Args:
        root (string): Root directory where images have been downloaded to.
        transform (callable, optional): A function/transform that augments the raw images.
        cfg (dict): ---
        target_transform (callable, optional): A function/transform that takes in the target (bbox) and transforms it.
    """

    def __init__(self,
                 root,
                 year=2017,
                 split='train',
                 transform=None,
                 cfg=None,
                 target_transform=COCOAnnotationTransform()):
        sys.path.append(osp.join(root, "PythonAPI"))
        from pycocotools.coco import COCO
        self.year = year
        self.split = split
        self.root = osp.join(root, 'images', '{}{}'.format(self.split, self.year))
        self.coco = COCO(osp.join(root, 'annotations', 'person_keypoints_{}{}.json'.format(self.split, self.year)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cfg = cfg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (person_imgs, keypoints).
                   target is the object returned by ``coco.loadAnns``.
        """
        # person_imgs, keypoints = self.pull_item(index)
        # return person_imgs, keypoints
        return self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(osp.join(self.root, path))
        img_h, img_w, _ = img.shape

        # TODO: add comment
        persons = self.target_transform(target)

        persons_ndarray = np.zeros((len(persons), 3, 300, 300))
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

            # target = np.array(target)
        # if self.transform is not None:
        #     img, boxes, labels = self.transform(img=img, boxes=target[:, :4], labels=target[:, -1])
        # else:
        #     boxes, labels = target[:, :4], target[:, -1]
        # bbox_target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # return torch.from_numpy(img).permute(2, 0, 1), bbox_target, img_h, img_w, img_id, path

        # return torch.from_numpy(persons_ndarray).permute(2, 0, 1), keypoints_list, img_h, img_w, img_id, path

    def pull_image(self, index):
        """ Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        """
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
