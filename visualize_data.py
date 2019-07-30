import sys
import argparse
from data import *
import torch
import torch.utils.data as data
import numpy as np
import cv2


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument("--dataset_root", type=str, required=True, help="set dataset root directory")
    parser.add_argument("--split", type=str, choices=('train', 'val'), default='train',
                        help="chose dataset's split (training/testing)")
    parser.add_argument("--batch_size", type=int, default=4, help='set batch size')
    parser.add_argument("--dim", type=int, choices=(300, 512), default=300, help="set input dimension")
    args = parser.parse_args()

    cfg = cfg_coco
    cfg['inp_dim'] = args.dim
    LABELS = cfg['classes']
    PERSON_KEYPOINTS = cfg['classes'][1:]

    # Load COCO dataset for human pose estimation
    # dataset = COCOPerson(root=args.dataset_root, year=2017, split=args.split, cfg=cfg,
    #                      transform=Augmentor(size=cfg['inp_dim'],
    #                                          mean=cfg['means'],
    #                                          kpts_mirror_map=cfg['kpts_mirror_map']))

    dataset = COCOPerson(root=args.dataset_root, year=2017, split=args.split, cfg=cfg,
                         transform=BaseTransform(size=cfg['inp_dim'], mean=cfg['means']))

    # Build data loader
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=1, shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # Build batch iterator
    batch_iterator = iter(data_loader)

    # Get images, targets in batch mode
    images, bbox_targets = next(batch_iterator)

    print(images.size())
    sys.exit()




    # Get images widths and heights
    height = images.size(2)
    width = images.size(3)

    scale = torch.Tensor([width, height, width, height])
    for i in range(args.batch_size):

        if args.verbose:
            print("\tImage: %d" % i)
            print("\t" + 80 * "=")

        img = images[i].numpy().transpose(1, 2, 0).astype(np.uint8).copy()

        for j in range(len(bbox_targets[i])):
            # Get object's bounding box and label
            bbox = bbox_targets[i][j][:4]
            label = int(bbox_targets[i][j][-1])
            # pt = (bbox * scale).numpy()
            if args.verbose:
                print("\t\tObject %02d ===> Label: %s" % (j, LABELS[label]))

            # color = (54, 241, 10)
            # cv2.rectangle(img, pt1=(pt[0], pt[1]), pt2=(pt[2], pt[3]), color=color, thickness=2)

        cv2.imshow("image: {}".format(i), img)
        cv2.waitKey()


if __name__ == "__main__":
    main()
