import sys
import argparse
from data import *
import torch.utils.data as data
import numpy as np
import cv2


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument("--dataset_root", type=str, required=True, help="set dataset root directory")
    parser.add_argument("--year", type=int, default=2017, choices=(2014, 2017), help="set COCO dataset year")
    parser.add_argument("--split", type=str, choices=('train', 'val'), default='train',
                        help="chose dataset's split (training/testing)")
    parser.add_argument("--batch_size", type=int, default=4, help='set batch size')
    parser.add_argument("--dim", type=int, choices=(300, 512), default=300, help="set input dimension")
    parser.add_argument('-a', '--augment', action='store_true', help="add augmentations (see data/augmentations.py)")
    args = parser.parse_args()

    cfg = cfg_coco
    cfg['inp_dim'] = args.dim

    # Load COCO dataset for human pose estimation (cropped persons with boby keypoints)
    if args.augment:
        transform = PersonAugmentor(size=cfg['inp_dim'], mean=cfg['means'])
    else:
        transform = BaseTransform(size=cfg['inp_dim'], mean=(0, 0, 0))
    dataset = COCOPerson(root=args.dataset_root, year=args.year, split=args.split, dim=cfg['inp_dim'], transform=transform)

    # Build data loader
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=1, shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # Build batch iterator
    batch_iterator = iter(data_loader)

    # Get images, targets in batch mode
    #   images (torch.Tensor)  : tensor of person images of size: torch.Size([num_persons, 3, inp_dim, inp_dim])
    #   keypoints (list)       : list of length equal to num_persons
    images, keypoints = next(batch_iterator)

    if args.verbose:
        print("# Found {} persons in {} images".format(len(keypoints), args.batch_size))

    # If number of persons are more than batch_size, sample randomly from them
    if len(keypoints) > args.batch_size:
        I = np.random.randint(low=0, high=len(keypoints), size=args.batch_size)
        images = images[I]
        keypoints = [keypoints[i] for i in I]

    for i in range(len(keypoints)):

        if args.verbose:
            print("*Person {}".format(i))
            print(" ==================")

        img = images[i].numpy().transpose(1, 2, 0).astype(np.uint8).copy()
        for k in keypoints[i]:
            k_x = int(cfg['inp_dim'] * k[0])
            k_y = int(cfg['inp_dim'] * k[1])
            k_label = k[2]

            if args.verbose:
                print("\t{}: ({}, {})".format(classes_coco[k_label], k_x, k_y))

            # Draw keypoint
            cv2.circle(img=img, center=(k_x, k_y), radius=2, color=(255, 0, 255), thickness=2)
            cv2.circle(img=img, center=(k_x, k_y), radius=4, color=(0, 255, 255), thickness=2)
            
            # Show image
            cv2.imshow("Person: {}".format(i), img)
            cv2.waitKey()


if __name__ == "__main__":
    main()
