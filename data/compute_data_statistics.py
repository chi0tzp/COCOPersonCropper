import sys
import numpy as np
import json
import argparse
import os.path as osp
# REVIEW: do we need to have the following?
module_path = osp.abspath(osp.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data import *
import torch.utils.data as data


def progress_updt(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r{}[{}] {:.0f}% {}".format(msg,
                                        "#" * block + "-" * (bar_length - block),
                                        round(progress * 100, 0),
                                        status)
    sys.stdout.write(text)
    sys.stdout.flush()


def write_dict(filename, d):
    f = open(filename, "w")
    f.write(json.dumps(d))
    f.close()


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Compute given dataset's per channel means.")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    parser.add_argument('--dataset', type=str, required=True, choices=('coco', 'aflw', 'qmulymc'),
                        help="select dataset")
    parser.add_argument('--mode', type=str, choices=('train', 'val'), default='train',
                        help="chose dataset's mode (training/validation)")
    parser.add_argument('--dataset_root', type=str, required=True, help="set dataset root directory")
    parser.add_argument('--dim', type=int, choices=(300, 512), default=300, help="input dimension (300 or 512)")
    args = parser.parse_args()

    if args.verbose:
        print("#. Create data loader...")

    # ================================================================================================================ #
    #                                                Load COCO dataset                                                 #
    # ================================================================================================================ #
    dataset = None
    if args.dataset == 'coco':
        cfg = cfg_coco
        cfg['inp_dim'] = args.dim
        dataset = COCOPerson(root=args.dataset_root, mode=args.mode, transform=None, cfg=cfg, use_keypoints=False)
    # Load AFLW dataset
    elif args.dataset == 'mpii':
        cfg = cfg_mpii
        cfg['inp_dim'] = args.dim
        dataset = MPII(root=args.dataset_root, mode=args.mode, transform=None, cfg=cfg)
    # Load AFLW dataset
    elif args.dataset == 'aflw':
        cfg = cfg_aflw
        cfg['inp_dim'] = args.dim
        dataset = AFLW(root=args.dataset_root, mode=args.mode, transform=None, cfg=cfg)
    # Load QMUL-YMC dataset
    elif args.dataset == 'qmulymc':
        cfg = cfg_qmulymc
        cfg['inp_dim'] = args.dim
        dataset = QMULYMC(root=args.dataset_root, mode=args.mode, transform=None, cfg=cfg)

    # Build data loader and batch iterator
    data_loader = data.DataLoader(dataset=dataset, batch_size=1, num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)

    batch_iterator = iter(data_loader)

    if args.verbose:
        print("#. Processing dataset...")

    # Define aux. TODO: +++
    per_channel_sum = np.zeros((1, 3))

    # Process COCO dataset
    for iteration in range(len(dataset)):
        # Get COCO image
        images, _ = next(batch_iterator)

        # Compute per channel sum
        img = images.squeeze(0).float()
        per_channel_sum += img.view(3, -1).mean(dim=1).numpy()

        # Show progress bar
        if args.verbose:
            progress_updt("  \_%s (dim=%d): " % (args.dataset, args.dim), len(dataset), iteration)
    per_channel_mean = per_channel_sum / len(data_loader)

    if args.verbose:
        print("")
        print("  \__Per channel mean : \n", per_channel_mean.astype(np.int))

    if args.verbose:
        print("#. Done!")


if __name__ == "__main__":
    main()
