# COCOPersonCropper

A PyTorch data loader class for on-the-fly cropping persons from [COCO dataset](http://cocodataset.org/#home), along with their body-landmarks annotations.

**Requirements**

- pytorch (version > 1.0)
- opencv
- [cocoapi](https://github.com/cocodataset/cocoapi)

An auxiliary visualization script is `visualize_data.py`, which can be run with various options (dataset split, dataset year version, cropped image dimensions, with/without augmentations) as shown below:

~~~
python visualize_data.py -h
usage: [-h] [-v] --dataset_root DATASET_ROOT [--year {2014,2017}] [--split {train,val}] [--batch_size BATCH_SIZE] [--dim {300,512}] [-a]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity
  --dataset_root DATASET_ROOT
                        set dataset root directory
  --year {2014,2017}    set COCO dataset year
  --split {train,val}   chose dataset's split (training/testing)
  --batch_size BATCH_SIZE
                        set batch size
  --dim {300,512}       set input dimension
  -a, --augment         add augmentations (see data/augmentations.py)
~~~



Argument `dataset_root` should be set to the root directory of the dataset for the given version (`year=2014` or `year=2017`). For instance, after downloading and extracting COCO 2017 dataset, its root directory should look as follows:

~~~
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
└── images
    ├── test2017
    ├── train2017
    └── val2017
~~~



Some examples of cropped person images taken from COCO 2017 training subset and resized to 300x300 are shown below.

<p align="center">
  <img width="300" height="300" src="./examples/example_1.png"></img>
  <img width="300" height="300" src="./examples/example_2.png"></img>
  <img width="300" height="300" src="./examples/example_3.png"></img>
  <img width="300" height="300" src="./examples/example_4.png"></img>
</p>

