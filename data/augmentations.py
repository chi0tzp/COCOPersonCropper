import torch
import cv2
import numpy as np
from numpy import random


class ConvertFromInts(object):
    def __call__(self, image, keypoints=None):
        return image.astype(np.float32), keypoints


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, keypoints=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), keypoints


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, keypoints=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, keypoints


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "Contrast upper must be >= lower."
        assert self.lower >= 0, "Contrast lower must be non-negative."

    def __call__(self, image, keypoints=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, keypoints


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, keypoints=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, keypoints


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, keypoints=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, keypoints


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, keypoints=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, keypoints


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "Contrast upper must be >= lower."
        assert self.lower >= 0, "Contrast lower must be non-negative."

    def __call__(self, image, keypoints=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, keypoints


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, keypoints=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, keypoints


class ToCV2Image(object):
    def __call__(self, tensor, keypoints=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), keypoints


class ToTensor(object):
    def __call__(self, cvimage, keypoints=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), keypoints


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, keypoints):
        im = image.copy()
        im, keypoints = self.rand_brightness(im, keypoints)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, keypoints = distort(im, keypoints)
        return self.rand_light_noise(im, keypoints)


class Compose(object):
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, img, keypoints=None):
        for t in self.transforms_list:
            img, keypoints = t(img, keypoints)
        return img, keypoints


class PersonAugmentor(object):
    def __init__(self, size=300, mean=(107, 114, 118)):
        self.mean = mean
        self.size = size
        self.augment = Compose([ConvertFromInts(),
                                PhotometricDistort(),
                                Resize(self.size),
                                SubtractMeans(self.mean)])

    def __call__(self, img, keypoints):
        return self.augment(img, keypoints)
