import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from scipy import ndimage


def random_rot_flip_3D(image_similar,image_dissimilar,label):
    k = np.random.randint(0, 4)
    image_similar = np.rot90(image_similar, k)
    image_dissimilar = np.rot90(image_dissimilar, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)

    image_similar = np.flip(image_similar, axis=axis).copy()
    image_dissimilar = np.flip(image_dissimilar, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image_similar,image_dissimilar, label

class RandomRotFlip_3D(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image_similar,image_dissimilar, label = sample['image_similar'],sample['image_dissimilar'],sample['label']
        image_similar, image_dissimilar,label = random_rot_flip_3D(image_similar,image_dissimilar, label)

        return {'image_similar': image_similar,'image_dissimilar': image_dissimilar,  'label': label}

class RandomCrop_3D(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_similar,image_dissimilar,label = sample['image_similar'],sample['image_dissimilar'], sample['label']


        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image_similar = np.pad(image_similar, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image_dissimilar = np.pad(image_dissimilar, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)


        (w, h, d) = image_similar.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_similar = image_similar[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_dissimilar = image_dissimilar[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image_similar': image_similar, 'image_dissimilar': image_dissimilar,  'label': label}

class ToTensor_3D(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_similar,image_dissimilar, = sample['image_similar'],sample['image_dissimilar']
        image_similar = image_similar.reshape(1, image_similar.shape[0], image_similar.shape[1], image_similar.shape[2]).astype(np.float32)
        image_dissimilar = image_dissimilar.reshape(1, image_dissimilar.shape[0], image_dissimilar.shape[1],
                                              image_dissimilar.shape[2]).astype(np.float32)

        if 'onehot_label' in sample:
            return {'image_similar': torch.from_numpy(image_similar),'image_dissimilar': torch.from_numpy(image_dissimilar),  'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image_similar': torch.from_numpy(image_similar),'image_dissimilar': torch.from_numpy(image_dissimilar),'label': torch.from_numpy(sample['label']).long()}













def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
