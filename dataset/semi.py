from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random
from scipy.ndimage.interpolation import zoom
from imgaug import augmenters as iaa
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
from dataset.tree_process import tree_process_3D_final,tree_process_3D_higra_final




class SemiDataset_Medical_Pancreas_CT(Dataset):
    def __init__(self, name, root, mode, output_size=None, id_path=None, nsample=None,tree_mode=0,tree_areathrehold=0,tree_topoprob=0):
        self.name = name
        self.root = root
        self.mode = mode
        self.output_size = output_size
        self.tree_mode = tree_mode
        self.tree_areathrehold = tree_areathrehold
        self.tree_topoprob = tree_topoprob

        self.transform1 = transforms.Compose([
            RandomCrop_3D(output_size),
        ])
        self.transform2 = transforms.Compose([
            ToTensor_3D(),
        ])
        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('partitions/%s/test.list' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        if self.mode == 'train_l' or self.mode == 'train_u':

            rand_type = np.random.randint(low=0, high=3, size=2, dtype='int')
            data_npz_similar = np.load(
                self.root + "/data_nonlinear/Pancreas_h5/" + id + "_norm_type_{}.npz".format(
                    rand_type[0] * 2))
            data_npz_dissimilar = np.load(
                    self.root + "/data_nonlinear/Pancreas_h5/" + id + "_norm_type_{}.npz".format(
                        rand_type[1] * 2))
            image_similar = data_npz_similar['image']
            image_dissimilar = data_npz_dissimilar['image']
            label = data_npz_similar['label']


            image_similar_max = np.max(image_similar)
            image_similar_min = np.min(image_similar)
            image_similar = (image_similar - image_similar_min) / (image_similar_max - image_similar_min + 1e-7)
            # image_similar_w, image_similar_h, image_similar_d = image_similar.shape
            # image_similar_reshape = image_similar.reshape(image_similar_w, image_similar_h * image_similar_d)

            image_dissimilar_max = np.max(image_dissimilar)
            image_dissimilar_min = np.min(image_dissimilar)
            image_dissimilar = (image_dissimilar - image_dissimilar_min) / (image_dissimilar_max - image_dissimilar_min + 1e-7)
            # image_dissimilar_w, image_dissimilar_h, image_dissimilar_d = image_dissimilar.shape
            # image_dissimilar_reshape = image_dissimilar.reshape(image_dissimilar_w, image_dissimilar_h * image_dissimilar_d)

            sample = {"image_similar": image_similar, "image_dissimilar": image_dissimilar, "label": label}
            sample = self.transform1(sample)


            image_similar = sample["image_similar"]
            image_dissimilar = sample["image_dissimilar"]

            
            rotate_angle = int(0)
            if self.tree_mode == 1:
                if random.random() > 0.5:
                    if random.random() >= 0:
                        rotate_angle = int(np.random.randint(low=0, high=4, size=1, dtype='int') * 90)
                    if random.random() > 0.5:
                        image_similar = tree_process_3D_final(image_similar,'maxtree', self.tree_areathrehold,self.tree_topoprob,0)
                        image_dissimilar = tree_process_3D_final(image_dissimilar, 'mintree', self.tree_areathrehold,self.tree_topoprob,rotate_angle)   
                    else:
                        image_similar = tree_process_3D_final(image_similar,  'mintree',self.tree_areathrehold,self.tree_topoprob,0)
                        image_dissimilar = tree_process_3D_final(image_dissimilar, 'maxtree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)   

            image_similar = image_similar * (image_similar_max - image_similar_min) + image_similar_min
            image_dissimilar = image_dissimilar * (image_dissimilar_max - image_dissimilar_min) + image_dissimilar_min

            
            sample["image_similar"] = image_similar
            sample["image_dissimilar"] = image_dissimilar
            sample = self.transform2(sample)


            sample["rotate_angle"]  = rotate_angle
            sample["id"] = id
            return sample

        else:
            h5f = h5py.File(self.root + "/data/Pancreas_h5/" + id + "_norm.h5", 'r')
            image = h5f["image"][:]
            label = h5f["label"][:]
            sample = {"image": image, "label": label}
            sample["id"] = id
            return sample

    def __len__(self):
        return len(self.ids)