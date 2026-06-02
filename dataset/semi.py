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
from dataset.tree_process import tree_process_final,tree_process_3D_final,tree_process_higra_final,tree_process_3D_higra_final


class SemiDataset_Medical_2D(Dataset):
    def __init__(self, name, root, mode, output_size=None, id_path=None, nsample=None,bezier_type=[-1,-1],gammacontrast_type=(0,0), tree_mode=0, tree_areathrehold=0,tree_topoprob=0):
        self.name = name
        self.root = root
        self.mode = mode
        self.output_size = output_size
        self.bezier_type = bezier_type
        self.gammacontrast_type = gammacontrast_type
        self.tree_mode = tree_mode
        self.tree_areathrehold = tree_areathrehold
        self.tree_topoprob = tree_topoprob

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('partitions/%s/val.list' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        if self.mode == 'train_l' or self.mode == 'train_u':
            if self.bezier_type==[-1,-1]:
                h5f = h5py.File(self.root + "/data/slices/{}.h5".format(id), "r")
                image = h5f["image"][:]
                label = h5f["label"][:]
                image_similar = image
                image_dissimilar = image
            else:
                rand_type = np.random.randint(low=0, high=self.bezier_type[0], size=2, dtype='int')
                data_npz_ori = h5py.File(self.root + "/data/slices/{}.h5".format(id), "r")
                data_npz_similar = np.load(
                    self.root + "/data_nolinear/slices/{}".format(id) + "_type_{}.npz".format(
                        rand_type[0] * 2))
                data_npz_dissimilar = np.load(
                    self.root + "/data_nolinear/slices/{}".format(id) + "_type_{}.npz".format(
                        rand_type[1] * 2))

                image_similar = data_npz_similar['image']
                image_dissimilar = data_npz_dissimilar['image']

                label = data_npz_similar['label']



            if random.random() > 0.5:
                image_similar, image_dissimilar, label = random_rot_flip_2D(image_similar, image_dissimilar,label)
            elif random.random() > 0.5:
                image_similar, image_dissimilar, label = random_rotate_2D(image_similar, image_dissimilar,label)



            x_img_similar, y_img_similar = image_similar.shape
            x_img_dissimilar, y_img_dissimilar = image_dissimilar.shape

            x_label, y_label = label.shape
            image_similar = zoom(image_similar,(self.output_size[0] / x_img_similar, self.output_size[1] / y_img_similar), order=0)
            image_dissimilar = zoom(image_dissimilar,
                                    (self.output_size[0] / x_img_dissimilar, self.output_size[1] / y_img_dissimilar),
                                    order=0)

            label = zoom(label, (self.output_size[0] / x_label, self.output_size[1] / y_label), order=0)




            rotate_angle = int(0)
            if random.random() > 0.5:
                if random.random() > 1:
                    rotate_angle = int(np.random.randint(low=0, high=4, size=1, dtype='int') * 90)
                if random.random() > 0.5:
                    if self.tree_mode == 1:
                        image_similar = tree_process_final(image_similar, 'maxtree', self.tree_areathrehold,  self.tree_topoprob, 0)
                        image_dissimilar = tree_process_final(image_dissimilar,'mintree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)
                    elif self.tree_mode == 2:
                        image_similar = tree_process_higra_final(image_similar, 'maxtree', self.tree_areathrehold,  self.tree_topoprob, 0)
                        image_dissimilar = tree_process_higra_final(image_dissimilar,'mintree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)

                else:
                    if self.tree_mode == 1:
                        image_similar = tree_process_final(image_similar, 'mintree', self.tree_areathrehold, self.tree_topoprob, 0)
                        image_dissimilar = tree_process_final(image_dissimilar,'maxtree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)
                    elif self.tree_mode == 2:
                        image_similar = tree_process_higra_final(image_similar, 'mintree', self.tree_areathrehold, self.tree_topoprob, 0)
                        image_dissimilar = tree_process_higra_final(image_dissimilar,'maxtree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)




            image_similar = torch.from_numpy(image_similar.astype(np.float32)).unsqueeze(0)
            image_dissimilar = torch.from_numpy(image_dissimilar.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))


            sample = {"image_similar": image_similar, "image_dissimilar": image_dissimilar, "label": label, 'rotate_angle':rotate_angle,'id':id}
            return sample

        else:
            h5f = h5py.File(self.root + "/data/{}.h5".format(id), "r")
            image = h5f["image"][:]
            label = h5f["label"][:]
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            sample = {"image": image, "label": label}
            sample["id"] = id
            return sample

    def __len__(self):
        return len(self.ids)





class SemiDataset_Medical_3D(Dataset):
    def __init__(self, name, root, mode, output_size=None, id_path=None, nsample=None,bezier_type=[-1,-1],gammacontrast_type=(0,0),tree_mode=0,tree_areathrehold=0,tree_topoprob=0):
        self.name = name
        self.root = root
        self.mode = mode
        self.output_size = output_size
        self.bezier_type = bezier_type
        self.gammacontrast_type = gammacontrast_type
        self.tree_mode = tree_mode
        self.tree_areathrehold = tree_areathrehold
        self.tree_topoprob = tree_topoprob

        if self.name == 'LA_DATASET':
            self.transform1 = transforms.Compose([
                RandomRotFlip_3D(),
                RandomCrop_3D(output_size),
                # ToTensor_3D(),
            ])
            self.transform2 = transforms.Compose([
                ToTensor_3D(),
            ])
        elif self.name == 'Pancreas_CT':
            self.transform1 = transforms.Compose([
                RandomCrop_3D(output_size),
                # ToTensor_3D(),
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
            if self.bezier_type==[-1,-1]:
                if self.name == 'LA_DATASET':
                    h5f = h5py.File(self.root + "/data/2018LA_Seg_Training Set/" + id + "/mri_norm2.h5", 'r')
                elif self.name == 'Pancreas_CT':
                    h5f = h5py.File(self.root + "/data/Pancreas_h5/" + id + "_norm.h5", 'r')
                image = h5f["image"][:]
                label = h5f["label"][:]
                image_similar = image
                image_dissimilar = image

            else:
                rand_type = np.random.randint(low=0, high=self.bezier_type[0], size=2, dtype='int')
                if self.name == 'LA_DATASET':
                    data_npz_similar = np.load(
                        self.root + "/data_nonlinear/2018LA_Seg_Training Set/"+id + "/mri_norm2_type_{}.npz".format(
                            rand_type[0] * 2))

                    data_npz_dissimilar = np.load(
                            self.root + "/data_nonlinear/2018LA_Seg_Training Set/"+id + "/mri_norm2_type_{}.npz".format(
                                rand_type[1] * 2))
                elif self.name == 'Pancreas_CT':
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

            if self.name == 'LA_DATASET':
                image_similar = image_similar.transpose(2, 0, 1)
                image_dissimilar = image_dissimilar.transpose(2, 0, 1)

            rotate_angle = int(0)
            if random.random() > 0.5:
                if random.random() >= 0:
                    rotate_angle = int(np.random.randint(low=0, high=4, size=1, dtype='int') * 90)
                if random.random() > 0.5:
                    if self.tree_mode == 1:
                        image_similar = tree_process_3D_final(image_similar,'maxtree', self.tree_areathrehold,self.tree_topoprob,0)
                        image_dissimilar = tree_process_3D_final(image_dissimilar, 'mintree', self.tree_areathrehold,self.tree_topoprob,rotate_angle)
                    elif self.tree_mode == 2:
                        image_similar = tree_process_3D_higra_final(image_similar, 'maxtree',self.tree_areathrehold,self.tree_topoprob,0)
                        image_dissimilar = tree_process_3D_higra_final(image_dissimilar, 'mintree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)
                else:
                    if self.tree_mode == 1:
                        image_similar = tree_process_3D_final(image_similar,  'mintree',self.tree_areathrehold,self.tree_topoprob,0)
                        image_dissimilar = tree_process_3D_final(image_dissimilar, 'maxtree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)
                    elif self.tree_mode == 2:
                        image_similar = tree_process_3D_higra_final(image_similar, 'mintree', self.tree_areathrehold, self.tree_topoprob,0)
                        image_dissimilar = tree_process_3D_higra_final(image_dissimilar, 'maxtree', self.tree_areathrehold, self.tree_topoprob,rotate_angle)
            if self.name == 'LA_DATASET':
                image_similar = image_similar.transpose(1, 2, 0)
                image_dissimilar = image_dissimilar.transpose(1, 2, 0)

            image_similar = image_similar * (image_similar_max - image_similar_min) + image_similar_min
            image_dissimilar = image_dissimilar * (image_dissimilar_max - image_dissimilar_min) + image_dissimilar_min

            
            sample["image_similar"] = image_similar
            sample["image_dissimilar"] = image_dissimilar
            sample = self.transform2(sample)


            sample["rotate_angle"]  = rotate_angle
            sample["id"] = id
            return sample


        else:
            if self.name == 'LA_DATASET':
                h5f = h5py.File(self.root + "/data/2018LA_Seg_Training Set/" + id + "/mri_norm2.h5", 'r')
            elif self.name == 'Pancreas_CT':
                h5f = h5py.File(self.root + "/data/Pancreas_h5/" + id + "_norm.h5", 'r')
            image = h5f["image"][:]
            label = h5f["label"][:]
            sample = {"image": image, "label": label}
            sample["id"] = id
            return sample

    def __len__(self):
        return len(self.ids)


