import cv2
import numpy as np
import random

from pathlib import Path
import copy
# import tree_trans as trtr
import tree_trans3D as trtr3D
import matplotlib.pyplot as plt
import json
from imgaug import augmenters as iaa
import h5py
import time
from scipy.ndimage import rotate
from PIL import Image
import higra as hg


def tree_view3D(root,tree,h,w,d,img):
    nodes = [root]
    index = 0
    img_new = np.zeros((h,w,d))
    while len(nodes) != 0:
        cur_node = nodes.pop()
        parents, next_nodes, members, area = tree[cur_node]
        nodes.extend(next_nodes)
        cur_x, cur_y,cur_z = cur_node // (w * d), cur_node % (w * d) // d, cur_node % (w * d) % d
        members = np.array(members)
        xs, yzs = np.divmod(members, w * d)
        ys, zs = np.divmod(yzs, d)
        img_new[xs,ys,zs] = img[cur_x,cur_y,cur_z]
        index += 1
    return img_new

def tree_areapruning(root,tree,area_threshold):
    tree_new = copy.deepcopy(tree)
    nodes = [root]

    while len(nodes) != 0:
        cur_node = nodes.pop()
        parents, next_nodes, members, area = tree_new[cur_node]
        if area[0] < area_threshold:
            tree_new[parents[0]][1].remove(cur_node)
            tree_new[parents[0]][2].extend(members)
            del tree_new[cur_node]
            merg_nodes = []
            merg_nodes.extend(next_nodes)
            while len(merg_nodes) != 0:
                merg_node = merg_nodes.pop()
                _, merg_next_nodes, merg_members, merg_area = tree_new[merg_node]
                tree_new[parents[0]][2].extend(merg_members)
                merg_nodes.extend(merg_next_nodes)
                del tree_new[merg_node]
        else:
            nodes.extend(next_nodes)

    return tree_new

def tree_topopruning(root,tree):
    tree_new = copy.deepcopy(tree)
    nodes = [root]
    while len(nodes) != 0:
        cur_node = nodes.pop()
        parents, next_nodes, members, area = tree_new[cur_node]
        if len(next_nodes) == 1:
            removed_node = next_nodes[0]
            tree_new[cur_node][1].extend(tree_new[removed_node][1])
            tree_new[cur_node][2].extend(tree_new[removed_node][2])
            tree_new[cur_node][1].remove(removed_node)
            for next_node in tree_new[removed_node][1]:
                tree_new[next_node][0] = [cur_node]
            nodes.append(cur_node)
            del tree_new[removed_node]
        else:
            nodes.extend(next_nodes)
    return tree_new


def tree_topopruning_random(root,tree,prob):
    tree_new = copy.deepcopy(tree)
    nodes = [root]
    while len(nodes) != 0:
        cur_node = nodes.pop()
        parents, next_nodes, members, area = tree_new[cur_node]
        if len(next_nodes) == 1:
            if random.random() < prob:
                nodes.extend(next_nodes)
            else:
                removed_node = next_nodes[0]
                tree_new[cur_node][1].extend(tree_new[removed_node][1])
                tree_new[cur_node][2].extend(tree_new[removed_node][2])
                tree_new[cur_node][1].remove(removed_node)
                for next_node in tree_new[removed_node][1]:
                    tree_new[next_node][0] = [cur_node]
                nodes.append(cur_node)
                del tree_new[removed_node]
        else:
            nodes.extend(next_nodes)

    return tree_new



def tree_process_3D_final(img,tree_type,area_threshold,topoprob,rotate_angle):


    img_h, img_w, img_d = img.shape
    img_reshape = img.reshape(img_h, img_w * img_d)
    if random.random() > 0.5:
        contrast = iaa.GammaContrast((0.5, 1.5))
        img_reshape = contrast.augment_image(img_reshape)

    img = img_reshape.reshape(img_h, img_w, img_d)

    img = rotate(img, axes=(1, 2), angle=rotate_angle, reshape=False, order=1)




    img = img *255
    img = img.astype(np.uint8)
    _, root, tree = trtr3D.min_max_tree3D(img,tree_type, 26)
    new_tree = tree_areapruning(root,tree,area_threshold)
    new_new_tree = tree_topopruning_random(root,new_tree,topoprob)
    img = tree_view3D(root, new_new_tree,img_h,img_w,img_d,img)

    img = img.astype(np.float32)
    img = img / 255


    return img


