import cv2
import numpy as np
import random

from pathlib import Path
import copy
import tree_trans3D as trtr
import matplotlib.pyplot as plt
import json
import h5py
import time
import random
from PIL import Image
# # random.seed(2023)
# root = Path("/mnt/d/Documents/Datasets/VesselDatasets/DRIVE/test/images/")
# # root = Path("./assets/ACDC_ORIGIN")
# save = Path("./assets/DRIVE_TEST_DEBUG")
# save.mkdir(parents=True, exist_ok=True)
# gray_save = save / 'gray'
# gray_save.mkdir(exist_ok=True)
# simplified_save = save / 'simplified'
# simplified_save.mkdir(exist_ok=True)
# linear_save = save / 'linear'
# linear_save.mkdir(exist_ok=True)

# def tree_transform(path):
#     img = cv2.imread(str(path), flags=cv2.IMREAD_GRAYSCALE)
#
#     # tree(dict), [parents, next_nodes, members, area]
#     # 由于生成的时候 map 是有序的，因此对后续有帮助
#     _, root, tree = trtr.min_max_tree(img, "maxtree", 8)
#
#     area_threshold = 1024
#     branch_await = 0
#     nodes = [root]
#     branch = []
#     branches = []
#     branch_nodes = []
#     nodes_list = []
#     while len(nodes) != 0:
#         cur_node = nodes.pop()
#         parents, next_nodes, members, area = tree[cur_node]
#
#         branch.extend(members)
#         branch_nodes.append(cur_node)
#         if area[0] < area_threshold:
#             branch_await -= 1
#         if len(next_nodes) != 1:
#             for next_node in next_nodes:
#                 nn_area = tree[next_node][3][0]
#                 if nn_area >= area_threshold:
#                     nodes.insert(0, next_node)
#                 else:
#                     branch_await += 1
#                     nodes.append(next_node)
#             if branch_await == 0:
#                 branches.append(branch)
#                 branch = []
#                 nodes_list.append(branch_nodes)
#                 branch_nodes = []
#         else:   # len(next_nodes) == 1 时
#             if tree[next_nodes[0]][3][0] < area_threshold:
#                     branch_await += 1
#             nodes.extend(next_nodes)
#
#     return img, branches, nodes_list



def tree_view(root,tree,h,w,d,img):
    nodes = [root]
    index = 0
    img_new = np.zeros((h,w,d))
    while len(nodes) != 0:
        cur_node = nodes.pop()
        parents, next_nodes, members, area = tree[cur_node]
        # print(cur_node, parents, next_nodes, members, area)
        nodes.extend(next_nodes)
        cur_x, cur_y,cur_z = cur_node // (w * d), cur_node % (w * d) // d, cur_node % (w * d) % d
        members = np.array(members)
        xs, yzs = np.divmod(members, w * d)
        ys, zs = np.divmod(yzs, d)
        img_new[xs,ys,zs] = img[cur_x,cur_y,cur_z]
        # for member in members:
        #     x,y,z = member // (w * d), member % (w * d) // d, member % (w * d) % d
        #     img_new[x,y,z] = img[cur_x,cur_y,cur_z]
            # img_new[y, x] = index
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


def tree_topopruning_random(root,tree):
    tree_new = copy.deepcopy(tree)
    nodes = [root]
    while len(nodes) != 0:
        cur_node = nodes.pop()
        parents, next_nodes, members, area = tree_new[cur_node]
        if len(next_nodes) == 1:
            if random.random() < 0.1:
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

def cal_toponum(root,tree):
    nodes = [root]
    index = 0
    while len(nodes) != 0:
        cur_node = nodes.pop()
        parents, next_nodes, members, area = tree[cur_node]
        if len(next_nodes) == 1:
            index += 1
        nodes.extend(next_nodes)
    return index
# for file in root.iterdir():
#     img, branches, nodes_list = tree_transform(file)
path = '/data/ylgu/Medical/Semi_medical_data/LA_DATASET/data/2018LA_Seg_Training Set/0RZDK210BSMWAA6467LU/mri_norm2.h5'
h5f = h5py.File(path, "r")
img = h5f["image"][:]
img = (img-np.min(img))/(np.max(img)-np.min(img))*255
img = img.astype(np.uint8)
# img = np.transpose(img,(0,2,1))
label = h5f["label"][:]
# image = Image.open(path).convert('L')
# img = np.array(image)
# img = np.transpose(img,(1,0))
# img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
h,w,d = img.shape
s = time.time()
_, root, tree = trtr.min_max_tree3D(img, "mintree", 6)
e = time.time()
print(e-s)
# image1 = tree_view(root, tree,h,w,d,img)

# different = image1-img
# pass
new_tree = tree_areapruning(root,tree,50)
# with open('new_tree.json', 'w') as file:
#     json.dump(new_tree, file)
#
# num = cal_toponum(root,new_tree)
image2 = tree_view(root, new_tree,h,w,d,img)
new_new_tree = tree_topopruning(root,new_tree)
new_new_random_tree = tree_topopruning_random(root,new_tree)
# with open('new_new_tree.json', 'w') as file:
#     json.dump(new_new_tree, file)
#
#
image3 = tree_view(root, new_new_tree,h,w,d,img)
image3_random = tree_view(root, new_new_random_tree,h,w,d,img)
#
#
#
#
# # print(image2-image3)
# different = image2-image3
img = (img-np.min(img))/(np.max(img)-np.min(img))*255
# image1 = image1/np.max(image1)*255
image2 = image2/np.max(image2)*255
image3 = image3/np.max(image3)*255
image3_random = image3_random/np.max(image3)*255
# cv2.imwrite('img.png',img)
# cv2.imwrite('image1.png',image1)
# cv2.imwrite('image2.png',image2)
# cv2.imwrite('image3.png',image3)
plt.subplot(2, 2, 1)
plt.imshow(img,cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(image2,cmap='gray')
plt.subplot(2,2, 3)
plt.imshow(image3,cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(image3_random,cmap='gray')
plt.show()
#
#
# # plt.imshow(image_new)
# # # plt.imshow(img)
# # plt.show()
# print('1')