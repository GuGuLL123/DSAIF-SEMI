import numpy as np
import torch
import math
from medpy import metric
from scipy.ndimage import zoom
from imgaug import augmenters as iaa
import torch.nn.functional as F

def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map





def validation_3D(net,val_loader, cfg, stride_xy, stride_z,patch_size):

    total_dice = 0.0
    for i_batch, sampled_batch in enumerate(val_loader):
        prediction, score_map = test_single_case_first_output(
            net,  sampled_batch["image"][0], stride_xy, stride_z, patch_size,num_classes =cfg['nclass'])
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, sampled_batch["label"].cpu().detach().numpy())
        total_dice += dice
    total_dice = torch.from_numpy(np.array(total_dice)).cuda()
    torch.distributed.all_reduce(total_dice)
    total_dice = total_dice.cpu().numpy()
    avg_dice = total_dice / (len(val_loader)*2)
    # print('average metric is {}'.format(avg_dice))

    return avg_dice



def epoch_view_3D(net1,net2,view_loader, cfg, stride_xy, stride_z,patch_size,up_down_inverse=0,mode='gt'):

    total_dice1 = 0.0
    total_dice2 = 0.0
    total_dice3 = 0.0
    if mode == 'gt':
        for i_batch, sampled_batch in enumerate(view_loader):
            prediction1, score_map1 = test_single_case_first_output(
                net1,  sampled_batch["image"][0], stride_xy, stride_z, patch_size,num_classes =cfg['nclass'],up_down_inverse=up_down_inverse)
            prediction2, score_map2 = test_single_case_first_output(
                net2,  sampled_batch["image"][0], stride_xy, stride_z, patch_size,num_classes =cfg['nclass'],up_down_inverse=up_down_inverse)
            if np.sum(prediction1) == 0:
                dice1 = 0
            else:
                dice1 = metric.binary.dc(prediction1, sampled_batch["label"].cpu().detach().numpy())
            if np.sum(prediction2) == 0:
                dice2 = 0
            else:
                dice2 = metric.binary.dc(prediction2, sampled_batch["label"].cpu().detach().numpy())
            total_dice1 += dice1
            total_dice2 += dice2
    else:
        for i_batch, sampled_batch in enumerate(view_loader):
            prediction1, score_map1 = test_single_case_first_output(
                net1,  sampled_batch["image"][0], stride_xy, stride_z, patch_size,num_classes =cfg['nclass'],up_down_inverse=up_down_inverse)
            prediction2, score_map2 = test_single_case_first_output(
                net2,  sampled_batch["image"][0], stride_xy, stride_z, patch_size,num_classes =cfg['nclass'],up_down_inverse=up_down_inverse)
            error_map1 = np.abs(prediction1 - sampled_batch["label"].cpu().detach().numpy())
            error_map2 = np.abs(prediction2 - sampled_batch["label"].cpu().detach().numpy())
            right_map1 = 1 - error_map1
            right_map2 = 1 - error_map2

            error_dice = metric.binary.dc(error_map1, error_map2)
            right_dice = metric.binary.dc(right_map1, right_map2)
            cross_dice = metric.binary.dc(prediction1, prediction2)
            total_dice1 += error_dice
            total_dice2 += right_dice
            total_dice3 += cross_dice
    total_dice1 = torch.from_numpy(np.array(total_dice1)).cuda()
    total_dice2 = torch.from_numpy(np.array(total_dice2)).cuda()
    total_dice3 = torch.from_numpy(np.array(total_dice3)).cuda()
    torch.distributed.all_reduce(total_dice1)
    torch.distributed.all_reduce(total_dice2)
    torch.distributed.all_reduce(total_dice3)

    total_dice1 = total_dice1.cpu().numpy()
    total_dice2 = total_dice2.cpu().numpy()
    total_dice3 = total_dice3.cpu().numpy()
    avg_dice1 = total_dice1 / (len(view_loader)*2)
    avg_dice2 = total_dice2 / (len(view_loader)*2)
    avg_dice3 = total_dice3 / (len(view_loader)*2)
    # print('average metric is {}'.format(avg_dice))

    return avg_dice1,avg_dice2,avg_dice3