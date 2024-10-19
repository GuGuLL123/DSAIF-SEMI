import os
import argparse
import torch
import yaml
from tqdm import tqdm
from model.VNet import VNet
from model.mcnet_3d import MCNet3d_v2
from model.unet_3D_dv_semi import unet_3D_dv_semi
from model.unet_3D import unet_3D
import h5py
from medpy import metric
import numpy as np
import math
import json
import torch.nn.functional as F
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
cfg = yaml.load(open(FLAGS.config, "r"), Loader=yaml.Loader)



num_classes = 2
if cfg['dataset'] == "LA_DATASET":
    patch_size = (112, 112, 80)
    with open(cfg['data_root'] + '/data/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [cfg['data_root']  + "/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                  image_list]
    # image_list = ['/data/ylgu/Medical/Semi_medical_data/LA_DATASET/data/2018LA_Seg_Training Set/VQ2L3WM8KEVF6L44E6G9/mri_norm2.h5']
elif cfg['dataset'] == "Pancreas_CT":
    patch_size = (96, 96, 96)
    with open(cfg['data_root']  + '/data/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [cfg['data_root']  + "/data/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    # image_list = ['/data/ylgu/Medical/Semi_medical_data/Pancreas_CT/data/Pancreas_h5/image0037_norm.h5']

def draw_mask(img_rgb, mask):
    overlay1 = img_rgb.copy()
    overlay1[mask == 1, 0] = 122
    overlay1[mask == 1, 1] = 190
    overlay1[mask == 1, 2] = 255
    # plt.imshow(overlay1)
    # plt.show()

    mask_color = overlay1.copy()
    mask_color[mask == 0, 0] = 0
    mask_color[mask == 0, 1] = 0
    mask_color[mask == 0, 2] = 0
    # plt.imshow(mask_color)
    # plt.show()
    # plt.imshow(overlay1)
    # plt.show()
    # overlay1 = 0.5 * overlay1 + 0.5 * img_rgb
    # cv2.addWeighted(overlay1, 0.1, img_rgb, 0.9, 0, img_rgb)
    return overlay1

def draw_all_case(num_outputs, model,model_name, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, metric_detail=1, up_down_inverse=0):
    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    slice_dice ={}
    # image_save_path = cfg['data_root'] + '/data/vis-final-10-new/image/'
    image_gt_save_path = cfg['data_root'] + '/data/vis-10-tmi/image_gt/'
    image_pre_save_path = cfg['data_root'] + '/data/vis-10-tmi/image_pre/'
    # image_score_save_path = cfg['data_root'] + '/data/vis-final-10/image_score_heat/'

    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        # data_name = image_path.split('/')[-2]
        data_name = image_path.split('/')[-1].split('.')[0]
        image = h5f['image'][:]
        label = h5f['label'][:]
        if up_down_inverse:
            image = -image
        prediction, score_map = draw_single_case_first_output(model, image, stride_xy, stride_z, patch_size,
                                                              num_classes=num_classes)
        for slice in range(image.shape[2]):
            prediction_slice = prediction[:, :, slice]
            # score_map_slice = score_map[:, :,:, slice]
            label_slice = label[:, :, slice]
            image_slice = np.expand_dims(image[:, :, slice],axis=2).repeat(3,axis=2)
            image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice)) * 255
            dice = calculate_metric_percase(prediction_slice, label_slice)
            slice_dice[data_name + '_' + str(slice)] = dice
            image_gt_save = draw_mask(image_slice, label_slice)
            image_pre_save = draw_mask(image_slice, prediction_slice)

            # prediction_slice = (prediction_slice-np.min(prediction_slice))/(np.max(prediction_slice)-np.min(prediction_slice)+1e-7)*255
            # score_map_slice = (score_map_slice-np.min(score_map_slice))/(np.max(score_map_slice)-np.min(score_map_slice)+1e-7)*255
            # score_map_slice = score_map_slice[0].astype(np.uint8)
            # score_map_slice = cv2.applyColorMap(score_map_slice, cv2.COLORMAP_JET)

            # label_slice = (label_slice-np.min(label_slice))/(np.max(label_slice)-np.min(label_slice)+1e-7)*255

            # if not os.path.exists(image_gt_save_path + '/' + data_name +'/'+ str(slice)+'/'):
            #     os.makedirs(image_gt_save_path + '/' + data_name +'/'+ str(slice)+'/')
            # if not os.path.exists(image_pre_save_path + '/' + data_name +'/'+ str(slice)+'/'):
            #     os.makedirs(image_pre_save_path + '/' + data_name +'/'+ str(slice)+'/')
            # if not os.path.exists(image_score_save_path + '/' + data_name +'/'+ str(slice)+'/'):
            #     os.makedirs(image_score_save_path + '/' + data_name +'/'+ str(slice)+'/')
            # if not os.path.exists(image_save_path + '/' + data_name +'/'+ str(slice)+'/'):
            #     os.makedirs(image_save_path + '/' + data_name +'/'+ str(slice)+'/')
            # cv2.imwrite(image_gt_save_path + '/' + data_name +'/' + str(slice)+'/'+'gt' + '.png', label_slice )
            # cv2.imwrite(image_pre_save_path + '/' + data_name + '/' + str(slice)+'/'+model_name + '.png', prediction_slice )
            # cv2.imwrite(image_score_save_path + '/' + data_name + '/' + str(slice)+'/'+model_name + '.png', score_map_slice )
            # cv2.imwrite(image_save_path + '/' + data_name + '/' + str(slice) + '/' + model_name + '.png',
            #             image_slice)


            if not os.path.exists(image_gt_save_path + '/' + data_name +'/'+ str(slice)+'/'):
                os.makedirs(image_gt_save_path + '/' + data_name +'/'+ str(slice)+'/')
            if not os.path.exists(image_pre_save_path + '/' + data_name +'/'+ str(slice)+'/'):
                os.makedirs(image_pre_save_path + '/' + data_name +'/'+ str(slice)+'/')
            cv2.imwrite(image_gt_save_path + '/' + data_name +'/' + str(slice)+'/'+'gt' + '.png', image_gt_save )
            cv2.imwrite(image_pre_save_path + '/' + data_name + '/' + str(slice)+'/'+model_name + '.png', image_pre_save )






    return slice_dice

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)


    return dice

def draw_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
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

def test_calculate_metric():
    our_net1 = VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=cfg['n_filters1'],
                  normalization='batchnorm', has_dropout=False)
    our_net2 = VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=cfg['n_filters2'],
                  normalization='batchnorm', has_dropout=False)
    our_net1.cuda()
    our_net2.cuda()
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    cfg['up_down_inverse'] = 0
    saved_mode_path1 = '/home/ylgu/experiments/semi_supervised_segmentation/Medical/differentmatch/exp_differentmatch/Pancreas_CT/10/consistency_0_bezier3_3_gama0.5_1.5_invert_0_rotate_1_resize_0_loss_0_ws_0_seed_1337_rampup1_40_tree_mode_1_tree_0_0_save/VNet_filter1_16_filter2_16/VNet_best_model1.pth'
    saved_mode_path2 = '/home/ylgu/experiments/semi_supervised_segmentation/Medical/differentmatch/exp_differentmatch/Pancreas_CT/10/consistency_0_bezier3_3_gama0.5_1.5_invert_0_rotate_1_resize_0_loss_0_ws_0_seed_1337_rampup1_40_tree_mode_1_tree_0_0_save/VNet_filter1_16_filter2_16/VNet_best_model2.pth'
    our_net1.load_state_dict(torch.load(saved_mode_path1), strict=False)
    our_net2.load_state_dict(torch.load(saved_mode_path2), strict=False)
    print('1'*50)
    print("init weight from {}".format(saved_mode_path1))
    print("init weight from {}".format(saved_mode_path2))
    our_net1.eval()
    our_net2.eval()




    cps_net = VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=cfg['n_filters1'],
                    normalization='batchnorm', has_dropout=False)
    cps_net.cuda()
    saved_cps_mode_path = '/home/ylgu/experiments/semi_supervised_segmentation/Medical/differentmatch/exp_differentmatch/Pancreas_CT/10/consistency_1_bezier-1_-1_gama0_0_invert_0_rotate_0_resize_0_loss_0_ws_0_seed_1337_rampup1_40/VNet_filter1_16_filter2_16/VNet_best_model1.pth'
    cps_net.load_state_dict(torch.load(saved_cps_mode_path), strict=False)
    print('2' * 50)
    print("init weight from {}".format(saved_cps_mode_path))
    cps_net.eval()


    mcnet = MCNet3d_v2(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], normalization='batchnorm', has_dropout=False).cuda()
    saved_mcnet_mode_path = '/home/ylgu/experiments/semi_supervised_segmentation/Medical/MC-Net/model/Pancreas_CT/MCnet_1_6/mcnet3d_v2/mcnet3d_v2_best_model.pth'
    mcnet.load_state_dict(torch.load(saved_mcnet_mode_path), strict=False)
    print('3' * 50)
    print("init weight from {}".format(saved_mcnet_mode_path))
    mcnet.eval()


    urpc_net = unet_3D_dv_semi(n_classes=cfg['nclass'], in_channels=cfg['image_channel']).cuda()
    saved_urpc_mode_path = '/home/ylgu/experiments/semi_supervised_segmentation/Medical/SSL4MIS-master/model/Pancreas_CT/URPC-1_6/unet_3D_dv_semi/unet_3D_dv_semi_best_model.pth'
    urpc_net.load_state_dict(torch.load(saved_urpc_mode_path), strict=False)
    print('4' * 50)
    print("init weight from {}".format(saved_urpc_mode_path))
    urpc_net.eval()


    uamt_net = unet_3D(n_classes=cfg['nclass'], in_channels=cfg['image_channel']).cuda()
    saved_uamt_mode_path = '/home/ylgu/experiments/semi_supervised_segmentation/Medical/SSL4MIS-master/model/Pancreas_CT/UAMT-1_6/unet_3D/unet_3D_best_model.pth'
    uamt_net.load_state_dict(torch.load(saved_uamt_mode_path), strict=False)
    print('5' * 50)
    print("init weight from {}".format(saved_uamt_mode_path))
    uamt_net.eval()


    net_dic = {"our_net1": our_net1,"our_net2": our_net2,"cps_net":cps_net,"mcnet":mcnet, "urpc_net":urpc_net,"uamt_net":uamt_net}
    # net_dic = {"our_net1": our_net1, "our_net2": our_net2,"mcnet":mcnet}
    final_dic = {}
    if cfg['dataset'] == "LA_DATASET":
        for key_net in net_dic:

            key_net_dic = draw_all_case(1, net_dic[key_net],key_net, image_list, num_classes=num_classes,
                                       patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                       metric_detail=1,up_down_inverse=0)
            final_dic[key_net] = key_net_dic
        # avg_metric2 = draw_all_case( 1, our_net2, image_list, num_classes=num_classes,
        #                            patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
        #                            metric_detail=1,up_down_inverse=cfg['up_down_inverse'])
    elif cfg['dataset'] == "Pancreas_CT":
        for key_net in net_dic:
            key_net_dic = draw_all_case( 1, net_dic[key_net],key_net, image_list, num_classes=num_classes,
                                       patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                       metric_detail=1, up_down_inverse=0)
            final_dic[key_net] = key_net_dic
        # avg_metric2 = draw_all_case( 1, our_net2, image_list, num_classes=num_classes,
        #                            patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
        #                            metric_detail=1, up_down_inverse=cfg['up_down_inverse'])

    different_iou_dir = {}
    for key_silce in final_dic["our_net1"]:
        our_reslut = max(final_dic["our_net1"][key_silce],final_dic["our_net2"][key_silce])
        other_reslut = max(final_dic["cps_net"][key_silce],final_dic["mcnet"][key_silce],final_dic["urpc_net"][key_silce],final_dic["uamt_net"][key_silce])
        # other_reslut = max(0, final_dic["mcnet"][key_silce])
        different_iou_dir[key_silce] = our_reslut - other_reslut
        sorted_by_value = sorted(different_iou_dir.items(), key=lambda x: x[1])
    with open("/data/ylgu/Medical/Semi_medical_data/Pancreas_CT/data/vis-10-tmi/Pancreas_CT_10_iou_sort.txt", "w") as f:
        json.dump(sorted_by_value, f)
    return 0


if __name__ == '__main__':
    avg_metric1 = test_calculate_metric()
    print(avg_metric1)
    print('-'*50)