import os
import argparse
import torch
import yaml
from util.test_patch import test_all_case
from model.VNet import VNet

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/Pancreas_CT.yaml')
parser.add_argument('--gpu', type=str, default='4', help='GPU to use')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
cfg = yaml.load(open(FLAGS.config, "r"), Loader=yaml.Loader)



num_classes = 2

if cfg['dataset'] == "Pancreas_CT":
    patch_size = (96, 96, 96)
    with open(cfg['data_root']  + '/data/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [cfg['data_root']  + "/data/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]




def test_calculate_metric():
    net1 = VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=cfg['n_filters1'],
                  normalization='batchnorm', has_dropout=False)
    net2 = VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=cfg['n_filters2'],
                  normalization='batchnorm', has_dropout=False)
    net1.cuda()
    net2.cuda()
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    cfg['up_down_inverse'] = 0
    saved_mode_path1 = './exp_DSAIF-SEMI/Pancreas_CT/5/tree_mode_0_tree_0_0_dsaif_final/VNet/VNet_best_model1.pth'
    saved_mode_path2 = './exp_DSAIF-SEMI/Pancreas_CT/5/tree_mode_0_tree_0_0_dsaif_final/VNet/VNet_best_model2.pth'
    net1.load_state_dict(torch.load(saved_mode_path1), strict=False)
    net2.load_state_dict(torch.load(saved_mode_path2), strict=False)
    print("init weight from {}".format(saved_mode_path1))
    print("init weight from {}".format(saved_mode_path2))
    net1.eval()
    net2.eval()


    if cfg['dataset'] == "Pancreas_CT":
        avg_metric1 = test_all_case( 1, net1, image_list, num_classes=num_classes,
                                   patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                   metric_detail=1, nms=FLAGS.nms,up_down_inverse=0)
        avg_metric2 = test_all_case( 1, net2, image_list, num_classes=num_classes,
                                   patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                   metric_detail=1, nms=FLAGS.nms,up_down_inverse=cfg['up_down_inverse'])

    return avg_metric1, avg_metric2


if __name__ == '__main__':
    avg_metric1, avg_metric2 = test_calculate_metric()
    print(avg_metric1)
    print('-'*50)
    print(avg_metric2)