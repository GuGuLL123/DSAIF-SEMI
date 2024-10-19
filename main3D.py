import argparse
from copy import deepcopy
import logging
import os
import pprint
import random
import numpy as np
import torch
import shutil
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
from torchvision import transforms
from dataset.semi import SemiDataset_Medical_Pancreas_CT
from model.VNet import VNet
from val_3D import validation_3D
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from util.losses import DiceLoss
import time
from medpy import metric

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--manual_seed', default=1337, type=int)


def sigmoid_rampup(current_epoch,rampup_weight, rampup_length=200):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current_epoch, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return rampup_weight *float(np.exp(-5.0 * phase * phase))

def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def main():


    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)


    labeled_id_path = './partitions/'+cfg['dataset']+'/1_{}/'.format(cfg['labeled_rate'])+'labeled.list'
    unlabeled_id_path = './partitions/' + cfg[
        'dataset'] + '/1_{}/'.format(cfg['labeled_rate']) + 'unlabeled.list'

    snapshot_path = "./exp_DSAIF-SEMI/{}/{}/tree_mode_{}_tree_{}_{}_dsaif_final/{}".format(
        cfg['dataset'],
        cfg['labeled_rate'],
        cfg['tree_mode'],cfg['tree_areathrehold'],cfg['tree_topoprob'],
        cfg['model'])


    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if args.manual_seed is not None:
        init_seeds(args.manual_seed)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank ==0:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)




    model1 = VNet(n_channels = cfg['image_channel'],n_classes = cfg['nclass'],n_filters=cfg['n_filters1'],normalization='batchnorm', has_dropout=True)
    model2 = VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'],n_filters=cfg['n_filters2'],normalization='batchnorm', has_dropout=True)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model1)))

    optimizer1 = SGD(model1.parameters(), lr=cfg['lr'],momentum=0.9, weight_decay=0.0001)
    optimizer2 = SGD(model2.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)

    local_rank = int(os.environ["LOCAL_RANK"])
    model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1)
    model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2)
    model1.cuda()
    model2.cuda()


    model1 = torch.nn.parallel.DistributedDataParallel(model1, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)


    ce_loss = nn.CrossEntropyLoss().cuda(local_rank)
    dice_loss= DiceLoss(cfg['nclass']).cuda(local_rank)


    trainset_u = SemiDataset_Medical_Pancreas_CT(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], unlabeled_id_path,
                            tree_mode =cfg['tree_mode'], tree_areathrehold=cfg['tree_areathrehold'],tree_topoprob=cfg['tree_topoprob'])
    trainset_l = SemiDataset_Medical_Pancreas_CT(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], labeled_id_path, nsample=len(trainset_u.ids),
                            tree_mode =cfg['tree_mode'], tree_areathrehold=cfg['tree_areathrehold'],tree_topoprob=cfg['tree_topoprob'])
    valset = SemiDataset_Medical_Pancreas_CT(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    previous_best1 = 0.0
    previous_best2 = 0.0
    iter_num = 0
    max_iterations = cfg['max_iterations']
    max_epoch = max_iterations // len(trainloader_l) + 1


    for epoch in range(max_epoch):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best1: {:.2f},Previous best2: {:.2f}'.format(
                epoch, optimizer1.param_groups[0]['lr'], previous_best1,previous_best2))


        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        for i, (labeled_sampled,unlabeled_sampled) in enumerate(loader):

            labeled_sampled_similar, labeled_sampled_dissimilar, labeled_label_batch,rotate_labeled = labeled_sampled['image_similar'], labeled_sampled['image_dissimilar'], labeled_sampled['label'], labeled_sampled['rotate_angle']
            labeled_sampled_similar, labeled_sampled_dissimilar, labeled_label_batch = labeled_sampled_similar.cuda(), labeled_sampled_dissimilar.cuda(), labeled_label_batch.cuda()
            unlabeled_sampled_similar, unlabeled_sampled_dissimilar,unlabeled_label_batch, rotate_unlabeled = unlabeled_sampled['image_similar'], unlabeled_sampled['image_dissimilar'], unlabeled_sampled['label'], unlabeled_sampled['rotate_angle']
            unlabeled_sampled_similar, unlabeled_sampled_dissimilar, unlabeled_label_batch = unlabeled_sampled_similar.cuda(), unlabeled_sampled_dissimilar.cuda(), unlabeled_label_batch.cuda()
            raw_size = labeled_sampled_similar.shape[2:]


            model1.train()
            model2.train()

            labeled_output1 = model1(labeled_sampled_similar)
            labeled_output1_soft = torch.softmax(labeled_output1, dim=1)
            unlabeled_output1 = model1(unlabeled_sampled_similar)
            unlabeled_output1_soft = torch.softmax(unlabeled_output1, dim=1)


            labeled_output2_ = model2(labeled_sampled_dissimilar)
            unlabeled_output2_ = model2(unlabeled_sampled_dissimilar)

            
            ##########################changed code
            labeled_output2 = torch.zeros_like(labeled_output2_)
            unlabeled_output2 = torch.zeros_like(unlabeled_output2_)

            for batch_index in range(labeled_output2.shape[0]):
                rotate_label_now = int(rotate_labeled[batch_index].item())
                labeled_output2_now = transforms.functional.rotate(labeled_output2_[batch_index], -rotate_label_now, resample=False, expand=False, center=None)
                labeled_output2[batch_index] = labeled_output2_now
            for batch_index in range(unlabeled_output2.shape[0]):
                rotate_unlabel_now = int(rotate_unlabeled[batch_index].item())
                unlabeled_output2_now = transforms.functional.rotate(unlabeled_output2_[batch_index], -rotate_unlabel_now, resample=False, expand=False, center=None)
                unlabeled_output2[batch_index] = unlabeled_output2_now



            labeled_output2_soft = torch.softmax(labeled_output2, dim=1)
            unlabeled_output2_soft = torch.softmax(unlabeled_output2, dim=1)

            consistency_weight = sigmoid_rampup(iter_num // 150,rampup_weight=cfg['rampup_weight'],rampup_length = cfg['rampup_length'] )

            pseudo_unlabeled_output1 = torch.argmax(unlabeled_output1_soft.detach(), dim=1, keepdim=False)
            pseudo_unlabeled_output2= torch.argmax(unlabeled_output2_soft.detach(), dim=1, keepdim=False)



            loss1 = 0.5 * (ce_loss(labeled_output1, labeled_label_batch[:].long()) + dice_loss(labeled_output1_soft, labeled_label_batch.unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(labeled_output2, labeled_label_batch[:].long()) + dice_loss(labeled_output2_soft, labeled_label_batch.unsqueeze(1)))


            pseudo_supervision1 = ce_loss(unlabeled_output1, pseudo_unlabeled_output2)
            pseudo_supervision2 = ce_loss(unlabeled_output2, pseudo_unlabeled_output1)



            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2
            loss = model1_loss + model2_loss

            torch.distributed.barrier()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            iter_num = iter_num + 1

            lr_ = cfg['lr'] * (1 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_



            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss1: {:.3f}, '
                            'Loss2: {:.3f}'.format(
                    iter_num, loss / (i+1), model1_loss / (i+1), model2_loss / (i+1)))



            # validation
            if iter_num > 800 and iter_num % 200 == 0:
                model1.eval()
                model2.eval()

                avg_dice1 = validation_3D(model1, valloader,cfg,stride_xy=16,stride_z=16, patch_size=cfg['crop_size'])
                avg_dice2 = validation_3D(model2, valloader,cfg,stride_xy=16,stride_z=16, patch_size=cfg['crop_size'])


                if avg_dice1 > previous_best1 and rank == 0:
                    previous_best1 = avg_dice1
                    save_mode_path1 = os.path.join(snapshot_path,'model1_iter_{}_dice_{}.pth'.format(iter_num, round(previous_best1, 4)))
                    save_best1 = os.path.join(snapshot_path,'{}_best_model1.pth'.format(cfg['model']))
                    torch.save(model1.module.state_dict(), save_mode_path1)
                    torch.save(model1.module.state_dict(), save_best1)

                if avg_dice2 > previous_best2 and rank == 0:
                    previous_best2 = avg_dice2
                    save_mode_path2 = os.path.join(snapshot_path,'model2_iter_{}_dice_{}.pth'.format(iter_num, round(previous_best2, 4)))
                    save_best2 = os.path.join(snapshot_path,'{}_best_model2.pth'.format(cfg['model']))
                    torch.save(model2.module.state_dict(), save_mode_path2)
                    torch.save(model2.module.state_dict(), save_best2)
                model1.train()
                model2.train()


            if iter_num % 3000 == 0 and rank == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.module.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.module.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))
            if iter_num >= max_iterations:
                save_mode_path1 = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_1.pth')
                save_mode_path2 = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_2.pth')
                torch.save(model1.module.state_dict(), save_mode_path1)
                torch.save(model2.module.state_dict(), save_mode_path2)
                logging.info("save model1 to {}".format(save_mode_path1))
                logging.info("save model2 to {}".format(save_mode_path2))
                break
        if iter_num >= max_iterations:
            break



if __name__ == '__main__':
    main()
