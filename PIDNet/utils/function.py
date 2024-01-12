# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [
                          x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(), ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            losses, pred, _, _ = model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(
                config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
    # 1D array where each element is the sum of the corresponding column of the confusion matrix.
    # It is the total number of instances for GT classes.
    pos = confusion_matrix.sum(1)
    # 1D array where each element is the sum of the corresponding row of the confusion matrix.
    # It is the total number of instances for predicted classes.
    res = confusion_matrix.sum(0)
    # tp will be a 1D array where each element is the number of true positives for each class.
    tp = np.diag(confusion_matrix)
    # pixel accuracy: correctly classified pixels / total pixels
    pixel_acc = tp.sum()/pos.sum()
    # mean accuracy: correctly classified pixels / total pixels per class
    # Average of the accuracies of each class.
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    # mean IoU: Average of the IoUs of each class.
    # pos + res - tp computes the union of the predictex and GT instances for each class.
    # np.maximum(1.0, pos + res - tp) makes sure that the denominator is not 0.
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, ori_size, name = batch
            ori_size = ori_size[0]
            size_list = torch.tensor(image.shape).tolist()
            size = torch.tensor([size_list[-1], size_list[-2], size_list[-3]])
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            # if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
            #     pred = F.interpolate(
            #         pred, size[-2:],
            #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            #     )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name, ori_size)
