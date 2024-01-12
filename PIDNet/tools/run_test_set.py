# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import logging
import timeit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/blood_vessel_seg/pidnet_small_blood_vessel_seg.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--model_path', default='output/blood_vessel_segmentation/pidnet_small_blood_vessel_seg/best.pt')
    parser.add_argument('--test_lst_path', default='data/list/blood_vessel_seg/test.lst')
    parser.add_argument('--out_dir', default='./output/blood_vessel_segmentation/pidnet_small_blood_vessel_seg/test_results')
    parser.add_argument('--test_data_root', default='./kaggle/input/blood-vessel-segmentation/')
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)
    pretrained_dict = torch.load(args.model_path)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # eval('datasets.'+config.DATASET.DATASET)() is equivalent to datasets.cityscapes() class.
    # This initializes the Cityscapes class in datasets/cityscapes.py.
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=args.test_data_root,
        list_path=args.test_lst_path,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    if ('test' in config.DATASET.TEST_SET) and ('blood' in config.DATASET.DATASET):
        df = test(config,
                 test_dataset,
                 testloader,
                 model,
                 sv_dir=args.out_dir)
        df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
