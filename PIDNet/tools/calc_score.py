import argparse
import os
import pandas as pd
import numpy as np
import cv2
import _init_paths
from utils.surface_dice_metric import score


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the ground truth to csv file')
    parser.add_argument('--gt_csv_path', default='./kidney_2_gt.csv',
                        type=str)
    parser.add_argument('--pred_csv_path', default='./submission.csv', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    gt_df = pd.read_csv(args.gt_csv_path)
    pred_df = pd.read_csv(args.pred_csv_path)
    mean_surface_dice = score(gt_df, pred_df, 'id', 'rle', 0.0)
    print(mean_surface_dice)
if __name__ == '__main__':
    main()
