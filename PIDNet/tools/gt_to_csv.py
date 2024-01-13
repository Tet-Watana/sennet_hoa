import argparse
import os
import pandas as pd
import numpy as np
import cv2
import _init_paths
from utils.surface_dice_metric import rle_encode


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the ground truth to csv file')
    parser.add_argument('--gt_dir', default='./kaggle/input/blood-vessel-segmentation/test/kidney_2/labels',
                        type=str)
    parser.add_argument('--csv_out_path', default='./kidney_2_gt.csv', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    df = pd.DataFrame(columns=['id', 'rle', 'width', 'height'])
    for root, _, files in os.walk(args.gt_dir):
        files.sort()
        for file in files:
            if file.endswith(".tif"):
                img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                img = np.clip(img, 0, 1)
                height, width = img.shape[:2]
                rle = rle_encode(img)
                kidney = root.split('/')[-2]
                id = kidney + '_' + os.path.splitext(file)[0]
                tmp_df = pd.DataFrame([[id, rle, width, height]], columns=['id', 'rle', 'width', 'height'])
                df = pd.concat([df, tmp_df], ignore_index=True)
    df.to_csv(args.csv_out_path, index=False)

if __name__ == '__main__':
    main()
