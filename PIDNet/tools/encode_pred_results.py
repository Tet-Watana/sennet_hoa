import os
import pandas as pd
import cv2
import numpy as np
import argparse
import _init_paths
from utils.surface_dice_metric import score, rle_encode


def parse_args():
    parser = argparse.ArgumentParser(description='Encode predicted results')

    parser.add_argument('--pred_dir',
                        help='Path to a directory with prediction results',
                        default="output/blood_vessel_segmentation/pidnet_small_blood_vessel_seg/test_results",
                        type=str)
    parser.add_argument('--out_csv_path', default = 'submission.csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pred_dir = args.pred_dir
    df = pd.DataFrame(columns=['id', 'rle'])
    for root, _, files in os.walk(pred_dir):
        for file in files:
            if file.endswith(".tif"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = np.clip(img, 0, 1)
                rle = rle_encode(img)
                img_name = str(os.path.splitext(os.path.basename(img_path))[0])
                img_dir = os.path.basename(root)
                img_id = img_dir + '_' + img_name
                tmp_df = pd.DataFrame([[img_id, rle]], columns=['id', 'rle'])
                df = pd.concat([df, tmp_df], ignore_index=True)
    df.to_csv(args.out_csv_path, index=False)


if __name__ == '__main__':
    main()
