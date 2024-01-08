import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import argparse


def create_submission(args):
    df = pd.DataFrame(columns=['id', 'rle'])
    out_data_all = []
    rle_sample = '1 0'
    for dirname, _, filenames in os.walk(args.test_set_path):
        for filename in filenames:
            dataset_name = os.path.basename(os.path.dirname(dirname))
            slice_name = str(filename.replace('.tif', ''))
            out_id = dataset_name + '_' + slice_name
            out_data = [out_id, rle_sample]
            out_data_all.append(out_data)

    df = pd.DataFrame(out_data_all, columns=['id', 'rle'])
    df.to_csv(args.csv_out_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Create submission file')
    parser.add_argument('--test_set_path', type=str, help='Test set path',
                        default='./blood-vessel-segmentation/test')
    parser.add_argument('--csv_out_path', type=str, help='Csv out path',
                        default='./submission/submission.csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    create_submission(args)
    return None


if __name__ == '__main__':
    main()
