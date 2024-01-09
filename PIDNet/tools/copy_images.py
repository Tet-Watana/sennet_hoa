import os
import shutil
import argparse


class ImageCopier:
    def __init__(self, labels_dir, sparse_images_dir, dense_images_dir):
        self.labels_dir = labels_dir
        self.sparse_images_dir = sparse_images_dir
        self.dense_images_dir = dense_images_dir

    def copy_images(self):
        # Get the list of file names in the labels directory
        label_files = os.listdir(self.labels_dir)
        os.makedirs(self.dense_images_dir, exist_ok=True)

        for label_file in label_files:
            # Check if the file exists in the sparse images directory
            sparse_image_file = os.path.join(
                self.sparse_images_dir, label_file)
            if os.path.isfile(sparse_image_file):
                # Copy the image to the dense images directory
                dense_image_file = os.path.join(
                    self.dense_images_dir, label_file)
                shutil.copyfile(sparse_image_file, dense_image_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy images from sparse directory to dense directory")
    parser.add_argument("--labels_dir", type=str,
                        default="data/blood_vessel_segmentation_original/train/kidney_3_dense/labels",
                        help="Path to the labels directory")
    parser.add_argument("--sparse_images_dir", type=str,
                        default="data/blood_vessel_segmentation_original/train/kidney_3_sparse/images",
                        help="Path to the sparse images directory")
    parser.add_argument("--dense_images_dir", type=str,
                        default="data/blood_vessel_segmentation_original/train/kidney_3_dense/images",
                        help="Path to the dense images directory")
    args = parser.parse_args()

    copier = ImageCopier(
        args.labels_dir, args.sparse_images_dir, args.dense_images_dir)
    copier.copy_images()
