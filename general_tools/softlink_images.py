import os
import shutil


def create_soft_links():
    labels_dir = "blood-vessel-segmentation/train/kidney_3_dense/labels"
    sparse_images_dir = "blood-vessel-segmentation/train/kidney_3_sparse/images"
    dense_images_dir = "blood-vessel-segmentation/train/kidney_3_dense/images"

    # Get the list of file names in the labels directory
    label_files = os.listdir(labels_dir)

    for label_file in label_files:
        # Check if the file exists in the sparse images directory
        sparse_image_file = os.path.join(sparse_images_dir, label_file)
        if os.path.isfile(sparse_image_file):
            # Copy the image to the dense images directory
            dense_image_file = os.path.join(dense_images_dir, label_file)
            shutil.copyfile(sparse_image_file, dense_image_file)


if __name__ == "__main__":
    create_soft_links()
