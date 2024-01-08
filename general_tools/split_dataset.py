import os
import random
import shutil


# Define the paths
dataset_path = "ViT-Adapter/segmentation/datasets/coco_stuff10k/"
images_path = os.path.join(dataset_path, "images_ori")
annotations_path = os.path.join(dataset_path, "annotations_ori")
train_images_path = os.path.join(dataset_path, "images/train")
val_images_path = os.path.join(dataset_path, "images/val")
test_images_path = os.path.join(dataset_path, "images/test")
train_annotations_path = os.path.join(dataset_path, "annotations/train")
val_annotations_path = os.path.join(dataset_path, "annotations/val")
test_annotations_path = os.path.join(dataset_path, "annotations/test")

# Create the directories if they don't exist
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(train_annotations_path, exist_ok=True)
os.makedirs(val_annotations_path, exist_ok=True)
os.makedirs(test_annotations_path, exist_ok=True)

# Get the list of image and annotation files
image_files = os.listdir(images_path)
annotation_files = os.listdir(annotations_path)

# Set random seed
random.seed(0)

# Sort image_files and annotation_files in ascending order
image_files.sort()
annotation_files.sort()

# Shuffle the index of image_files
index_list = list(range(len(image_files)))
random.shuffle(index_list)

# Calculate the split sizes
total_files = len(image_files)
train_size = int(total_files * 0.8)
val_size = int(total_files * 0.1)
test_size = total_files - train_size - val_size

# Split the files based on index_list
train_images = [image_files[i] for i in index_list[:train_size]]
val_images = [image_files[i]
              for i in index_list[train_size:train_size+val_size]]
test_images = [image_files[i] for i in index_list[train_size+val_size:]]

train_annotations = [annotation_files[i] for i in index_list[:train_size]]
val_annotations = [annotation_files[i]
                   for i in index_list[train_size:train_size+val_size]]
test_annotations = [annotation_files[i]
                    for i in index_list[train_size+val_size:]]

# Move the files to the respective directories
for image_file, annotation_file in zip(train_images, train_annotations):
    shutil.move(os.path.join(images_path, image_file),
                os.path.join(train_images_path, image_file))
    shutil.move(os.path.join(annotations_path, annotation_file),
                os.path.join(train_annotations_path, annotation_file))

for image_file, annotation_file in zip(val_images, val_annotations):
    shutil.move(os.path.join(images_path, image_file),
                os.path.join(val_images_path, image_file))
    shutil.move(os.path.join(annotations_path, annotation_file),
                os.path.join(val_annotations_path, annotation_file))

for image_file, annotation_file in zip(test_images, test_annotations):
    shutil.move(os.path.join(images_path, image_file),
                os.path.join(test_images_path, image_file))
    shutil.move(os.path.join(annotations_path, annotation_file),
                os.path.join(test_annotations_path, annotation_file))
