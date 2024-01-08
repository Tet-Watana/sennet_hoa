import os
import cv2
import multiprocessing

# Define the directories
train_dir = "data/blood_vessel_segmentation/train"
val_dir = "data/blood_vessel_segmentation/val"

# Define the target size
target_size = (1512, 1704)

# Function to resize and save a single image


def resize_and_save_image(image_path):
    # Open the image
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = cv2.resize(image, target_size)

    # Check if "label" is in image_path
    if "label" in image_path:
        # Convert resized_image to binary image
        _, resized_image = cv2.threshold(
            resized_image, 127, 255, cv2.THRESH_BINARY)

    # Save the resized image
    cv2.imwrite(image_path, resized_image)

# Function to process images in parallel


def process_images(directory):
    # Create a pool of worker processes
    pool = multiprocessing.Pool()

    # Iterate over the root, directories, and files in the directory
    for root, directories, files in os.walk(directory):
        # Iterate over the files
        for file in files:
            # Check if the file is an image
            if file.endswith(".tif"):
                # Process the image in parallel
                image_path = os.path.join(root, file)
                pool.apply_async(resize_and_save_image, (image_path,))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()


# Resize and save images in the train directory
process_images(train_dir)

# Resize and save images in the val directory
process_images(val_dir)
