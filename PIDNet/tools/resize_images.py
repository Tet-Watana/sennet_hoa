import os
import cv2
import multiprocessing
import argparse


class ImageResizer:
    def __init__(self, target_size, output_dir):
        self.target_size = target_size
        self.output_dir = output_dir

    def resize_and_save_image(self, image_path):
        # Open the image
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, self.target_size)

        # Check if "label" is in image_path
        if "label" in image_path:
            # Convert resized_image to binary image
            _, resized_image = cv2.threshold(
                resized_image, 127, 255, cv2.THRESH_BINARY)

        parent_dir = os.path.basename(
            os.path.dirname(image_path))  # images or labels
        gparent_dir = os.path.basename(os.path.dirname(
            os.path.dirname(image_path)))  # kidney1_dense,
        ggparent_dir = os.path.basename(os.path.dirname(
            os.path.dirname(os.path.dirname(image_path))))  # train, val
        # Get the output image path
        output_image_path = os.path.join(
            self.output_dir, ggparent_dir, gparent_dir, parent_dir, os.path.basename(image_path))
        # print(output_image_path)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # Save the resized image
        cv2.imwrite(output_image_path, resized_image)

    def process_images(self, directory):
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
                    pool.apply_async(self.resize_and_save_image, (image_path,))

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images")
    parser.add_argument("--train_dir", type=str, default="data/blood_vessel_segmentation_original/train",
                        help="Directory containing training images")
    parser.add_argument("--val_dir", type=str, default="data/blood_vessel_segmentation_original/val",
                        help="Directory containing validation images")
    parser.add_argument("--width", type=int, default=1512,
                        help="Target width of the resized images")
    parser.add_argument("--height", type=int, default=1704,
                        help="Target height of the resized images")
    parser.add_argument("--output_dir", type=str, default="data/blood_vessel_segmentation/",
                        help="Directory to save the resized images")

    args = parser.parse_args()

    target_size = (args.width, args.height)

    resizer = ImageResizer(target_size, args.output_dir)

    # Resize and save images in the train directory
    resizer.process_images(args.train_dir)

    # Resize and save images in the val directory
    resizer.process_images(args.val_dir)
