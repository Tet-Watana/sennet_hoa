import cv2
import argparse
import os


def overlay_binary_image(color_image, binary_image):
    # Convert the binary image to a 3-channel image
    binary_image_3ch = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Create a mask by thresholding the binary image
    ret, mask = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Bitwise AND the color image and the inverted mask
    color_image_bg = cv2.bitwise_and(color_image, color_image, mask=mask_inv)

    # Bitwise OR the color image background and the binary image
    overlayed_image = cv2.bitwise_or(color_image_bg, binary_image_3ch)

    return overlayed_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--color_image_dir", type=str, default="blood-vessel-segmentation/train/kidney_3_sparse/images")
    parser.add_argument(
        "--binary_image_dir", type=str, default="blood-vessel-segmentation/train/kidney_3_sparse/labels")
    return parser.parse_args()


def main():
    args = parse_args()
    image_files = []
    for (root, dirs, files) in os.walk(args.color_image_dir):
        for file in files:
            if file.endswith(".tif"):
                image_files.append(os.path.join(root, file))
    image_files.sort()  # Sort the image files in ascending order
    current_image_index = 0
    total_images = len(image_files)

    while True:
        color_image = cv2.imread(image_files[current_image_index])
        binary_image = cv2.imread(os.path.join(
            args.binary_image_dir, os.path.basename(image_files[current_image_index])), cv2.IMREAD_GRAYSCALE)
        overlayed_image = overlay_binary_image(
            color_image, binary_image)

        # Create a resizable window
        cv2.namedWindow("Overlayed Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Overlayed Image", overlayed_image)
        cv2.resizeWindow("Overlayed Image", 800, 600)  # Set the window size

        key = cv2.waitKey(0)

        if key == ord('n'):
            current_image_index = (current_image_index + 1) % total_images
        elif key == ord('b'):
            current_image_index = (current_image_index - 1) % total_images
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
