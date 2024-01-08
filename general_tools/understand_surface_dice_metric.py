from surface_dice_metric import score, rle_encode, rle_decode
import pandas as pd
import cv2
import numpy as np


#####################################################
# Checking the score function.
# No groups (2D images).
solution = pd.DataFrame({
    'id': [0, 1],
    'rle': ['1 12 20 2', '1 6'],
    'width': [5, 5],
    'height': [5, 5],
})
# Perfect submission.
submission = pd.DataFrame({
    'id': [0, 1],
    'rle': ['1 12 20 2', '1 6'],
})
# print(solution)
mean_surface_dice = score(solution, submission, 'id', 'rle', 0.0)
# print(mean_surface_dice)

#####################################################
# Encoding a masked image to rle.
img = cv2.imread(
    "blood-vessel-segmentation/train/kidney_1_dense/labels/0054.tif", cv2.IMREAD_GRAYSCALE)
img = np.clip(img, 0, 1)
print(img.shape)
print(np.max(img))
print(np.min(img))
rle = rle_encode(img)
print(rle)

#####################################################
# Decoding rle to a masked image.
masked_img = rle_decode(rle, (img.shape[0], img.shape[1]))*255
print(masked_img.shape)
print(np.max(masked_img))
print(np.min(masked_img))
cv2.imwrite("test.png", masked_img)
