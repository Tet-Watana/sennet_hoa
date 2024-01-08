# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset


class BloodVesselSegmentation(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=1,
                 multi_scale=True,
                 flip=True,
                 ignore_label=0,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(BloodVesselSegmentation, self).__init__(ignore_label, base_size,
                                                      crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {-1: ignore_label, 0: 0, 255: 1}
        self.class_weights = torch.FloatTensor([1, 1]).cuda()

        self.bd_dilate_size = bd_dilate_size

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                # v is the key, k is the value in the label_mapping dictionary.
                # k == 255 is the ignore label, because 255 does not exists in the label_mapping(v), nothing will be changed.
                # At k == 7, label at 7 will be changed to 0, which is the road class.
                # At k == 8, label at 8 will be changed to 1, which is the sidewalk class. and so on.
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    # __getitem__ is a method that is called when you use dataset[i]
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, 'blood_vessel_segmentation', item["img"]),
                           cv2.IMREAD_UNCHANGED)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root, 'blood_vessel_segmentation', item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        # This operation compresses the preds' dimension from (1, 19, 1024, 2048) to (1, 1024, 2048)
        # It acts like a nms operation which only keeps the class with the highest score
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
