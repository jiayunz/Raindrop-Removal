import os
from tqdm import tqdm
import numpy as np
import cv2
import random
import re


class GenerateData():
    def __init__(self, input_dir, label_dir, patch_size=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.input_files = os.listdir(input_dir)
        self.patch_size = patch_size

    def align_to_four(self, img):
        a_row = int(img.shape[0] / 4) * 4
        a_col = int(img.shape[1] / 4) * 4
        img = img[0:a_row, 0:a_col]
        return img

    def read_data(self):
        self.data = []
        self.details = []
        self.labels = []
        input_files = os.listdir(self.input_dir)
        for f in tqdm(input_files, total=len(input_files)):
            f_info = re.split('_', f)
            if len(f_info) and f_info[0].isdigit():
                rainy = np.array(self.align_to_four(cv2.imread(self.input_dir + f)), dtype=np.float32)
                label = np.array(self.align_to_four(cv2.imread(self.label_dir + f_info[0] + '_clean' + f[-4:])), dtype=np.float32)
                if np.max(rainy) > 1.:
                    rainy = rainy / 255.
                if np.max(label) > 1.:
                    label = label / 255.
                # detail
                base = cv2.GaussianBlur(rainy, (41, 41), 0)
                detail = rainy - base

                self.data.append(np.array(rainy, dtype=np.float32))
                self.details.append(np.array(detail, dtype=np.float32))
                self.labels.append(np.array(label, dtype=np.float32))

    def next(self, batch_size):
        data = []
        details = []
        labels = []
        masks = []
        while len(data) < batch_size:
            r_idx = random.randint(0, len(self.input_files) - 1)
            f_info = re.split('_', self.input_files[r_idx])
            if not (len(f_info) and f_info[0].isdigit()):
                continue

            rainy = np.array(cv2.imread(self.input_dir + self.input_files[r_idx]), dtype=np.float32)
            label = np.array(cv2.imread(self.label_dir + f_info[0] + '_clean' + self.input_files[r_idx][-4:]), dtype=np.float32)
            if np.max(rainy) > 1.:
                rainy = rainy / 255.
            if np.max(label) > 1.:
                label = label / 255.
            mask = np.expand_dims(cv2.cvtColor(rainy - label, cv2.COLOR_BGR2GRAY), -1)

            # detail
            base = cv2.GaussianBlur(rainy, (41, 41), 0)
            detail = rainy - base

            if self.patch_size:
                x = random.randint(0, rainy.shape[0] - self.patch_size)
                y = random.randint(0, rainy.shape[1] - self.patch_size)

                rainy = rainy[x: x + self.patch_size, y: y + self.patch_size, :]
                label = label[x: x + self.patch_size, y: y + self.patch_size, :]
                detail = detail[x: x + self.patch_size, y: y + self.patch_size, :]
                mask = mask[x: x + self.patch_size, y: y + self.patch_size, :]

            data.append(rainy)
            labels.append(label)
            details.append(detail)
            masks.append(mask)

        data = np.array(data, dtype=np.float32)
        details = np.array(details, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        return data, details, labels, masks
