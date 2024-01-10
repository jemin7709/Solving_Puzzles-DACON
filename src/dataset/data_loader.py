import json

import cv2
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transform=None):
        with open("configs/config.json", "r") as f:
            CFG = json.load(f)
        self.img_path_list = CFG["PATH"] + img_path_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        # PIL 이미지로 불러오기
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)

        if self.label_list is not None:
            label = torch.tensor(self.label_list[index], dtype=torch.long) - 1
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)
