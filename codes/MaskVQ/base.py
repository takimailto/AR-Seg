import os
import cv2
import bisect
import torch
import random
import numpy as np
#import albumentations
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False):

        self.root_dir = paths
        self.database = []
        if self.root_dir == "/data/chentao/DATA/LIDC/train":
            patient_dirs = glob(os.path.join(self.root_dir, 'images', '*')) + glob(os.path.join("/data/chentao/DATA/LIDC/val", 'images', '*'))
        else:
            patient_dirs = glob(os.path.join(self.root_dir, 'images', '*'))
        for i in range(len(patient_dirs)):
            img_paths = glob(os.path.join(patient_dirs[i], '*'))
            for j in range(len(img_paths)):
                datapoint = dict()
                img_path = img_paths[j]
                datapoint["image"] = img_path
                # get the corresponding ground truth labels
                gt_base_path = img_path.replace('images', 'gt')
                for l in range(4):
                    gt_path = gt_base_path.replace('.png', '_l{}.png'.format(l))
                    datapoint["gt_{}".format(l)] = gt_path
                self.database.append(datapoint)

        if "train" in self.root_dir:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.CenterCrop((128, 128)),  # 调整图像大小
                transforms.ToTensor(),         # 转换为 Tensor
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop((128, 128)),  # 调整图像大小
                transforms.ToTensor(),         # 转换为 Tensor
            ])

    def __len__(self):
        return len(self.database)
    
    def __getitem__(self, idx):
        """
        根据索引 idx 获取数据。
        :param idx: 索引。
        :return: 处理后的图像数据。
        """
        example = dict()
        l = random.choice(range(4))
        label_path = self.database[idx]["gt_{}".format(l)]
        label = Image.open(label_path).convert("L")  # 假设是灰度图像，如果是彩色图像可以去掉 .convert("L")
        image_path = self.database[idx]["image"]
        image = Image.open(image_path)
        state = torch.get_rng_state()
        image = self.transform(image)
        torch.set_rng_state(state)
        label = self.transform(label)
        image = cv2.resize(image.repeat(3, 1, 1).numpy().transpose(1, 2, 0), (1024, 1024), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
        image = (image - image.min()) / np.clip(image.max() - image.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3)
        assert np.max(image)<=1.0 and np.min(image)>=0.0, 'image should be normalized to [0, 1]'
        image = torch.from_numpy(image)
        label=torch.where(label > 0, 1, 0).float()
        example["label"] = label
        example["image"] = image
        
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
