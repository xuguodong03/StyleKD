import os
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, Union

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        raw_cifar: Union[CIFAR10, Subset], # CIFAR10 or subset of it
        transform: Optional[Callable] = None,
        albumentations: bool = True,
    ) -> None:
        super().__init__()
        self.raw_cifar = raw_cifar
        self.transform = transform
        self.albumentations = albumentations

    def __len__(self) -> int:
        return len(self.raw_cifar)

    def __getitem__(self, idx):
        # this dataset works with albumentations transforms
        image, label = self.raw_cifar[idx]
        # be careful! image is not a numpy array

        if self.transform is not None:
            image = np.array(image)
            if self.albumentations:
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                image = self.transform(image)

        return image


def prepare_cifar_datasets(trainval_raw_dataset, test_raw_dataset):
    val_size = 0.2

    train_indices, val_indices = train_test_split(
        np.arange(len(trainval_raw_dataset)),
        test_size=val_size,
    )
    train_raw_subset = Subset(trainval_raw_dataset, train_indices)
    val_raw_subset = Subset(trainval_raw_dataset, val_indices)

    train_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset = CIFAR10Dataset(train_raw_subset, transform=train_transforms)
    val_dataset = CIFAR10Dataset(val_raw_subset, transform=val_transforms)

    test_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_dataset = CIFAR10Dataset(test_raw_dataset, transform=test_transforms)

    return train_dataset, val_dataset, test_dataset


class FFHQ_Dataset(Dataset):
    '''
    Usage:
        Self-coded class for loading the FFHQ data
    '''

    def __init__(self, image_folder, transform = None):
        images_list = os.listdir(image_folder)
        self.images_list = sorted([os.path.join(image_folder, image) for image in images_list])
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.images_list[index]
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images_list)


class church_dataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        EXTS = ['jpg', 'jpeg', 'png']
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
