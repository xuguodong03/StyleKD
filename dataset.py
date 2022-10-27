from io import BytesIO
import os

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class FFHQ_Dataset(Dataset):
    '''
    Usage:
        Self-coded class for loading the FFHQ data
    '''

    def __init__(self, image_folder, transform = None):
        images_list = os.listdir(image_folder)
        self.images_list = sorted([os.path.join(image_folder, image) for image in images_list])
        print(len(images_list))
        print(images_list[:10])
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
