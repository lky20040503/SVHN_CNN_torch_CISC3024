import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from torchvision import datasets
from scipy.io import loadmat


class augmentation_dataset(datasets.SVHN):
    def __init__(self,root = None,split='train',download=False,transform = None):
        super(augmentation_dataset,self).__init__(root=root,split = split, download=download, transform=transform)

    def __getitem__(self, idx):
        image, label = self.data[idx], int(self.labels[idx])
        image = np.transpose(image, (1, 2, 0))

        if self.transform is not None:
            augmented = self.transform(image = image)
            image = augmented['image']

        return image, label



