from astroNN.datasets import load_galaxy10
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import torch
from torch.utils.data import Dataset

class GalaxyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.permute(data,(0,3,1,2))
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return {'data': data, 'label': label}


def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def load_images_labels(step=1):
    # To load images and labels (will download automatically at the first time)
    # First time downloading location will be ~/.astroNN/datasets/
    images, labels = load_galaxy10()

    # No need to convert the labels to categorical 10 classes
    # labels = to_categorical(labels, 10)

    # To convert to desirable type
    labels = torch.tensor(labels[::step], dtype=torch.long)  # Changed dtype to torch.long
    images = torch.tensor(images[::step], dtype=torch.float32)
    return images, labels

def split_dataset(images, labels):
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
    return train_images, train_labels, test_images, test_labels

def get_classifier(y):
    return y
