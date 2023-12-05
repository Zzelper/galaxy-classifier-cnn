from astroNN.datasets import load_galaxy10
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

"""
data_import.py

This module provides functions and classes for loading and preprocessing the Galaxy10 dataset. 
It uses the astroNN library to load the dataset, and sklearn to split the dataset into training and testing sets.

The module assumes that the Galaxy10 dataset is located in the ~/.astroNN/datasets/ directory, 
and will automatically download the dataset if it is not found.
"""


class GalaxyDataset(Dataset):
    """
    This class overrides the __len__ and __getitem__ methods for compatibility with PyTorch's data loading utilities. 
    It also provides an optional transform argument in its constructor for data augmentation.

    Attributes:
        data (torch.Tensor): The image data, permuted to match PyTorch's expected input format.
        labels (torch.Tensor): The class labels.
        transform (callable, optional): An optional transform to be applied on the data.

    Methods:
        __len__: Returns the number of images in the dataset.
        __getitem__: Returns a dictionary containing the image data and label for a given index, 
                     with optional transformations applied to the data.
    """

    def __init__(self, data, labels, transform=None):
        """
        Initialize the GalaxyDataset instance.

        Args:
            data (torch.Tensor): The image data.
            labels (torch.Tensor): The class labels.
            transform (callable, optional): An optional transform to be applied on the data.
        """
        self.data = torch.permute(data,(0,3,1,2))
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a dictionary containing the image data and label for a given index.

        Args:
            idx (int): The index of the data.

        Returns:
            dict: A dictionary containing the image data and label.
        """
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return {'data': data, 'label': label}


def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def load_images_labels(step=1):
    """
    Return a dictionary containing the image data and label for a given index.

    Args:
        idx (int): The index of the data.

    Returns:
        dict: A dictionary containing the image data and label.
    """    
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
    """
    Split the dataset into training and testing sets.

    Args:
        images (torch.Tensor): The image data.
        labels (torch.Tensor): The class labels.

    Returns:
        tuple: A tuple containing the training images, training labels, testing images, and testing labels.
    """
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
    return train_images, train_labels, test_images, test_labels

def get_classifier(y):
    return y
