from astroNN.datasets import load_galaxy10
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def load_images_labels():
    # To load images and labels (will download automatically at the first time)
    # First time downloading location will be ~/.astroNN/datasets/
    images, labels = load_galaxy10()

    # To convert the labels to categorical 10 classes
    labels = to_categorical(labels, 10)

    # To convert to desirable type
    labels = torch.tensor(labels[:1000], dtype=torch.float32)
    images = torch.tensor(images[:1000], dtype=torch.float32)
    return images, labels

def split_dataset(images, labels):
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
    return train_images, train_labels, test_images, test_labels