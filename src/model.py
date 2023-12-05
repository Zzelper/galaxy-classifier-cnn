import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torchvision
#from data_import import *

# Define the ConvNet architecture
class GalaxyCNN(nn.Module):
    """
    The architecture of the model is as follows:
    - Convolutional layer (conv1) with 16 output channels, a 3x3 kernel, and padding of 1.
    - ReLU activation function.
    - Max pooling layer with a 2x2 kernel and stride of 2.
    - Fully connected layer (fc1) with 512 output features.
    - ReLU activation function.
    - Fully connected layer (fc2) with a number of output features equal to the number of classes.

    The forward method applies these layers in sequence, reshaping the tensor before passing it to the fully connected layers.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        relu (nn.ReLU): The ReLU activation function.
        pool (nn.MaxPool2d): The max pooling layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.

    Methods:
        forward(x): Defines the forward pass of the model.
    """

    def __init__(self, num_classes):
        super(GalaxyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 16 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 128 * 128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


