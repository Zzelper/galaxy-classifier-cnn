import numpy as np
import torch


def accuracy(model, loader, device):
    """
    GalaxyCNN is a simple convolutional neural network for image classification.

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
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch["data"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total



