import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from astroNN.datasets import load_galaxy10
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from results import *

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def load_images_labels(samples=100):
    # To load images and labels (will download automatically at the first time)
    # First-time downloading location will be ~/.astroNN/datasets/
    images, labels = load_galaxy10()

    # To convert the labels to categorical 10 classes
    labels = to_categorical(labels, 10)

    # if not taking full data, take random samples
    idx = np.random.choice(np.arange(len(labels)), samples, replace=False)
    labels = labels[idx]
    images = images[idx]

    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    #return new_images, new_labels
    return images, labels

def split_dataset(images, labels):
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
    return train_images, train_labels, test_images, test_labels

# Define your CNN model
class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.reshape(-1, 128 * 32 * 32)
        x = self.dropout_fc(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Load your data
    images, labels = load_images_labels()
    images = images.permute(0, 3, 1, 2)
    
    # Normalize the images
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #images = normalize(images)

    # Split the data into training and testing sets
    train_images, train_labels, test_images, test_labels = split_dataset(images, labels)

    # Create DataLoader for training set
    batch_size = 8  # Choose an appropriate batch size
    train_dataset = list(zip(train_images, train_labels))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = list(zip(test_images, test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Define the number of classes and epochs
    num_epochs = 5  # Increase the number of epochs for better learning
    num_classes = 10

    # Instantiate the model
    model = ComplexCNN(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.argmax(dim=1).type(torch.long))  # Ensure labels are torch.long
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    # Save the trained model if needed
    torch.save(model.state_dict(), 'trained_model.pth')

    new_confusion_matrix(train_loader)
