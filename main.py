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

def load_images_labels():
    # To load images and labels (will download automatically at the first time)
    # First time downloading location will be ~/.astroNN/datasets/
    images, labels = load_galaxy10()

    # To convert the labels to categorical 10 classes
    labels = to_categorical(labels, 10)

    labels1 = labels[:10]
    labels2 = labels[1000:1010]
    labels3 = labels[2000:2010]
    labels4 = labels[3000:3010]
    labels5 = labels[5000:5010]
    labels6 = labels[6000:6010]
    labels7 = labels[7000:7010]
    labels8 = labels[8000:8010]
    labels9 = labels[9000:9010]
    labels10 = labels[10000:10010]
    labels11 = labels[11000:11010]
    labels12 = labels[12000:12010]
    new_labels = np.concatenate((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
                                 labels11, labels12), axis=0)

    images1 = images[:10]
    images2 = images[1000:1010]
    images3 = images[2000:2010]
    images4 = images[3000:3010]
    images5 = images[5000:5010]
    images6 = images[6000:6010]
    images7 = images[7000:7010]
    images8 = images[8000:8010]
    images9 = images[9000:9010]
    images10 = images[10000:10010]
    images11 = images[11000:11010]
    images12 = images[12000:12010]
    new_images = np.concatenate((images1, images2, images3, images4, images5, images6, images7, images8, images9, images10,
                                 images11, images12), axis=0)

    # To convert to desirable type
    new_labels = torch.tensor(new_labels[:100], dtype=torch.float32)
    new_images = torch.tensor(new_images[:100], dtype=torch.float32)
    return new_images, new_labels

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

    test_model(test_loader)
