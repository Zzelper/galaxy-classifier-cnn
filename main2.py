import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from astroNN.datasets import load_galaxy10
import numpy as np
from sklearn.model_selection import train_test_split

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def load_images_labels():
    # To load images and labels (will download automatically at the first time)
    # First time downloading location will be ~/.astroNN/datasets/
    images, labels = load_galaxy10()

    # To convert the labels to categorical 10 classes
    labels = to_categorical(labels, 10)

    # To convert to desirable type
    labels = torch.tensor(labels[:11], dtype=torch.float32)
    images = torch.tensor(images[:11], dtype=torch.float32)
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

# Main code
if __name__ == "__main__":
    # Load your data
    images, labels = load_images_labels()  # Assuming this function loads your data
    images = images.permute(0, 3, 1, 2)  # Permute the dimensions to match the model's input
    images = images.unsqueeze(0)  # Add a dimension for the batch size
    train_loader = zip(images, labels)

    # Define the number of classes and epochs
    num_epochs = 1
    num_classes = 10

    # Instantiate the model
    model = ComplexCNN(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = list(zip(images, labels))

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.type(torch.long)
            print(labels.shape)
            print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    # Save the trained model if needed
    torch.save(model.state_dict(), 'trained_model.pth')
