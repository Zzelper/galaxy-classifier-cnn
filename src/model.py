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
    def __init__(self, num_classes):
        super(GalaxyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(128*4*4*16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        #x = self.conv1(x) # [64,3,256,256] -> [64,16,256,256]
        #x = self.relu(x)
        #x = self.pool(x) # [64,16,256,256] -> [64,16,128,128]
        #x = self.conv2(x) # [64,16,128,128] -> [64, 64, 65,65]
        #x = self.relu(x)
        #x = self.pool(x) #[64,64, 32, 32]
        x = self.pool(self.relu(self.conv1(x))) #[64, 3, 256, 256] -> [64, 16, 256, 256]
        x = self.pool(self.relu(self.conv2(x))) # [64, 16, 128, 128] -> [64,
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(-1, 128*4*4*16)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x



class GalaxyTestCNN(nn.Module):
    def __init__(self, num_classes):
        super(GalaxyTestCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.1)


        self.fc1 = nn.Linear(64*32*32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)

        return x