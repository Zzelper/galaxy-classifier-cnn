import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from galaxy10_decals_dataset import Galaxy10
import json

# Load hyperparameters from params.json
with open('params.json') as f:
    params = json.load(f)

# Extract hyperparameters
num_classes = params["num_classes"]
batch_size = params["batch_size"]
learning_rate = params["learning_rate"]
num_epochs = params["num_epochs"]
random_seed = params["random_seed"]

# Set random seed for reproducibility
torch.manual_seed(random_seed)

# Define the CNN model
class GalaxyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GalaxyCNN, self).__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification
        self.classification = nn.Sequential(
            nn.Linear(16 * 128 * 128, 256),
            nn.Tanh(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x

# Load the Galaxy10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

galaxy_dataset = Galaxy10(transform=transform)
train_size = int(0.8 * len(galaxy_dataset))
test_size = len(galaxy_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(galaxy_dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model
model = GalaxyCNN(num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {100 * accuracy:.2f}%')
