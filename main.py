from src.data_import import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjusted the input size here
        self.fc2 = nn.Linear(512, num_classes)
        # Dropout
        self.dropout = nn.Dropout(0.5)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # Check the size after convolution and pooling
        # print(x.size())  # Uncomment to print the size for debugging
        x = x.view(-1, 128 * 32 * 32)  # Adjusted the view size here
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
if __name__ == "__main__":
    images, labels = load_images_labels()
    images = images.permute(0, 3, 1, 2)
    train_loader = zip(images, labels)

    # Instantiate the model
    num_classes = 10  # Adjusted for 10 classes
    num_epochs = 5  # Define num_epochs
    model = CNN(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # This part can be in another script where you load data, preprocess, and train the model.
    # For brevity, here's a placeholder for training the model using your dataset:

    # Training loop (you need to replace this with your actual data loading and training process)
    for epoch in range(num_epochs):  # Define num_epochs
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            
            # Ensure the shapes of outputs and labels match
            # Resize labels to match the batch size of the outputs
            labels_resized = labels[:outputs.shape[0]].long()  # Convert labels to torch.long
            
            loss = criterion(outputs, labels_resized)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # Save the trained model (optional)
    torch.save(model.state_dict(), 'trained_model.pth')

