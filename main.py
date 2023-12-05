from src.data_import import *
from src.model import *
import matplotlib.pyplot as plt
import torch
from src.results import *
from torch.cuda.amp import GradScaler, autocast

"""
piersonmain.py

This script is the main entry point for a PyTorch-based image classification task. 
It mainly loads and preprocesses the image data, defines a convolutional neural network model, 
rains the model using a specified number of epochs. 

The script uses data augmentation techniques such as random flipping and rotation, and normalizes the images. 
It also uses mixed precision training with the help of GradScaler.

The script imports from the 'src' directory necessary modules: 'data_import', 'model', and 'results'. The 'data_import' 
The script is designed to run on a CUDA-enabled GPU if available, and falls back to CPU otherwise.
"""

# module contains functions for loading and preprocessing the data. The 'model' module contains the GalaxyCNN class,
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images, labels = load_images_labels(1)

    train_images, train_labels, test_images, test_labels = split_dataset(images, labels)


    # Define the transformations
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create your GalaxyDataset instance for your training data
    train_dataset = GalaxyDataset(train_images, train_labels, transform=data_transforms)
    test_dataset = GalaxyDataset(test_images, test_labels)

    # Create your DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Initialize the model
    model = GalaxyCNN(num_classes=10).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000005)

    train_accuracies = []
    test_accuracies = []

    # Training loop
    num_epochs = 100

    scaler = GradScaler()
    
    
    """
     The training loop iterates over the training data for a specified number of epochs.
     In each epoch, it iterates over the training data in batches. For each batch, it:
     - Zeroes the gradients of the model parameters
     - Runs the forward pass of the model in mixed precision using autocast
     - Computes the loss
     - Scales the loss using the GradScaler and backpropagates the scaled loss
     - Steps the optimizer and updates the GradScaler
     After each epoch, it computes and stores the training and testing accuracies.
    """
    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs, labels = batch["data"].to(device), batch["label"].to(device)

            optimizer.zero_grad()

            # Use autocast to run the forward pass in mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Use the scaler to scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        train_accuracies.append(accuracy(model, train_loader, device))
        test_accuracies.append(accuracy(model, test_loader, device))

    # plot accuracies and dave plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    epochs = list(range(num_epochs))
    ax.plot(epochs, train_accuracies, label="Training")
    ax.plot(epochs, test_accuracies, label="Testing")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.savefig("./data/accuracies.png")
