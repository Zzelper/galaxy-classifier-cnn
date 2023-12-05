from src.data_import import *
from src.model import *
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import GradScaler, autocast

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

    # Training loop
    num_epochs = 100

    scaler = GradScaler()

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

        if epoch % 10 == 9:
            # Testing loop
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    inputs, labels = batch["data"].to(device), batch["label"].to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()  
                accuracy = 100 * correct / total
                print(f"Accuracy: {accuracy:.2f}%")
