from src.data_import import *
from src.model import *
import matplotlib.pyplot as plt
import torch
from src.results import *

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images, labels = load_images_labels(1)

    train_images, train_labels, test_images, test_labels = split_dataset(images, labels)

    # Load your dataset and apply data augmentation
    train_dataset = GalaxyDataset(train_images, train_labels)
    test_dataset = GalaxyDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = GalaxyCNN(num_classes=10).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0000005)

    train_accuracies = []
    test_accuracies = []

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs, labels = batch["data"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        train_accuracies.append(accuracy(model, train_loader))
        test_accuracies.append(accuracy(model, test_loader))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    epochs = list(range(num_epochs))
    ax.plot(epochs, train_accuracies, label="Training")
    ax.plot(epochs, test_accuracies, label="Testing")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.savefig("./data/accuracies.png")
