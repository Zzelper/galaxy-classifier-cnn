from src.data_import import *
from src.model import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #print("Hello World")
    images, labels = load_images_labels(5000)
    train_images, train_labels, test_images, test_labels = split_dataset(images, labels)
    #print(images[0])
    #print(test_images)
    # Load your dataset and apply data augmentation
    train_dataset = GalaxyDataset(train_images, train_labels)
    test_dataset = GalaxyDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = GalaxyCNN(num_classes=10)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Testing loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
