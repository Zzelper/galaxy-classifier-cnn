from src.data_import import *
from src.model import *
import matplotlib.pyplot as plt
from torch import optim

if __name__ == "__main__":
    images, labels = load_images_labels(10)
    train_images, train_labels, test_images, test_labels = split_dataset(images, labels)

    # Load your dataset and apply data augmentation
    train_dataset = GalaxyDataset(train_images, train_labels)
    test_dataset = GalaxyDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = GalaxyCNN(num_classes=10)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

    # Training loop
    accuracies = []
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch["data"], batch["label"]
            #print(inputs.size(), labels.size())
            optimizer.zero_grad()
            #print(inputs)
            outputs = model(inputs)
            #print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Testing loop
        model.eval()
        correct = 0
        total = 0
        validation_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch["data"], batch["label"]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                num_batches += 1
                predicted = outputs.argmax(1)
                total += labels.size(0)
                labels = labels.argmax(1)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Test Accuracy: {accuracy:.2f}%")
        scheduler.step(validation_loss)
        #scheduler.step()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(len(accuracies))), accuracies)
    plt.show()

