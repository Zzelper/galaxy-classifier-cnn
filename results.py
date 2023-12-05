import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from main import ComplexCNN
from main import ComplexCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from main import load_images_labels, split_dataset
from torch.utils.data import DataLoader
import json

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(df_confusion.columns))
    #plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    #plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    plt.show()

def test_model(test_loader, PATH='trained_model.pth', num_classes=10):

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

    # Testing loop
    model = ComplexCNN(num_classes)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_actu = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0], batch[1]
            outputs = model(inputs)
            predicted = outputs.argmax(1)
            y_pred.append(predicted)  

            total += labels.size(0)
            labels = labels.argmax(1)
            y_actu.append(labels) 
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plotting confusion matrix
    print('actual')
    print(y_actu)
    print(y_actu[0])
    print(labels)
    print('predicted')
    print(y_pred)
    print(y_pred[0])
    print(predicted)
    df_confusion = pd.crosstab(y_actu[0], y_pred[0])
    plot_confusion_matrix(df_confusion)
    
def new_confusion_matrix(test_loader, PATH='trained_model.pth', num_classes=10):
    # Testing loop
    model = ComplexCNN(num_classes)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    expected = np.array([])
    test_labels = np.array([])

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0], batch[1]
            outputs = model(inputs)
            predicted = outputs.argmax(1)
            labels = labels.argmax(1)
            expected = np.concatenate((expected, labels))
            test_labels = np.concatenate((test_labels, predicted))

    with open('src/classes.json') as f:
        d = json.load(f)
        print(list(d.keys()))
        print(list(d.values()))

    cm = confusion_matrix(test_labels, expected)
    ConfusionMatrixDisplay(cm).plot()
    plt.xticks(np.arange(10), list(d.values()), rotation=45)
    plt.yticks(np.arange(10), list(d.values()), rotation=45)
    plt.show()
    return
