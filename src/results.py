import matplotlib.pyplot as plt
import numpy as np
import torch
from main import GalaxyCNN as ComplexCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

    
def make_confusion_matrix(test_loader, PATH='trained_model.pth', num_classes=10):
    '''
    Function that loads and tests the model, then creates a confusion matrix 

    input:
    - test_loader: test data
    - PATH: relative path to model
    - num_classes: number of classifications

    Output:
    - None
    - Prints confusion matrix
    '''

    # Testing loop
    model = ComplexCNN(num_classes)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    expected = np.array([])
    test_labels = np.array([])

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch["data"], batch["label"]
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.argmax(1)
            expected = np.concatenate((expected, labels))
            test_labels = np.concatenate((test_labels, predicted))

    with open('src/classes.json') as f:
        d = json.load(f)

    cm = confusion_matrix(test_labels, expected)
    ConfusionMatrixDisplay(cm).plot()
    plt.xticks(np.arange(10), list(d.values()), rotation=45)
    plt.yticks(np.arange(10), list(d.values()), rotation=45)
    plt.savefig('ConfusionMatrix.png')
    plt.show()
    return

def accuracy(model, loader, device):
    '''
    Function that loads and tests the model, then reports the accuracy

    input:
    - model: model to evaluate
    - loader: test data
    - device: device over which to run the model (gpu or cpu)

    Output:
    - accuracy: number of correct predictions / number of total predictions
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch["data"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total