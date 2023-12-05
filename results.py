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
