import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from main import ComplexCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    expected = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0], batch[1]
            outputs = model(inputs)
            predicted = outputs.argmax(1)
            labels = labels.argmax(1)
            expected.append(labels)
            test_labels.append(outputs)

    cm = confusion_matrix(test_labels, expected)
    ConfusionMatrixDisplay(cm).plot()
    return