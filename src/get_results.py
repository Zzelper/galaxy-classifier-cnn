import numpy as np
import torch

def accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch["data"], batch["label"]
            outputs = model(inputs)
            predicted = outputs.argmax(1)
            labels = labels.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct/total
