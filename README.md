# PHYS 449 - Galaxy Image Classification
Final project for the PHYS-449 class.
Group members: Dvir Zagury, Pierson Tomietto, Evan Ru-Xiang Chow, Patrick Thompson.
This repository is a reimplementation of the paper:

**Title:** Galaxy Classification Using Deep Learning  
**Authors:** P. Ghadekar, K. Chanda, S. Manmode, S. Rawate, S. Chaudhary, R. Suryawanshi  
**Published in:** Advances in Smart Computing and Information Security (ASCIS) 2022  
**DOI:** [10.1007/978-3-031-23092-9_1](https://doi.org/10.1007/978-3-031-23092-9_1)  
**Published on:** 11 January 2023  
**Publisher:** Springer, Cham  
**Print ISBN:** 978-3-031-23091-2  
**Online ISBN:** 978-3-031-23092-9


## Purpose
This repository contains a re-implementation of the CNN galaxy classification model described in the paper:

    "Galaxy Classification Using Deep Learning"
    by: Premanand Ghadekar, Kunal Chanda, Sakshi Manmode, Sanika Rawate, Shivam Chaudhary & Resham Suryawanshi 

This is a PyTorch-based image classification task for galaxy images. 
The goal is to classify galaxy images into 10 different classes, outlined in the classes.json file.

Accuracy is calculated by counting the number of correct predictions in the testing batch over the number of inccorect predictions.

## How to Run
To run the main script, navigate to the root directory of the repository and run the following command:

```sh
python main.py
```

## Repository Structure
The repository is structured as follows:
- `src/`: Contains the source code modules.
  - `data_import.py`: Contains functions for loading and preprocessing the image data.
  - `model.py`: Contains the `GalaxyCNN` class, a convolutional neural network for image classification.
  - `results.py`: Contains functions for analyzing and visualizing the results.
- `main.py`: The main script that ties everything together.

- `params.json`: Contains configurable hyperparameters: learnig rate, number of epochs, and batch size.
- `classes.json`: Galaxy classficiations and their numerical label.


## Dependencies
The project depends on the following Python libraries:
- PyTorch
- torchvision
- numpy
- matplotlib
- sklearn
- astroNN
- json

## AI Usage Statement
This project leverages ChatGPT and GitHub Copilot as secondary yet important tools.
LLM resources were used for brainstorming, asking questions, seeking clarifications, generating code snippets, and suggesting different templates for code implementation.
