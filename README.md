## Overview
This project focuses on classifying histopathological images for tumor detection.
The project uses the PCam (PatchCamelyon) dataset. 
A ResNet-18 model with Transfer Learning is used for binary classification, while Grad-CAM is used to improve model interpretability.

## Problem Statement
The model is designed to classify histopathological images as either tumor or healthy tissue.
In medical applications, a prediction alone may not be sufficient; it is important to understand the basis of the model's decision.
Grad-CAM helps identify the regions of an image that contribute to the model's prediction, improving the transparency of the model's decisions.

### Data Preparation
We used 5,000 samples from the PCam dataset, with 4,000 samples for training and 1,000 samples for validation. 
Different transformations were applied to the training and validation sets. 
Data augmentation was applied only to the training set to increase data diversity and help the model generalize better to unseen images.

### Model Architecture

We use ResNet-18 as the main architecture in this project. Transfer Learning is applied using pretrained weights from ImageNet, allowing the model to leverage features learned from a large-scale dataset. The original final fully connected (FC) layer, which was designed for 1,000 ImageNet classes, is replaced with a new FC layer for our binary classification task. The modified model produces a single output logit for distinguishing between tumor and healthy tissue.

### Training Strategy
The model is trained using `BCEWithLogitsLoss`, which is suitable for our binary classification task with a single output logit. The Adam optimizer is used to update the trainable parameters of the model. After each training epoch, the model is evaluated on the validation set to monitor its performance on unseen samples. We use `ReduceLROnPlateau` to reduce the learning rate when the validation loss stops improving. The best model checkpoint is saved based on the validation loss.

### Training Configuration

| Parameter | Value |
|---|---|
| Dataset samples | 5,000 |
| Training samples | 4,000 |
| Validation samples | 1,000 |
| Epochs | 20 |
| Batch size | 16 |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| Freeze mode | Partial |

## Results

## Explainability with Grad-CAM

## Project Structure
The repository currently contains the main project documentation files:

- `README.md`: Provides an overview of the project, its methodology, training strategy, and other important information.
- `PROJECT_SUMMARY.md`: Contains a more detailed summary of the project and its main components.

  ## Installation
  To run this project locally, Python and the required libraries need to be installed first. The exact dependencies will be listed in the `requirements.txt` file.
  
### Environment
The project was developed using Python 3.12. The main libraries used in the project include PyTorch, Torchvision, Hugging Face Datasets, and Matplotlib.
