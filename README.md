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
