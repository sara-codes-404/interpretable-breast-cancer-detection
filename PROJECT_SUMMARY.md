# Project Summary

## 1. Project Title

Interpretable Breast Cancer Detection using Transfer Learning and Grad-CAM on Histopathological Images

## 2. Problem

In this project, we work with histopathological images to classify tissue samples as tumor or healthy. This is a binary classification problem. The project is not limited to classification; model interpretability is also an important objective.

## 3. Dataset

The dataset used in this project is PCam (PatchCamelyon). It contains histopathological images with a resolution of 96 × 96 pixels. We used 5,000 samples in this project, with 4,000 samples for training and 1,000 samples for validation.

## 4. Task & Labels

This project is a binary classification task with two classes: Healthy and Tumor. Healthy is represented by label 0, while Tumor is represented by label 1.

## 5. Model

ResNet-18 is the architecture used in this project. We use Transfer Learning with pretrained weights from ImageNet. One of the main modifications is that the original ResNet-18 was designed for 1,000 ImageNet classes. We replaced the final fully connected (FC) layer to make the model suitable for our binary classification task. The modified model produces a single output logit for the binary classification task.
