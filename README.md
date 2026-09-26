# Interpretable Breast Cancer Detection using Transfer Learning and Grad-CAM

## Overview

This project focuses on classifying histopathological images for tumor
detection. The project uses the PCam (PatchCamelyon) dataset. A ResNet-18
model with Transfer Learning is used for binary classification, while
Grad-CAM is used to improve model interpretability.

This is the first of two related projects exploring breast cancer detection
from complementary data modalities — see [Future Work](#future-work) for how
this connects to a second project on gene-expression-based subtype
prediction.

## Problem Statement

The model is designed to classify histopathological images as either tumor
or healthy tissue. In medical applications, a prediction alone may not be
sufficient; it is important to understand the basis of the model's
decision. Grad-CAM helps identify the regions of an image that contribute
to the model's prediction, improving the transparency of the model's
decisions.

## Data Preparation

We used 5,000 samples from the PCam dataset, with 4,000 samples for
training and 1,000 samples for validation. Different transformations were
applied to the training and validation sets. Data augmentation was applied
only to the training set to increase data diversity and help the model
generalize better to unseen images.

## Model Architecture

We use ResNet-18 as the main architecture in this project. Transfer
Learning is applied using pretrained weights from ImageNet, allowing the
model to leverage features learned from a large-scale dataset. The
original final fully connected (FC) layer, which was designed for 1,000
ImageNet classes, is replaced with a new FC layer for our binary
classification task. The modified model produces a single output logit
for distinguishing between tumor and healthy tissue.

## Training Strategy

The model is trained using `BCEWithLogitsLoss`, which is suitable for our
binary classification task with a single output logit. The Adam optimizer
is used to update the trainable parameters of the model. After each
training epoch, the model is evaluated on the validation set to monitor
its performance on unseen samples. We use `ReduceLROnPlateau` to reduce
the learning rate when the validation loss stops improving. The best
model checkpoint is saved based on the validation loss.

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

The model achieved a best validation accuracy of 90.4% during training.
The lowest validation loss was 0.2212, achieved at epoch 18.

The training and validation curves show that both loss and accuracy
improved during training. After epoch 18, the validation loss started to
increase slightly, while the validation accuracy remained around 90%. The
best checkpoint was therefore saved based on the lowest validation loss.

### Training Results

| Metric | Value |
|---|---|
| Best Validation Accuracy | 90.4% |
| Best Validation Loss | 0.2212 |
| Best Epoch | 18 |
| Final Training Accuracy | 90.9% |
| Final Validation Accuracy | 90.4% |

![Training curves](results/figures/training_curves_partial.png)

## Test Set Evaluation

Validation accuracy alone can be misleading, since it is also used to
select the best checkpoint during training. To get an unbiased estimate
of generalization, we additionally evaluated the final model on the
**official, held-out PCam test split** — a set of slides the model never
saw during training or checkpoint selection.

| Metric | Validation | Official Test Set |
|---|---|---|
| Accuracy | 90.4% | ~82–84% |
| Precision | — | ~0.90–0.92 |
| Recall | — | ~0.74–0.75 |
| F1-score | — | ~0.81–0.82 |
| ROC-AUC | — | ~0.92–0.93 |

This gap between validation and test performance is real and is discussed
in detail in [Limitations](#limitations) below, since understanding *why*
it exists turned out to be one of the more instructive parts of this
project.

## Explainability with Grad-CAM

Grad-CAM was used to visualize which regions of each image most
influenced the model's prediction. Heatmaps were computed on the
`layer3` activations of ResNet-18 rather than the more commonly used
`layer4` — with 96×96 inputs, `layer4`'s activation map shrinks to just
3×3, too coarse to localize meaningfully. `layer3`'s 6×6 map gives a much
better resolution/semantic trade-off for small image patches like these.

The resulting heatmaps consistently highlight regions of high nuclear
density — consistent with real histopathological markers of malignancy
that pathologists look for.

![Grad-CAM examples](results/figures/gradcam_examples.png)

## Limitations

- **Limited training data:** only 5,000 of the 262,144 images available
  in the full PCam training set were used, due to restricted dataset
  access (see [Environment & Access Constraints](#environment--access-constraints)).
- **Validation/test generalization gap:** validation accuracy (~90%) was
  consistently higher than official test accuracy (~82–84%). Investigation
  traced part of this gap to a data sampling issue — taking the first
  5,000 samples in dataset order captured limited slide-level diversity,
  since PCam images are ordered by source whole-slide image. Switching to
  samples spread across the full training range partially narrowed the
  gap but did not close it entirely. This is consistent with published
  results showing PCam accuracy scaling directly with the number of
  distinct training slides used, not just the number of patches — with the
  full dataset, this gap would be expected to shrink substantially.
- **Recall/precision trade-off:** the model currently misses more true
  tumor cases (recall ~0.74) than it produces false alarms (precision
  ~0.90–0.92). In a real screening context, this trade-off would need to
  be revisited, since false negatives are typically the costlier error.

## Environment & Access Constraints

This project was developed under real infrastructure constraints worth
documenting: Google Colab and Kaggle are both restricted in the
developer's region, requiring VPN access for GPU-backed training. The
original PCam HDF5 files (hosted on Google Drive) were also inaccessible
due to download quota limits, so the Hugging Face Hub mirror
(`1aurent/PatchCamelyon`) was used instead, loading a constrained subset
of the full dataset rather than all 262,144 training images.

The project was developed using Python 3.12. The main libraries used
include PyTorch, Torchvision, Hugging Face `datasets`, scikit-learn, and
Matplotlib.

## Future Work

- Train on the full PCam dataset to close the validation/test gap
  described above.
- Apply stain normalization (e.g., the Macenko method) to reduce
  slide-to-slide color variation — a likely contributor to the
  generalization gap.
- Quantitative Grad-CAM evaluation (e.g., deletion/insertion metrics)
  rather than qualitative visual inspection alone.
- **Multimodal integration** with a companion project, *Graph Neural
  Network for Breast Cancer Subtype Prediction from Gene Expression
  Data*: comparing Grad-CAM attention regions against PAM50 molecular
  subtypes, and eventually building a fusion model combining image-level
  and gene-expression features. Together, these two projects are intended
  as early steps toward interpretable, multimodal breast cancer diagnosis.

## Project Structure

```
Interpretable-Breast-Cancer-Detection/
├── src/
│   ├── dataset.py     # PCam dataset loading (HF Hub + diverse sampling)
│   ├── model.py         # ResNet-18 with configurable layer freezing
│   ├── train.py          # Training loop, LR scheduling, checkpointing
│   ├── evaluate.py       # Held-out test set evaluation
│   └── gradcam.py        # Grad-CAM implementation and visualization
├── results/
│   ├── checkpoints/      # Saved model weights (not tracked in git)
│   ├── figures/           # Training curves, Grad-CAM examples
│   └── training_history_*.json
├── requirements.txt
├── README.md
└── PROJECT_SUMMARY.md
```

## Installation

To run this project locally, Python and the required libraries need to be
installed first. All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
# Train the model
python src/train.py

# Evaluate on the official held-out test set
python src/evaluate.py

# Generate Grad-CAM visualizations
python src/gradcam.py
```

**Note:** training was run on Google Colab (free GPU tier); the code is
device-agnostic and runs on CPU as well, though considerably slower.

## Data

- **Dataset:** [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) —
  96×96 histopathology patches with binary tumor/normal labels.
- **Source used:** [`1aurent/PatchCamelyon`](https://huggingface.co/datasets/1aurent/PatchCamelyon)
  on the Hugging Face Hub.

## References

- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-based Localization.* ICCV.
- Veeling, B. S., et al. (2018). *Rotation Equivariant CNNs for Digital
  Pathology.* MICCAI.
