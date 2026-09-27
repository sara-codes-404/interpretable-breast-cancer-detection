"""
evaluate.py

Final evaluation of the trained model on the official PCam test split.

The test set is completely separate from the training and validation data,
so it is not used during training or model selection.

Why is this evaluation needed?

The validation accuracy is used to select the best checkpoint during training.
Therefore, it is indirectly involved in the model selection process.

The test set provides a final and unbiased evaluation of the trained model
on data that was not used during training or model selection.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import HFPCamDataset, get_default_transforms
from model import build_resnet18, get_device


@torch.no_grad()
def get_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple:
    """
Collect model predictions for the entire DataLoader.

Args:
    model: Trained model used for evaluation.
    loader: DataLoader containing the data to evaluate.
    device: 'cuda' or 'cpu'.

Returns:
    A tuple containing (y_true, y_pred, y_prob):
        - y_true: True labels as a NumPy array.
        - y_pred: Predicted labels using a 0.5 threshold as a NumPy array.
        - y_prob: Predicted probability for the positive class after sigmoid,
          used for calculating ROC-AUC.
"""
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1) # Shape: (batch,)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)

    return y_true, y_pred, y_prob


def evaluate_on_test_set(
    checkpoint_path: str = "results/checkpoints/best_model_partial.pth",
    freeze_mode: str = "partial",
    n_test_samples: int = 5000,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
Evaluate the saved model on the official PCam test split.

Args:
    checkpoint_path: Path to the saved model weights (.pth).
    freeze_mode: Must match the freeze_mode used during training because
        it determines the model architecture.
    n_test_samples: Number of samples to evaluate from the test split.
        The official test set contains 32,768 samples. A smaller subset
        can be used for faster evaluation.
    batch_size: Batch size used for inference. This can be larger than
        the training batch size because gradients are not calculated.

Returns:
    A dictionary containing accuracy, precision, recall, F1-score,
    and ROC-AUC on the test set.
"""
    device = get_device()
    
    print(f"Loading {n_test_samples} samples from the official 'test' split...")
    # Note: We use the 'test' split instead of 'train' because this set
    # is completely separate from the data used to train the model.
    test_hf = load_dataset(
        "1aurent/PatchCamelyon", split=f"test[:{n_test_samples}]"
    )
    test_dataset = HFPCamDataset(
        hf_dataset=test_hf, transform=get_default_transforms(train=False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")

    print(f"Loading model from {checkpoint_path} (freeze_mode='{freeze_mode}')...")
    model = build_resnet18(freeze_mode=freeze_mode, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("Evaluation results on the official TEST SET (unseen data)")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name:>12}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(f"Predicted: healthy   Predicted: tumor")
    print(f"Actual: healthy       {cm[0, 0]:>6}          {cm[0, 1]:>6}")
    print(f"Actual: tumor         {cm[1, 0]:>6}          {cm[1, 1]:>6}")
    print("=" * 50)

    # Explain the two main types of errors in medical diagnosis:
    fn = cm[1, 0]  # The model predicted "healthy", but the actual label was "tumor".
                   # This is the more serious type of error in medical diagnosis.
    fp = cm[0, 1]  # The model predicted "tumor", but the actual label was "healthy".
    print(f"\nFalse Negative (tumor predicted as healthy): {fn}")
    print(f"False Positive (healthy predicted as tumor): {fp}")
    print(
    "\nNote: In medical diagnosis, false negatives are usually more serious "
    "because a real tumor may be missed. Make sure to highlight this metric "
    "in the final report."
)

    return metrics


if __name__ == "__main__":
    evaluate_on_test_set(
        checkpoint_path="results/checkpoints/best_model_partial.pth",
        freeze_mode="partial",
        n_test_samples=5000,
    )
