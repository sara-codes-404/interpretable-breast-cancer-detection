"""
model.py

Defines the ResNet-18 model with transfer learning for binary
classification (tumor/healthy) on the PCam dataset.

The model is device-agnostic:
the `get_device()` function automatically detects whether
a GPU or CPU is available, so the same code can run locally
and on Kaggle without changing the model code.
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


def get_device() -> torch.device:
    """
    Returns the available device (GPU or CPU).

    This function allows the project to run in both the local environment
    (without a GPU) and Kaggle (with a GPU) without changing the code.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")
    return device


def build_resnet18(
    freeze_mode: Literal["full", "partial", "none"] = "partial",
    pretrained: bool = True,
) -> nn.Module:
    """
    Builds a ResNet-18 model for binary classification on PCam.

    Args:
        freeze_mode: Controls which parts of the model are frozen:
            - "full": All layers except the final FC layer are frozen.
              This is the fastest option and is suitable for CPU training.
            - "partial": conv1, layer1, and layer2 are frozen.
              layer3, layer4, and the FC layer remain trainable.
              This provides a good balance between speed and accuracy.
            - "none": No layers are frozen, so the whole model is fine-tuned.
              This can provide better accuracy but requires more computation.
        pretrained: If True, loads pretrained ImageNet weights.
            This is recommended when using transfer learning.

    Returns:
        nn.Module: A ResNet-18 model with one output logit.
        The output is a raw logit without sigmoid because
        BCEWithLogitsLoss applies sigmoid internally.

    Example:
        >>> model = build_resnet18(freeze_mode="full")
        >>> device = get_device()
        >>> model = model.to(device)
"""
    
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

   # Step 1: Freeze layers according to freeze_mode
    if freeze_mode == "full":
        # Freeze all parameters; only the FC layer replaced below will be trainable.
        for param in model.parameters():
            param.requires_grad = False

    elif freeze_mode == "partial":
    # Freeze only conv1, bn1, layer1, and layer2.
    # These layers learn low-level features such as edges and textures,
    # which are useful for both ImageNet and medical images.
    # The deeper layers learn more specific features, so layer3 and layer4
    # remain trainable to adapt to the medical image domain. 
        layers_to_freeze = [model.conv1, model.bn1, model.layer1, model.layer2]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    elif freeze_mode == "none":
    # Nothing to do here; all parameters are trainable by default.
        pass

    else:
        raise ValueError(
            f"freeze_mode must be one of 'full', 'partial', 'none', "
            f"No '{freeze_mode}'"
        )

    # Step 2: Replace the final FC layer
    # The original ResNet-18 is designed for 1,000 ImageNet classes,
    # but we only need one output (a binary logit).
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),  # Reduce overfitting, especially because we only have 5,000 samples.
        nn.Linear(num_features, 1),
    )
    # The FC layer is always trainable, even with freeze_mode="full".
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_trainable_params(model: nn.Module) -> None:
    """
    Prints the number of trainable parameters compared to the total number
    of parameters in the model.

    This function is useful for checking whether freeze_mode works correctly.
    For example, with freeze_mode="full", only a small number of parameters
    should be trainable instead of the whole model.

    Args:
        model: The model whose parameters will be checked.
"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} From {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )


if __name__ == "__main__":
    device = get_device()

    # Use freeze_mode="full" on the local CPU because it is the fastest option.
    # On Kaggle with a GPU, you can use "partial" or "none" to fine-tune
    # more layers and potentially get better accuracy.
    freeze_mode = "full" if device.type == "cpu" else "partial"
    print(f"freeze_mode selected based on device: '{freeze_mode}'")

    model = build_resnet18(freeze_mode=freeze_mode, pretrained=True)
    model = model.to(device)

    count_trainable_params(model)
    
    # Quick test: pass a dummy input to the model to check the output shape.
    dummy_input = torch.randn(4, 3, 96, 96).to(device)  # batch_size=4
    output = model(dummy_input)
    print(f"Model output format (should be [4, 1]): {output.shape}")
