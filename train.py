import json
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HFPCamDataset, get_default_transforms, load_diverse_pcam_subset
from model import build_resnet18, count_trainable_params, get_device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs one complete training epoch.

    Args:
        model: The model being trained.
        loader: DataLoader for the training data.
        criterion: Loss function, such as BCEWithLogitsLoss.
        optimizer: Optimizer, such as Adam.
        device: The device used for training ('cuda' or 'cpu').

    Returns:
        A tuple containing the average loss and accuracy for this epoch.
"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape [batch, 1] to match the model output.

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Calculate accuracy: since we use raw logits, a threshold of 0
        # is equivalent to a probability of 0.5 after sigmoid.
        predictions = (outputs > 0.0).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation data without computing gradients.

    Args:
        model: The model being evaluated.
        loader: DataLoader for the validation data.
        criterion: Loss function.
        device: The device used for evaluation ('cuda' or 'cpu').

    Returns:
        A tuple containing the average loss and accuracy on the validation data.
"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predictions = (outputs > 0.0).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    n_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    n_samples: int = 5000,
    freeze_mode_override: str = None,
    resume_from: str = None,
) -> Dict[str, List[float]]:
    """
    Runs the complete training process from start to finish.

    Args:
        n_epochs: Number of training epochs.
        batch_size: Number of samples in each batch.
        learning_rate: Learning rate used by the Adam optimizer.
        n_samples: Number of samples to load from Hugging Face.
        freeze_mode_override: If specified, uses this freeze mode instead
            of selecting it automatically based on the device.
        resume_from: Path to an existing checkpoint (.pth). If specified,
            the saved model weights are loaded instead of starting from
            scratch. This is useful for continuing training in another
            environment without losing previous progress.
    """
    device = get_device()

    # Step 1: Load and split the data
    print(f"\nLoading {n_samples} spread-out samples from Hugging Face...")
    # Instead of train[:n_samples], which only covers a few slides,
    # use spread-out sampling to reduce the validation/test gap.
    hf_data = load_diverse_pcam_subset(split="train", n_samples=n_samples, n_chunks=10)

    # Important: Why do we create two separate HFPCamDataset objects
    # instead of using random_split on one dataset?
    # If train and validation share the same Dataset object, changing
    # the transform for one would also affect the other because they
    # share the same transform attribute. Creating two separate datasets
    # from different subsets avoids this problem.
    indices = list(range(len(hf_data)))
    generator = torch.Generator().manual_seed(42)  # For reproducible results.
    shuffled_indices = torch.randperm(len(indices), generator=generator).tolist()

    train_size = int(0.8 * len(indices))
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_hf = hf_data.select(train_indices)
    val_hf = hf_data.select(val_indices)

    train_subset = HFPCamDataset(hf_dataset=train_hf, transform=get_default_transforms(train=True))
    val_subset = HFPCamDataset(hf_dataset=val_hf, transform=get_default_transforms(train=False))

    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    # Step 2: Calculate class weights to handle possible class imbalance.
    train_labels = [int(train_hf[i]["label"]) for i in range(len(train_hf))]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    print(f"Class distribution (train): {n_neg} healthy, {n_pos} tumor")
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight.item():.3f}")

    # Step 3: Create DataLoaders
    # Note: num_workers=0 is safer on Windows/CPU because it avoids multiprocessing issues.
    # On Linux/Kaggle, you can increase this value if needed.
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    ) 
    
    # Step 4: Build the model
    freeze_mode = freeze_mode_override or ("full" if device.type == "cpu" else "partial")
    print(f"freeze_mode: '{freeze_mode}'")
    model = build_resnet18(freeze_mode=freeze_mode, pretrained=True).to(device)

    if resume_from is not None:
        # Note: map_location=device allows a checkpoint saved on CPU to be loaded
        # on a GPU, or vice versa, without device-related errors.
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Loaded weights from {resume_from} — continuing training from this point.")

    count_trainable_params(model)
    
    # Step 5: Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Note: Only parameters with requires_grad=True are passed to the optimizer
    # to avoid unnecessary updates and computations.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Note: ReduceLROnPlateau automatically lowers the learning rate when
    # val_loss does not improve for several consecutive epochs (factor=0.5).
    # This helps the model make smaller updates near convergence instead of
    # overshooting the best point with a fixed learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Step 6: Training loop with best-model checkpointing
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []
    }
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # The scheduler uses val_loss to decide whether to reduce the learning rate.
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"اپوک {epoch}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

    # Save the best model based on val_loss.
    # The filename includes freeze_mode so different runs (full/partial/none)
    # do not overwrite each other.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f"results/checkpoints/best_model_{freeze_mode}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best model saved ({checkpoint_path}, val_loss={val_loss:.4f})")

    # Save the full training history to a JSON file.
    # This is useful later for writing the README or comparing different runs
    # without training the model again.
    history_path = f"results/training_history_{freeze_mode}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"\nTraining history saved to {history_path}")

    plot_training_history(history, freeze_mode)

    return history


def plot_training_history(history: Dict[str, List[float]], freeze_mode: str = "run") -> None:
    """
Plot and save the training and validation loss and accuracy curves.

Args:
    history: Training history returned by train_model.
    freeze_mode: Used in the output filename so plots from different
        runs do not overwrite each other.
"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss During Training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy During Training")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = f"results/figures/training_curves_{freeze_mode}.png"
    plt.savefig(fig_path, dpi=120)
    print(f"Training plot saved to {fig_path}")


if __name__ == "__main__":
    import os

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Quick test of partial unfreezing: the full-freeze model reached about
    # 80-82% validation accuracy, so we now unfreeze layer3 and layer4
    # to see if the model can improve further.
    # If this works well, we can run the full 15-20 epoch training on Kaggle
    # with a GPU using the diverse sample selection.
    history = train_model(
        n_epochs=20,
        batch_size=16,
        n_samples=5000,
        freeze_mode_override="partial",
    )

    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.3f}")
