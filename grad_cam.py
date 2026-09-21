"""
gradcam.py

Implementation of Grad-CAM (Gradient-weighted Class Activation Mapping)
for interpreting the ResNet-18 model trained on the PCam dataset.

Main idea of Grad-CAM:
    Grad-CAM shows which regions of an image have the most influence on
    the model's prediction. It does this by using the gradients of the
    model output with respect to the activations of a convolutional layer.

    Convolutional layers are useful for this because they still keep
    spatial information about the image, unlike the final fully connected
    layers.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (2017)
    https://arxiv.org/abs/1610.02391
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
Grad-CAM implementation for a pretrained CNN model.

This class uses forward and backward hooks to capture the activations
and gradients of a target layer and then uses them to create a heatmap.

Attributes:
    model: CNN model used for generating predictions.
    target_layer: Layer used to capture activations and gradients.
    activations: Activations saved during the forward pass.
    gradients: Gradients saved during the backward pass.
"""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
Initialize the Grad-CAM object and register hooks on the target layer.

Args:
    model: Trained model used for generating the Grad-CAM.
    target_layer: Convolutional layer used to capture activations and
        gradients. For ResNet-18 with large images such as ImageNet
        (224x224), model.layer4[-1] is commonly used.

        For small images such as PCam (96x96), layer3[-1] is a better
        choice because layer4 produces a very small 3x3 activation map,
        while layer3 keeps a larger 6x6 spatial resolution.
"""
       
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor = None
        self.gradients: torch.Tensor = None
        
        # Forward hook: saves the output activations of the target layer.
        self.target_layer.register_forward_hook(self._save_activation)
        # Backward hook: saves the gradients of the target layer's output.
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input_, output: torch.Tensor) -> None:
        """Hook used to save activations during the forward pass."""
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input, grad_output: Tuple[torch.Tensor]) -> None:
        """Hook used to save gradients during the backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
Generate a Grad-CAM heatmap for an input image.

Args:
    input_tensor: Input tensor with shape (1, 3, H, W). The batch size
        must be 1 because Grad-CAM is generated for one image at a time.

Returns:
    A NumPy array with shape (H, W), normalized to values between 0 and 1.
    Higher values indicate regions that had a stronger influence on the
    model's output.

Note:
    This is a binary classification problem with a single output logit.
    Therefore, we can directly backpropagate from the model output without
    selecting a specific class as in multi-class classification.
"""
        self.model.zero_grad()

        # Forward pass — this triggers the activation hook.
        output = self.model(input_tensor)  # Shape: (1, 1)

       # Backward pass from the raw logit — the model has a single output.
        output.backward(gradient=torch.ones_like(output))

        # Gradients and activations have shape (1, C, H', W').
        # C is the number of channels in the target layer (512 for layer4 in ResNet-18).
        gradients = self.gradients[0]      # shape: (C, H', W')
        activations = self.activations[0]  # shape: (C, H', W')

        # Weight for each channel = mean gradient across the spatial dimensions.
        # This is the global average pooling step used in the original Grad-CAM method.
        weights = gradients.mean(dim=(1, 2))  # shape: (C,)

        # Weighted sum of the activations across the channels.
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H', W')
        for c in range(activations.shape[0]):
            cam += weights[c] * activations[c]

        # ReLU: keep only regions that have a positive influence on the model output.
        # Regions with a negative influence are removed.
        cam = F.relu(cam)
 
        # Normalize the heatmap to the range [0, 1] for visualization.
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
 
        # Resize the heatmap to match the input image size (96x96) using interpolation.
        cam = cam.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H', W')
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().numpy()

        return cam


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
Overlay a Grad-CAM heatmap on the original image.

Args:
    image: Original image as a NumPy array with shape (H, W, 3)
        and values between 0 and 1.
    heatmap: Grad-CAM heatmap with shape (H, W) and values between 0 and 1.
    alpha: Opacity of the heatmap. 0 shows only the original image,
        while 1 shows only the heatmap.

Returns:
    A blended image with shape (H, W, 3) and values between 0 and 1,
    ready to be displayed with matplotlib.
"""
    import matplotlib.cm as cm

    # Convert the single-channel heatmap to a colored image using the "jet"
    # colormap. Red indicates high attention, while blue indicates low attention.
    colored_heatmap = cm.jet(heatmap)[:, :, :3]  # Remove the alpha channel.
    overlay = (1 - alpha) * image + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    from dataset import HFPCamDataset, get_default_transforms
    from model import build_resnet18, get_device

    device = get_device()

    # Step 1: Load the trained model
    # Note: Grad-CAM is run on the CPU because it is used only for analysis,
    # not training, so execution speed is less critical.
    model = build_resnet18(freeze_mode="partial", pretrained=False)
    model.load_state_dict(
        torch.load("results/checkpoints/best_model_partial.pth", map_location="cpu")
    )
    model.eval()
    
# Step 2: Create Grad-CAM using layer3 as the target layer (instead of layer4).
# Important: Because PCam images are only 96x96 pixels, after passing
# through the full ResNet-18, the layer4 activation map is only 3x3 pixels.
# This is so small that the resulting Grad-CAM can appear nearly uniform
# and lack spatial detail.
# layer3 produces a 6x6 activation map, providing 4x more spatial detail
# while still retaining sufficiently high-level semantic features.
# This makes it a better balance for small medical images such as PCam.
    grad_cam = GradCAM(model, target_layer=model.layer3[-1])
    
    # Step 3: Load a few validation samples for testing.
    hf_data = load_dataset("1aurent/PatchCamelyon", split="train[4000:4010]")
    val_dataset = HFPCamDataset(hf_dataset=hf_data, transform=get_default_transforms(train=False))

    # Step 4: Generate and display Grad-CAM visualizations for several samples.
    n_samples = 6
    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    for i in range(n_samples):
        image_tensor, label = val_dataset[i]
        input_batch = image_tensor.unsqueeze(0)  # Add the batch dimension.

        heatmap = grad_cam.generate(input_batch)

       # Undo normalization to recover the original image values for visualization.
        img_display = image_tensor.permute(1, 2, 0).numpy()
        img_display = img_display * imagenet_std + imagenet_mean
        img_display = np.clip(img_display, 0, 1)

        overlay = overlay_heatmap(img_display, heatmap)

        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Original ({'Tumor' if label.item() == 1 else 'Healthy'})")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("results/figures/gradcam_examples.png", dpi=120)
    print("Grad-CAM examples saved to results/figures/gradcam_examples.png")
