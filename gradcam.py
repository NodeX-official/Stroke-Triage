# Grad-CAM explainability for Model A / Model B (research visualization only).
# Grad-CAM highlights regions that influenced a model output. It is NOT a
# lesion localization tool and must never be used to rule in/out hemorrhage
# or to support thrombolysis decisions.

import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

_GRADCAM_LOCK = threading.Lock()

GRADCAM_DISCLAIMER = (
    "Grad-CAM is a research visualization of model attention only. "
    "It does not localize lesions and must not be used to rule in or "
    "rule out hemorrhage or to determine treatment eligibility."
)


class GradCAM:
    """Minimal Grad-CAM (Selvaraju et al., 2017) for CNN classifiers."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._capture_activations),
            target_layer.register_full_backward_hook(self._capture_gradients),
        ]

    def _capture_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _capture_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self, tensor: torch.Tensor, class_index: Optional[int] = None
    ) -> Tuple[np.ndarray, int, Dict[str, float]]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        if class_index is None:
            class_index = int(torch.argmax(logits, dim=1).item())
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy().astype(np.float32)

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        probs = torch.softmax(logits.detach(), dim=1).squeeze(0).cpu().numpy()
        probabilities = {
            str(i): float(p) for i, p in enumerate(probs)
        }
        return cam, class_index, probabilities


def _find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", model)
    layer4 = getattr(backbone, "layer4", None)
    if layer4 is None:
        raise ValueError("Grad-CAM target layer not found in model")
    try:
        return layer4[-1]
    except (TypeError, KeyError):
        return layer4


def _resize_cam(cam: np.ndarray, size: int) -> np.ndarray:
    resized = Image.fromarray(cam.astype(np.float32), mode="F").resize(
        (size, size), Image.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32)


def jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Map a normalized CAM to a jet-like RGB uint8 image (no dependency)."""
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def render_gradcam(
    display_array: np.ndarray, cam: np.ndarray, alpha: float = 0.45
) -> Tuple[Image.Image, Image.Image]:
    """Return (heatmap, overlay) PIL images for a display slice and CAM."""
    size = int(display_array.shape[0])
    cam = _resize_cam(cam, size)
    heatmap = jet_colormap(cam)

    base = display_array.astype(np.float32)
    if base.max() <= 1.0:
        base = base * 255.0
    base_rgb = np.stack([base] * 3, axis=-1)
    overlay = np.clip(
        (1.0 - alpha) * base_rgb + alpha * heatmap.astype(np.float32), 0, 255
    ).astype(np.uint8)

    return Image.fromarray(heatmap, mode="RGB"), Image.fromarray(overlay, mode="RGB")


def gradcam_for_slice(
    service: Any,
    tensor: torch.Tensor,
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Grad-CAM for a model service on one preprocessed slice tensor."""
    model = service.model
    device = getattr(service, "device", torch.device("cpu"))
    classes = list(getattr(service, "classes", []))

    tensor = tensor.to(device).float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    in_chans = int(getattr(model, "in_chans", 1) or 1)
    if tensor.shape[1] == 1 and in_chans != 1:
        tensor = tensor.repeat(1, in_chans, 1, 1)

    target_index: Optional[int] = None
    if target_class is not None:
        if target_class not in classes:
            raise ValueError(
                f"Unknown target class '{target_class}'. "
                f"Expected one of: {', '.join(classes)}"
            )
        target_index = classes.index(target_class)

    model.eval()
    cam_engine = GradCAM(model, _find_target_layer(model))
    try:
        with _GRADCAM_LOCK:
            cam, class_index, raw_probs = cam_engine(tensor, target_index)
    finally:
        cam_engine.remove()

    probabilities = {
        classes[i] if i < len(classes) else str(i): float(p)
        for i, p in enumerate(raw_probs.values())
    }

    return {
        "cam": cam,
        "class_index": class_index,
        "target_class": (
            classes[class_index] if class_index < len(classes) else "unknown"
        ),
        "confidence": float(probabilities.get(
            classes[class_index] if class_index < len(classes) else "", 0.0
        )),
        "class_probabilities": probabilities,
        "model_version": getattr(service, "model_version", "unknown"),
    }
