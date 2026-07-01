"""
Explainable AI: Grad-CAM for the 3D disease classifier.

Uses MONAI's GradCAM wrapper (built on the same idea as Selvaraju et al.
2017) against the last convolutional block of the DenseNet3D classifier to
produce a voxel-wise heatmap showing which lung regions most influenced a
given disease prediction. This is what gets rendered as the overlay in the
"Heatmaps" tab of the viewer.
"""
from __future__ import annotations
import numpy as np

from app.core.config import settings
from app.classification.disease_classifier import get_disease_classifier, DISEASE_LABELS

try:
    import torch
    from monai.visualize import GradCAM
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def generate_gradcam(volume: np.ndarray, target_label: str) -> np.ndarray:
    """
    Returns a (Z, Y, X) float32 heatmap in [0,1], same shape as the input
    volume, highlighting voxels driving the prediction for `target_label`.

    If no trained classifier weights are loaded, returns a smooth intensity-
    based saliency proxy (gradient magnitude of the HU volume) so the UI
    still has something meaningful to render in demo mode - clearly labeled
    downstream as "proxy" rather than true Grad-CAM.
    """
    classifier = get_disease_classifier()
    if not (classifier.weights_loaded and _TORCH_AVAILABLE):
        return _saliency_proxy(volume)

    target_idx = DISEASE_LABELS.index(target_label)
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(settings.DEVICE)

    cam = GradCAM(nn_module=classifier.model, target_layers="class_layers.relu")
    result = cam(x=tensor, class_idx=target_idx)
    heatmap = result.squeeze().cpu().numpy()

    # resize heatmap (typically coarser than input) back to full volume res
    from scipy.ndimage import zoom
    zoom_factors = [v / h for v, h in zip(volume.shape, heatmap.shape)]
    heatmap = zoom(heatmap, zoom_factors, order=1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)
    return heatmap.astype(np.float32)


def _saliency_proxy(volume: np.ndarray) -> np.ndarray:
    gx, gy, gz = np.gradient(volume)
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    mag = (mag - mag.min()) / (mag.ptp() + 1e-8)
    return mag.astype(np.float32)
