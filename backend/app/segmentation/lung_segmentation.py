"""
Lung parenchyma segmentation.

Primary path: MONAI SegResNet (3D) trained on lung CT masks (e.g. from
Medical Segmentation Decathlon Task06_Lung or LUNA16 lung masks).

Fallback path: classical HU-threshold + morphology segmentation. This runs
with zero trained weights so the product demo works out of the box; it's
noticeably less precise at lung boundaries than the learned model, which is
exactly the kind of tradeoff you'd document in a real handoff doc.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy import ndimage

from app.core.config import settings

try:
    import torch
    from monai.networks.nets import SegResNet
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def build_segresnet():
    """Instantiate the MONAI SegResNet architecture for binary lung segmentation."""
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # background, lung
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )


class LungSegmenter:
    def __init__(self):
        self.model = None
        self.weights_loaded = False
        if _TORCH_AVAILABLE:
            weights_path = Path(settings.WEIGHTS_DIR) / settings.LUNG_SEG_WEIGHTS
            self.model = build_segresnet()
            if weights_path.exists():
                state = torch.load(weights_path, map_location=settings.DEVICE)
                self.model.load_state_dict(state)
                self.weights_loaded = True
            self.model.eval()

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        volume: normalized (Z, Y, X) float32 array in [0,1] (from dicom_loader).
        returns: binary mask (Z, Y, X) uint8, 1 = lung tissue, 0 = background.
        """
        if self.weights_loaded and _TORCH_AVAILABLE:
            return self._segment_deep(volume)
        return self._segment_classical(volume)

    def _segment_deep(self, volume: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(settings.DEVICE)
            logits = self.model(tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        return pred.astype(np.uint8)

    def _segment_classical(self, volume: np.ndarray) -> np.ndarray:
        """
        Threshold-based lung segmentation:
          1. Air/lung tissue is low-intensity after HU windowing -> threshold.
          2. Remove the body/table by keeping only the largest connected
             air-like region touching the interior (not image border).
          3. Fill holes so vessels/nodules inside the lung stay included.
          4. Morphological opening to smooth boundaries.
        """
        binary = volume < 0.45  # lungs are dark relative to soft tissue in the window

        # clear anything touching the outer border (background air outside body)
        labeled, n = ndimage.label(binary)
        border_labels = set()
        border_labels.update(np.unique(labeled[0, :, :]))
        border_labels.update(np.unique(labeled[-1, :, :]))
        border_labels.update(np.unique(labeled[:, 0, :]))
        border_labels.update(np.unique(labeled[:, -1, :]))
        border_labels.update(np.unique(labeled[:, :, 0]))
        border_labels.update(np.unique(labeled[:, :, -1]))
        border_labels.discard(0)

        mask = np.isin(labeled, list(border_labels), invert=True) & binary
        # keep only reasonably sized components (removes small noise specks)
        labeled2, n2 = ndimage.label(mask)
        if n2 > 0:
            sizes = ndimage.sum(mask, labeled2, range(1, n2 + 1))
            keep = np.argsort(sizes)[::-1][:2] + 1  # left + right lung
            mask = np.isin(labeled2, keep)

        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)))
        return mask.astype(np.uint8)


_segmenter_singleton: LungSegmenter | None = None


def get_lung_segmenter() -> LungSegmenter:
    global _segmenter_singleton
    if _segmenter_singleton is None:
        _segmenter_singleton = LungSegmenter()
    return _segmenter_singleton
