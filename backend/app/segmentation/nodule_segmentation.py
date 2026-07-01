"""
Nodule segmentation using Swin UNETR (MONAI implementation).

Swin UNETR was chosen over a plain 3D U-Net because its shifted-window
self-attention captures long-range context better on small, texture-subtle
lung nodules, and it's become a de-facto strong baseline for 3D medical
segmentation since MSD/BTCV leaderboards. Runs on a small ROI crop around
each candidate detected by the nodule detector, not the whole volume, to
keep inference tractable on CPU.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from app.core.config import settings

try:
    import torch
    from monai.networks.nets import SwinUNETR
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

ROI_SIZE = (64, 64, 64)


def build_swin_unetr():
    return SwinUNETR(
        img_size=ROI_SIZE,
        in_channels=1,
        out_channels=2,  # background, nodule
        feature_size=24,
        use_checkpoint=True,
    )


class NoduleSegmenter:
    def __init__(self):
        self.model = None
        self.weights_loaded = False
        if _TORCH_AVAILABLE:
            weights_path = Path(settings.WEIGHTS_DIR) / settings.NODULE_SEG_WEIGHTS
            self.model = build_swin_unetr()
            if weights_path.exists():
                state = torch.load(weights_path, map_location=settings.DEVICE)
                self.model.load_state_dict(state)
                self.weights_loaded = True
            self.model.eval()

    def segment_roi(self, volume: np.ndarray, center_zyx: tuple[int, int, int]) -> np.ndarray:
        """
        Crops a fixed ROI around a candidate nodule center and returns a
        binary segmentation mask in the *cropped* coordinate frame, along
        with the crop bounds so the caller can paste it back into full volume
        space if desired.
        """
        crop, bounds = self._extract_roi(volume, center_zyx)

        if self.weights_loaded and _TORCH_AVAILABLE:
            with torch.no_grad():
                tensor = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0).to(settings.DEVICE)
                logits = self.model(tensor)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            return pred.astype(np.uint8), bounds

        # Fallback: intensity-based blob segmentation within the ROI, useful
        # for demo purposes when no trained weights are present.
        thresh = crop > (crop.mean() + 0.5 * crop.std())
        return thresh.astype(np.uint8), bounds

    def _extract_roi(self, volume: np.ndarray, center_zyx: tuple[int, int, int]):
        z, y, x = center_zyx
        dz, dy, dx = [s // 2 for s in ROI_SIZE]
        z0, z1 = max(0, z - dz), min(volume.shape[0], z + dz)
        y0, y1 = max(0, y - dy), min(volume.shape[1], y + dy)
        x0, x1 = max(0, x - dx), min(volume.shape[2], x + dx)

        crop = volume[z0:z1, y0:y1, x0:x1]
        pad = [(0, ROI_SIZE[i] - crop.shape[i]) for i in range(3)]
        crop = np.pad(crop, pad, mode="constant", constant_values=0)
        return crop, (z0, z1, y0, y1, x0, x1)


_nodule_segmenter_singleton: NoduleSegmenter | None = None


def get_nodule_segmenter() -> NoduleSegmenter:
    global _nodule_segmenter_singleton
    if _nodule_segmenter_singleton is None:
        _nodule_segmenter_singleton = NoduleSegmenter()
    return _nodule_segmenter_singleton
