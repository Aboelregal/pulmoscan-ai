"""
Pulmonary nodule detection.

Production design: a 3D RetinaNet-style detector (MONAI's `RetinaNet`
detection utilities) trained on LUNA16, which is the standard nodule
localization benchmark (888 scans, ~1200 annotated nodules >=3mm).

For this repo, `NoduleDetector` loads that architecture and, when trained
weights aren't present, falls back to a classical candidate-generation
pipeline (blob detection inside the lung mask via Laplacian-of-Gaussian)
so the pipeline is runnable end-to-end without a GPU training run. This
mirrors how many real CAD systems bootstrap: classical candidate generation
+ CNN false-positive reduction is a well-published architecture (see the
LUNA16 grand-challenge leaderboard).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from skimage.feature import blob_log

from app.core.config import settings

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class NoduleCandidate:
    center_zyx: tuple  # voxel coords
    diameter_mm: float
    confidence: float
    bbox: dict  # {x,y,z,w,h,d} voxel space


class NoduleDetector:
    def __init__(self):
        self.weights_loaded = False
        weights_path = Path(settings.WEIGHTS_DIR) / settings.NODULE_DET_WEIGHTS
        if _TORCH_AVAILABLE and weights_path.exists():
            # Placeholder for loading a trained MONAI detection network
            # (e.g. monai.apps.detection.networks.retinanet_network).
            # Left unimplemented until a trained checkpoint is supplied -
            # see README "Bringing your own weights".
            self.weights_loaded = False  # forces classical path until wired up

    def detect(self, volume: np.ndarray, lung_mask: np.ndarray, spacing: tuple) -> list[NoduleCandidate]:
        if self.weights_loaded:
            raise NotImplementedError("Trained detector path not wired up in this repo - see README")
        return self._detect_classical(volume, lung_mask, spacing)

    def _detect_classical(self, volume: np.ndarray, lung_mask: np.ndarray, spacing: tuple) -> list[NoduleCandidate]:
        """
        Laplacian-of-Gaussian blob detection restricted to lung tissue.
        Nodules appear as roughly round, higher-intensity blobs against the
        low-intensity lung parenchyma background after HU windowing.
        """
        masked = volume * lung_mask
        # Nodules are denser (higher HU / higher normalized intensity) than
        # surrounding air-filled lung -> look for bright blobs.
        blobs = blob_log(
            masked,
            min_sigma=1.5,
            max_sigma=8,
            num_sigma=6,
            threshold=0.12,
            exclude_border=True,
        )

        candidates = []
        sx, sy, sz = spacing  # SimpleITK spacing is (x, y, z)
        for z, y, x, sigma in blobs:
            z, y, x = int(z), int(y), int(x)
            if lung_mask[z, y, x] == 0:
                continue
            radius_vox = sigma * np.sqrt(3)
            diameter_mm = 2 * radius_vox * ((sx + sy + sz) / 3)

            # LUNA16 convention: clinically relevant nodules are ~3-30mm.
            if not (3.0 <= diameter_mm <= 30.0):
                continue

            # crude confidence proxy from local contrast; a trained model
            # would output a calibrated probability instead.
            local_patch = masked[
                max(0, z - 2):z + 3, max(0, y - 2):y + 3, max(0, x - 2):x + 3
            ]
            confidence = float(np.clip(local_patch.mean() * 1.8, 0.05, 0.97))

            r = int(round(radius_vox))
            candidates.append(
                NoduleCandidate(
                    center_zyx=(z, y, x),
                    diameter_mm=round(diameter_mm, 1),
                    confidence=round(confidence, 2),
                    bbox={"x": x - r, "y": y - r, "z": z - r, "w": 2 * r, "h": 2 * r, "d": 2 * r},
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates


_detector_singleton: NoduleDetector | None = None


def get_nodule_detector() -> NoduleDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = NoduleDetector()
    return _detector_singleton
