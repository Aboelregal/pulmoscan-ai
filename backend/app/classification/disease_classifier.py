"""
Multi-label chest CT disease classification.

Architecture: 3D DenseNet (MONAI's DenseNet121 with spatial_dims=3) trained
multi-label over the disease taxonomy below. DenseNet's feature reuse via
dense connections works well with the relatively small labeled 3D CT
datasets typically available (vs. training a huge 3D transformer from
scratch), and it's a common, defensible choice reviewers will recognize.

Label set intentionally mirrors publicly available datasets so training
data sourcing is realistic:
  - Pulmonary Nodule / Mass    <- LUNA16 / LIDC-IDRI
  - Lung Cancer (suspicious)   <- LIDC-IDRI malignancy scores
  - Emphysema                  <- COPDGene-style radiographic pattern
  - Pneumonia                  <- MosMedData / MIDRC
  - COVID-19                   <- MosMedData
  - Fibrosis                   <- OSIC Pulmonary Fibrosis dataset
  - Pleural Effusion           <- NSCLC-Radiomics / TCIA annotations
  - Atelectasis                <- NSCLC-Radiomics / TCIA annotations
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from app.core.config import settings

try:
    import torch
    from monai.networks.nets import DenseNet121
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

DISEASE_LABELS = [
    "pulmonary_nodule",
    "lung_cancer_suspicious",
    "emphysema",
    "pneumonia",
    "covid_19",
    "fibrosis",
    "pleural_effusion",
    "atelectasis",
]

SEVERITY_LEVELS = ["none", "mild", "moderate", "severe"]


def build_classifier():
    return DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(DISEASE_LABELS),
    )


class DiseaseClassifier:
    def __init__(self):
        self.model = None
        self.weights_loaded = False
        if _TORCH_AVAILABLE:
            weights_path = Path(settings.WEIGHTS_DIR) / settings.CLASSIFIER_WEIGHTS
            self.model = build_classifier()
            if weights_path.exists():
                state = torch.load(weights_path, map_location=settings.DEVICE)
                self.model.load_state_dict(state)
                self.weights_loaded = True
            self.model.eval()

    def classify(self, volume: np.ndarray, lung_mask: np.ndarray) -> dict:
        """
        Returns {label: {probability, severity}} for every disease in
        DISEASE_LABELS. Falls back to a transparent heuristic (based on lung
        volume affected + intensity stats) when no trained weights exist -
        this is clearly a placeholder and is labeled as such in the output,
        never presented as a diagnosis.
        """
        if self.weights_loaded and _TORCH_AVAILABLE:
            return self._classify_deep(volume)
        return self._classify_heuristic(volume, lung_mask)

    def _classify_deep(self, volume: np.ndarray) -> dict:
        with torch.no_grad():
            tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(settings.DEVICE)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return {
            label: {
                "probability": float(p),
                "severity": self._severity_from_prob(p),
                "source": "model",
            }
            for label, p in zip(DISEASE_LABELS, probs)
        }

    def _classify_heuristic(self, volume: np.ndarray, lung_mask: np.ndarray) -> dict:
        lung_voxels = volume[lung_mask.astype(bool)]
        if lung_voxels.size == 0:
            mean_intensity, opacity_fraction = 0.0, 0.0
        else:
            mean_intensity = float(lung_voxels.mean())
            # ground-glass/consolidation tends to raise intensity above
            # normal aerated lung in the HU window we normalized to.
            opacity_fraction = float((lung_voxels > 0.55).sum() / lung_voxels.size)

        result = {label: {"probability": 0.02, "severity": "none", "source": "heuristic_placeholder"}
                   for label in DISEASE_LABELS}
        result["pneumonia"]["probability"] = round(min(0.9, opacity_fraction * 2.0), 2)
        result["covid_19"]["probability"] = round(min(0.6, opacity_fraction * 1.2), 2)
        result["fibrosis"]["probability"] = round(min(0.5, mean_intensity * 0.3), 2)
        for label in ("pneumonia", "covid_19", "fibrosis"):
            result[label]["severity"] = self._severity_from_prob(result[label]["probability"])
        return result

    @staticmethod
    def _severity_from_prob(p: float) -> str:
        if p < 0.2:
            return "none"
        if p < 0.45:
            return "mild"
        if p < 0.7:
            return "moderate"
        return "severe"


_classifier_singleton: DiseaseClassifier | None = None


def get_disease_classifier() -> DiseaseClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = DiseaseClassifier()
    return _classifier_singleton
