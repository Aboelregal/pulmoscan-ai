import numpy as np
from app.preprocessing.dicom_loader import hu_window_normalize, LUNG_HU_MIN, LUNG_HU_MAX


def test_hu_window_normalize_range():
    volume = np.array([[-2000, -600, 0, 200, 1000]], dtype=np.float32)
    normalized = hu_window_normalize(volume)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_hu_window_normalize_midpoint():
    mid = (LUNG_HU_MIN + LUNG_HU_MAX) / 2
    volume = np.array([[mid]], dtype=np.float32)
    normalized = hu_window_normalize(volume)
    assert abs(normalized[0, 0] - 0.5) < 1e-3
