"""
DICOM series loading + preprocessing for Chest CT volumes.

Pipeline:
  1. Load a DICOM series folder into a 3D SimpleITK image (handles slice ordering,
     spacing, orientation).
  2. Resample to isotropic spacing (default 1x1x1 mm) so downstream CNN models
     see consistent voxel geometry regardless of scanner protocol.
  3. Clip to a lung-appropriate Hounsfield Unit window and normalize to [0, 1].
  4. Return both the SimpleITK image (for viewer metadata) and a numpy array
     (for model inference).
"""
from __future__ import annotations
import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path
from dataclasses import dataclass


# Standard chest CT lung window (HU). Radiologists typically view lungs at
# window center -600, width 1500 -> range [-1350, 150].
LUNG_HU_MIN = -1350
LUNG_HU_MAX = 150


@dataclass
class CTVolume:
    array: np.ndarray          # normalized float32 volume, shape (Z, Y, X)
    spacing: tuple              # (x, y, z) mm
    origin: tuple
    direction: tuple
    raw_hu: np.ndarray          # unclipped Hounsfield-unit volume, for measurements
    study_uid: str
    series_uid: str


def load_dicom_series(dicom_dir: str | Path) -> sitk.Image:
    """Read a DICOM series folder into a single 3D SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    # If multiple series exist (e.g. multiple reconstructions), pick the one
    # with the most slices - typically the primary axial acquisition.
    best_files, best_len = None, -1
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)
        if len(files) > best_len:
            best_files, best_len = files, len(files)

    reader.SetFileNames(best_files)
    image = reader.Execute()
    return image


def get_study_series_uids(dicom_dir: str | Path) -> tuple[str, str]:
    """Pull StudyInstanceUID / SeriesInstanceUID from the first DICOM file."""
    files = sorted(Path(dicom_dir).glob("*"))
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            return str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID)
        except Exception:
            continue
    return "unknown-study", "unknown-series"


def resample_isotropic(image: sitk.Image, target_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)  # air HU
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(image)


def hu_window_normalize(volume: np.ndarray, hu_min=LUNG_HU_MIN, hu_max=LUNG_HU_MAX) -> np.ndarray:
    clipped = np.clip(volume, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def preprocess_ct_study(dicom_dir: str | Path, isotropic: bool = True) -> CTVolume:
    """Full preprocessing entrypoint used by the API layer."""
    image = load_dicom_series(dicom_dir)
    study_uid, series_uid = get_study_series_uids(dicom_dir)

    if isotropic:
        image = resample_isotropic(image)

    raw_hu = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)
    normalized = hu_window_normalize(raw_hu)

    return CTVolume(
        array=normalized,
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction=image.GetDirection(),
        raw_hu=raw_hu,
        study_uid=study_uid,
        series_uid=series_uid,
    )


def save_volume_npz(volume: CTVolume, out_path: str | Path) -> None:
    np.savez_compressed(
        out_path,
        array=volume.array,
        raw_hu=volume.raw_hu,
        spacing=volume.spacing,
        origin=volume.origin,
        direction=volume.direction,
    )
