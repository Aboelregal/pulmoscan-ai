"""
Generates a synthetic Chest CT DICOM series for demoing / testing PulmoScan AI
without needing to download a real dataset first.

This is NOT real patient data and is NOT anatomically accurate - it's a
simple geometric phantom (body cylinder, two lung cavities, one nodule,
one patchy ground-glass-like region) with correct DICOM tags and HU-like
intensities, sufficient to exercise the full upload -> preprocess ->
segment -> detect -> classify -> report pipeline end-to-end.

Usage:
    python generate_demo_ct.py --out demo_ct_series --slices 80
Then zip the output folder and upload it via the Study Browser.
"""
import argparse
import os
import uuid
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def build_phantom_volume(depth=80, height=320, width=320):
    """Returns a (Z, Y, X) int16 volume in Hounsfield Units."""
    vol = np.full((depth, height, width), -1000, dtype=np.int16)  # air

    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2, width / 2

    # body outline (soft tissue, ~40 HU) as an ellipse cross-section, present
    # across the FULL depth so lungs are always fully enclosed (a body
    # "cap" above/below the lungs) - required for the classical
    # border-clearing lung segmentation to work correctly.
    body_mask = ((yy - cy) ** 2 / (140 ** 2) + (xx - cx) ** 2 / (120 ** 2)) < 1

    for z in range(depth):
        vol[z][body_mask] = 40

    # lungs only occupy the middle portion of the volume, well inside the
    # body's z-extent, so they never touch the top/bottom slice faces
    lz0, lz1 = int(depth * 0.15), int(depth * 0.85)
    for z in range(lz0, lz1):
        left_lung = ((yy - cy) ** 2 / (90 ** 2) + (xx - (cx - 55)) ** 2 / (55 ** 2)) < 1
        right_lung = ((yy - cy) ** 2 / (90 ** 2) + (xx - (cx + 55)) ** 2 / (55 ** 2)) < 1
        vol[z][left_lung & body_mask] = -800
        vol[z][right_lung & body_mask] = -800

        # spine (bone, ~700 HU), posterior midline
        spine = ((yy - (cy + 95)) ** 2 / (12 ** 2) + (xx - cx) ** 2 / (14 ** 2)) < 1
        vol[z][spine] = 700

    # a nodule: small, dense (~150 HU), roughly spherical, mid-volume in the
    # right lung
    nz, ny, nx, nr = int(depth * 0.45), int(cy - 20), int(cx + 70), 6
    zz, yy3, xx3 = np.mgrid[0:depth, 0:height, 0:width]
    nodule = ((zz - nz) ** 2 + (yy3 - ny) ** 2 * 0.6 + (xx3 - nx) ** 2 * 0.6) < nr ** 2
    vol[nodule] = 150

    # a patchy ground-glass-like region in the left lower lobe (mild opacity,
    # ~ -400 HU) to give the classifier something to flag
    ggo_z0, ggo_z1 = int(depth * 0.6), int(depth * 0.75)
    ggo = ((yy - (cy + 40)) ** 2 / (35 ** 2) + (xx - (cx - 60)) ** 2 / (35 ** 2)) < 1
    for z in range(ggo_z0, ggo_z1):
        region = ggo & body_mask
        vol[z][region] = np.maximum(vol[z][region], -450)

    return vol


def write_dicom_series(volume: np.ndarray, out_dir: str, spacing=(0.7, 0.7, 2.0)):
    os.makedirs(out_dir, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref_uid = generate_uid()

    depth, height, width = volume.shape
    sx, sy, sz = spacing

    for z in range(depth):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        ds.PatientName = "PulmoScan^Demo"
        ds.PatientID = "DEMO-0001"
        ds.PatientBirthDate = "19700101"
        ds.PatientSex = "O"

        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.FrameOfReferenceUID = frame_of_ref_uid

        ds.StudyDate = "20260701"
        ds.SeriesDate = "20260701"
        ds.StudyDescription = "Synthetic Chest CT (demo phantom)"
        ds.SeriesDescription = "Synthetic axial series"
        ds.SeriesNumber = 1
        ds.InstanceNumber = z + 1

        ds.ImagePositionPatient = [-(width * sx) / 2, -(height * sy) / 2, z * sz]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [sy, sx]
        ds.SliceThickness = sz
        ds.SliceLocation = z * sz

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = height, width
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed

        # Standard DICOM CT rescale: stored_value * slope + intercept = HU
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1

        slice_arr = volume[z].astype(np.int16)
        ds.PixelData = slice_arr.tobytes()

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.save_as(os.path.join(out_dir, f"slice_{z:04d}.dcm"), write_like_original=False)

    print(f"Wrote {depth} DICOM slices to {out_dir}/")
    print(f"StudyInstanceUID: {study_uid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="demo_ct_series", help="output directory")
    parser.add_argument("--slices", type=int, default=80, help="number of axial slices")
    args = parser.parse_args()

    volume = build_phantom_volume(depth=args.slices)
    write_dicom_series(volume, args.out)
