# Tests for multi-frame DICOM expansion (whole-series-in-one-file exports).
# Safety: unreadable frames are never fabricated; originals are kept on error.

import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_TMP_DIR = tempfile.mkdtemp(prefix="multiframe_test_")
_DB_PATH = os.path.join(_TMP_DIR, "multiframe.db").replace("\\", "/")
os.environ["STUDIES_DB_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SKIP_MODEL_LOAD"] = "1"

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import app.api as api_module


def _make_multiframe(path: str, frames: int = 3, size: int = 32) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.Rows, ds.Columns = size, size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.NumberOfFrames = frames
    pixels = np.arange(frames * size * size, dtype=np.int16).reshape(
        frames, size, size
    )
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_multiframe_expands_into_per_frame_slices():
    path = os.path.join(_TMP_DIR, "series.dcm")
    _make_multiframe(path, frames=3)

    expanded = api_module._expand_multiframe([path])

    assert len(expanded) == 3
    for expected, frame_path in enumerate(expanded, start=1):
        frame = pydicom.dcmread(frame_path)
        assert frame.pixel_array.ndim == 2
        assert int(frame.NumberOfFrames) == 1
        assert int(frame.InstanceNumber) == expected


def test_single_frame_files_are_kept_as_is():
    path = os.path.join(_TMP_DIR, "single.dcm")
    _make_multiframe(path, frames=1)

    assert api_module._expand_multiframe([path]) == [path]


def test_unreadable_files_are_kept_for_safe_failure():
    path = os.path.join(_TMP_DIR, "not_a_dicom.bin")
    with open(path, "wb") as handle:
        handle.write(b"this is not a dicom file")

    assert api_module._expand_multiframe([path]) == [path]
