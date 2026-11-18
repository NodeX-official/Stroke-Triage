# Tests for PNG/JPEG slice input and DICOM -> PNG conversion.
# Safety: unsupported content is rejected explicitly (no fabricated output).

import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.dashboard_utils import (
    classify_upload,
    dicom_to_png_slices,
    load_slice_arrays,
)
from imaging.preprocessing import preprocess_for_model_a

_TMP_DIR = tempfile.mkdtemp(prefix="image_input_test_")


def _png_bytes(size: int = 32, value: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(
        np.full((size, size), value, dtype=np.uint8), mode="L"
    ).save(buffer, format="PNG")
    return buffer.getvalue()


def _dicom_bytes(frames: int = 1, size: int = 16) -> bytes:
    path = os.path.join(_TMP_DIR, f"dicom_{frames}_{size}.dcm")
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
    if frames > 1:
        ds.NumberOfFrames = frames
        pixels = np.arange(frames * size * size, dtype=np.int16).reshape(
            frames, size, size
        )
    else:
        pixels = np.arange(size * size, dtype=np.int16).reshape(size, size)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    with open(path, "rb") as handle:
        return handle.read()


def test_classify_upload():
    assert classify_upload(_dicom_bytes()) == "dicom"
    assert classify_upload(_png_bytes()) == "image"
    jpeg = io.BytesIO()
    Image.fromarray(
        np.full((16, 16), 100, dtype=np.uint8), mode="L"
    ).save(jpeg, format="JPEG")
    assert classify_upload(jpeg.getvalue()) == "image"
    assert classify_upload(b"not a medical image") is None
    assert classify_upload(b"") is None


def test_png_slice_is_preprocessed_without_dicom():
    path = os.path.join(_TMP_DIR, "slice_as_dcm.dcm")
    with open(path, "wb") as handle:
        handle.write(_png_bytes())
    tensor, display = preprocess_for_model_a(filepath=path, target_size=224)
    assert tuple(tensor.shape) == (1, 1, 224, 224)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert display.shape == (224, 224)
    assert display.dtype == np.uint8


def test_dicom_converts_to_png_slices():
    single = dicom_to_png_slices(_dicom_bytes())
    assert len(single) == 1
    assert single[0].startswith(b"\x89PNG")

    multi = dicom_to_png_slices(_dicom_bytes(frames=3))
    assert len(multi) == 3
    for png in multi:
        assert png.startswith(b"\x89PNG")


def test_load_slice_arrays_accepts_both_formats():
    png_item = {"name": "slice.png", "bytes": _png_bytes()}
    dicom_item = {"name": "slice.dcm", "bytes": _dicom_bytes()}

    arrays = load_slice_arrays([png_item, dicom_item])
    assert len(arrays) == 2
    assert arrays[0] is not None and arrays[0].ndim == 2
    assert arrays[1] is not None and arrays[1].ndim == 2
    assert float(arrays[1].max()) == -769.0  # rescale intercept applied


def test_unreadable_content_never_fabricates_arrays():
    arrays = load_slice_arrays([{"name": "junk", "bytes": b"junk"}])
    assert arrays == [None]
