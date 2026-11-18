# Tests for Grad-CAM explainability (research visualization only).
# Verifies CAM math, rendering, and the /explain/gradcam endpoint.

import base64
import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_TMP_DIR = tempfile.mkdtemp(prefix="gradcam_test_")
_DB_PATH = os.path.join(_TMP_DIR, "gradcam.db").replace("\\", "/")
os.environ["STUDIES_DB_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SKIP_MODEL_LOAD"] = "1"

import numpy as np
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

import app.api as api_module
from app.gradcam import gradcam_for_slice, jet_colormap, render_gradcam


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_chans = 1
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
        )
        self.backbone = nn.Module()
        self.backbone.layer4 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU())]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        features = self.features(x)
        features = self.backbone.layer4[0](features)
        return self.fc(self.pool(features).flatten(1))


class _FakeModelService:
    def __init__(self):
        self.model = _TinyNet()
        self.device = torch.device("cpu")
        self.classes = ["normal", "ischemic", "hemorrhagic"]
        self.model_version = "tiny-test-v1"


def _tensor(size=32):
    return torch.rand(1, 1, size, size)


def _png_slice(size=64):
    buffer = io.BytesIO()
    Image.fromarray(
        np.linspace(0, 255, size * size, dtype=np.uint8).reshape(size, size),
        mode="L",
    ).save(buffer, format="PNG")
    return buffer.getvalue()


def test_gradcam_matches_feature_map_and_normalizes():
    service = _FakeModelService()
    result = gradcam_for_slice(service, _tensor())

    assert result["cam"].shape == (32, 32)
    assert result["cam"].min() >= 0.0
    assert result["cam"].max() <= 1.0
    assert set(result["class_probabilities"]) == {"normal", "ischemic",
                                                  "hemorrhagic"}
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-4
    assert result["target_class"] in service.classes


def test_gradcam_respects_explicit_target_class():
    service = _FakeModelService()
    result = gradcam_for_slice(service, _tensor(), target_class="ischemic")
    assert result["target_class"] == "ischemic"


def test_gradcam_rejects_unknown_target_class():
    service = _FakeModelService()
    try:
        gradcam_for_slice(service, _tensor(), target_class="bogus")
    except ValueError:
        return
    raise AssertionError("Unknown target class should raise ValueError")


def test_render_gradcam_shapes_and_colormap():
    cam = np.random.rand(8, 8).astype(np.float32)
    display = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    heatmap, overlay = render_gradcam(display, cam)
    assert heatmap.size == (64, 64) and heatmap.mode == "RGB"
    assert overlay.size == (64, 64) and overlay.mode == "RGB"

    colors = jet_colormap(np.array([[0.0, 1.0]], dtype=np.float32))
    assert colors.shape == (1, 2, 3)
    assert colors.dtype == np.uint8


def test_gradcam_endpoint_returns_png_payloads():
    client = TestClient(api_module.app)
    previous = api_module._model_a_service
    api_module._model_a_service = _FakeModelService()
    try:
        response = client.post(
            "/explain/gradcam",
            files={"file": ("slice.png", _png_slice(), "image/png")},
            params={"model": "a", "target_class": "hemorrhagic"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "a"
        assert payload["target_class"] == "hemorrhagic"
        assert "does not localize lesions" in payload["disclaimer"]

        for key in ("heatmap_png_base64", "overlay_png_base64"):
            raw = base64.b64decode(payload[key])
            assert raw.startswith(b"\x89PNG")
            image = Image.open(io.BytesIO(raw))
            assert image.size == (224, 224)

        bad = client.post(
            "/explain/gradcam",
            files={"file": ("slice.png", _png_slice(), "image/png")},
            params={"model": "a", "target_class": "bogus"},
        )
        assert bad.status_code == 400

        bad_model = client.post(
            "/explain/gradcam",
            files={"file": ("slice.png", _png_slice(), "image/png")},
            params={"model": "c"},
        )
        assert bad_model.status_code == 400
    finally:
        api_module._model_a_service = previous
