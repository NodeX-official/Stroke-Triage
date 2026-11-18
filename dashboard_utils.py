# Pure helpers for the Streamlit dashboard (no streamlit import).
# Safety: never displays "administer thrombolysis"; never implies diagnosis;
# pseudonymous study IDs only; conservative language everywhere.

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pydicom
import requests
from PIL import Image

from rules.eka_context import validate_document_upload

API_BASE_URL_DEFAULT = "http://127.0.0.1:8000"

DISCLAIMER = (
    "Research prototype; not clinically validated; "
    "not for autonomous diagnosis or treatment decisions."
)

EKA_DISCLAIMER = (
    "Eka Care output is unconfirmed clinical-context support only. It does not "
    "diagnose stroke, does not determine thrombolysis eligibility, and requires "
    "clinician confirmation."
)

# ---------------------------------------------------------------------------
# Status presentation
# ---------------------------------------------------------------------------

STATUS_META: Dict[str, Dict[str, str]] = {
    "HEMORRHAGE_SUSPECTED": {
        "label": "Hemorrhage suspected",
        "color": "#c62828",
        "recommendation": (
            "Urgent radiologist review. A hemorrhage pattern on CT requires "
            "specialist assessment. Decision support only."
        ),
    },
    "URGENT_RADIOLOGY_REVIEW": {
        "label": "Urgent radiology review",
        "color": "#ef6c00",
        "recommendation": (
            "Prompt radiologist review recommended. Imaging or timing requires "
            "specialist assessment before any pathway decision."
        ),
    },
    "INSUFFICIENT_INFORMATION": {
        "label": "Insufficient information",
        "color": "#616161",
        "recommendation": (
            "Required information is missing (e.g., last-known-well time). "
            "Obtain details before any pathway decision."
        ),
    },
    "OUTSIDE_STANDARD_WINDOW": {
        "label": "Outside standard window",
        "color": "#ef6c00",
        "recommendation": (
            "The recorded time window may be exceeded — specialist review "
            "required. This is not a treatment decision."
        ),
    },
    "POTENTIALLY_ELIGIBLE_PENDING_REVIEW": {
        "label": "Pending clinician review",
        "color": "#2e7d32",
        "recommendation": (
            "Checklist complete pending clinician confirmation. Eligibility is "
            "always a clinician decision — never automatic."
        ),
    },
    "NO_HEMORRHAGE_DETECTED": {
        "label": "No hemorrhage detected",
        "color": "#1565c0",
        "recommendation": (
            "No hemorrhage pattern detected by the model. A negative result "
            "never implies treatment eligibility."
        ),
    },
    "NO_ACUTE_STROKE_EVIDENCE": {
        "label": "No acute stroke evidence",
        "color": "#1565c0",
        "recommendation": (
            "No acute stroke pattern detected by the model. A negative result "
            "never implies treatment eligibility."
        ),
    },
    "ISCHEMIC_PATTERN_POSSIBLE": {
        "label": "Ischemic pattern possible",
        "color": "#1565c0",
        "recommendation": (
            "An ischemic pattern may be present — radiologist review required. "
            "This is not a diagnosis."
        ),
    },
    "failed": {
        "label": "Processing failed",
        "color": "#c62828",
        "recommendation": (
            "No result is available. Nothing was fabricated — retry the upload "
            "or escalate to radiology."
        ),
    },
}

TOOLTIPS = {
    "lkw": (
        "Last known well: the last time the patient was seen in their normal "
        "state. Missing LKW triggers INSUFFICIENT_INFORMATION."
    ),
    "hemorrhage_probability": (
        "Highest Model A hemorrhage probability across processed CT slices. "
        "At or above the 0.50 threshold the result is flagged for urgent "
        "radiology review."
    ),
    "imaging_status": (
        "Raw Model A imaging classification. The rule engine maps this to a "
        "conservative final triage status."
    ),
    "rule_trace": (
        "Identifier of the declarative rule that produced the final status. "
        "Full trace is stored for audit."
    ),
    "clinician_confirmation": (
        "Every output requires a clinician to confirm or override it before "
        "it informs any pathway decision."
    ),
    "eka_context": (
        "Unconfirmed clinical context extracted from Eka Care. Never a "
        "diagnosis, never a contraindication, never an automatic exclusion."
    ),
    "model_version": (
        "Version string stored with every prediction for auditability."
    ),
}


def status_display(status: Optional[str]) -> Dict[str, str]:
    """Return label/color/recommendation for a status (unknown → review)."""
    key = status or "URGENT_RADIOLOGY_REVIEW"
    meta = STATUS_META.get(key)
    if meta is None:
        meta = STATUS_META["URGENT_RADIOLOGY_REVIEW"]
        return {
            "label": key.replace("_", " ").title(),
            "color": meta["color"],
            "recommendation": meta["recommendation"],
        }
    return dict(meta)


def triage_recommendation(status: Optional[str]) -> str:
    """Conservative recommendation text — never a treatment instruction."""
    return status_display(status)["recommendation"]


def patient_pseudonym(study_id: str) -> str:
    """Deterministic pseudonymous patient label (never a real identifier)."""
    if not study_id:
        return "—"
    digest = hashlib.sha256(str(study_id).encode("utf-8")).hexdigest()[:10]
    return f"PT-{digest.upper()}"


def format_probability(value: Any) -> str:
    """Format a 0-1 probability as a percentage string."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "—"
    if 0.0 <= p <= 1.0:
        return f"{p * 100:.1f}%"
    return f"{p:.1f}%"


def _parse_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def window_countdown(lkw_time: Any, current_time: Any) -> Dict[str, Any]:
    """Elapsed hours since LKW, window flag, and remaining hours (or None)."""
    lkw = _parse_time(lkw_time)
    now = _parse_time(current_time) or datetime.now(timezone.utc)
    if lkw is None:
        return {
            "elapsed_hours": None,
            "within_window": None,
            "remaining_hours": None,
        }
    elapsed = round((now - lkw).total_seconds() / 3600.0, 1)
    within = elapsed <= 4.5
    remaining = round(4.5 - elapsed, 1)
    return {
        "elapsed_hours": elapsed,
        "within_window": within,
        "remaining_hours": remaining if within else None,
    }


def allowed_document(filename: str, content_type: str, content: bytes):
    """Delegate to the shared Eka document validator (DICOM blocked)."""
    return validate_document_upload(filename, content_type, content)


def is_dicom_upload(filename: str, content_type: str, content: bytes) -> bool:
    from rules.eka_context import is_dicom_upload as _is_dicom

    return _is_dicom(filename, content_type, content)


# ---------------------------------------------------------------------------
# Slice content handling (DICOM -> PNG conversion; PNG/JPEG accepted directly)
# ---------------------------------------------------------------------------

_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",        # JPEG
    b"GIF87a",
    b"GIF89a",
    b"BM",                  # BMP
    b"II*\x00",             # TIFF little-endian
    b"MM\x00*",             # TIFF big-endian
)


def _is_image_bytes(content: bytes) -> bool:
    return bool(content) and any(content.startswith(m) for m in _IMAGE_MAGIC)


def _is_dicom_bytes(content: bytes) -> bool:
    if not content:
        return False
    if len(content) > 132 and content[128:132] == b"DICM":
        return True
    try:
        pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
        return True
    except Exception:
        return False


def classify_upload(content: bytes) -> Optional[str]:
    """Return 'dicom', 'image', or None (unsupported)."""
    if _is_dicom_bytes(content):
        return "dicom"
    if _is_image_bytes(content):
        try:
            Image.open(io.BytesIO(content)).verify()
            return "image"
        except Exception:
            return None
    return None


def display_normalize(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalized uint8 view for display."""
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi <= lo:
        hi = lo + 1.0
    scaled = (np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def brain_window(arr: np.ndarray, wl: float = 40.0, ww: float = 80.0) -> np.ndarray:
    """Brain window mapping to uint8 (HU values as input)."""
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    scaled = (np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-6) * 255.0
    return scaled.astype(np.uint8)


def dicom_frames(content: bytes) -> List[np.ndarray]:
    """Decode a DICOM (single- or multi-frame) into 2D frames (HU applied)."""
    ds = pydicom.dcmread(io.BytesIO(content))
    pixels = np.asarray(ds.pixel_array, dtype=np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    pixels = pixels * slope + intercept
    if pixels.ndim == 2:
        return [pixels]
    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        return [pixels[..., :3].mean(axis=2)]
    if pixels.ndim == 3:
        return [np.asarray(pixels[i], dtype=np.float32)
                for i in range(pixels.shape[0])]
    return []


def _frame_png_bytes(frame: np.ndarray, is_ct: bool) -> bytes:
    image = brain_window(frame) if is_ct else display_normalize(frame)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def dicom_to_png_slices(content: bytes) -> List[bytes]:
    """Convert a DICOM file (possibly multi-frame) into PNG slice bytes.

    The dashboard uses this so the backend receives standard PNG slices
    instead of DICOM — avoiding transfer-syntax/codec issues entirely.
    """
    ds = pydicom.dcmread(io.BytesIO(content))
    is_ct = str(getattr(ds, "Modality", "")).upper() == "CT"
    return [_frame_png_bytes(frame, is_ct) for frame in dicom_frames(content)]


def load_slice_arrays(files: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
    """Load 2D float arrays for display from DICOM or image bytes."""
    arrays: List[Optional[np.ndarray]] = []
    for item in files:
        content = item.get("display_bytes") or item["bytes"]
        try:
            if _is_dicom_bytes(content):
                arrays.extend(dicom_frames(content))
            else:
                image = Image.open(io.BytesIO(content)).convert("L")
                arrays.append(np.asarray(image, dtype=np.float32))
        except Exception:
            arrays.append(None)
    return arrays


def friendly_error(err: Any) -> str:
    """Turn API/client exceptions into safe, actionable messages."""
    status = getattr(err, "status_code", None)
    detail = getattr(err, "detail", None) or str(err)
    if status == 0:
        return (
            "API unreachable — start the backend with "
            "`uvicorn app.api:app --port 8000`."
        )
    if status == 404:
        return "Not found — check the study ID or reset the session."
    if status == 409:
        return "Study has not finished processing yet."
    if status == 502:
        return "Eka Care request failed — clinical context is unavailable."
    if status == 503:
        return "Eka Care is not configured (set EKA_CLIENT_ID / EKA_CLIENT_SECRET)."
    return str(detail)


def build_audit_export(
    study_id: str,
    result: Optional[Dict[str, Any]],
    eligibility: Optional[Dict[str, Any]],
    audit_events: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Assemble a JSON-safe audit export (no file bytes, no identifiers)."""
    model_a = (result or {}).get("model_a_result") or {}
    return {
        "study_id": study_id,
        "patient_pseudonym": patient_pseudonym(study_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imaging_status": (result or {}).get("imaging_status"),
        "model_a_summary": {
            k: model_a.get(k)
            for k in (
                "max_hemorrhagic_probability",
                "max_ischemic_probability",
                "max_normal_probability",
                "predicted_class",
                "processed_slices",
                "failed_slices",
            )
        },
        "model_a_version": (result or {}).get("model_a_version"),
        "model_b_version": (result or {}).get("model_b_version"),
        "thresholds": (result or {}).get("thresholds"),
        "prediction_timestamp": (result or {}).get("timestamp"),
        "eligibility": {
            "final_status": (eligibility or {}).get("final_status"),
            "rule_trace": (eligibility or {}).get("rule_trace"),
            "elapsed_hours": (eligibility or {}).get("elapsed_hours"),
            "within_standard_window": (eligibility or {}).get(
                "within_standard_window"
            ),
            "requires_radiologist_review": (eligibility or {}).get(
                "requires_radiologist_review"
            ),
            "rationale": (eligibility or {}).get("rationale"),
            "eka_rule_traces": (eligibility or {}).get("eka_rule_traces") or [],
        },
        "audit_events": audit_events or [],
        "disclaimer": DISCLAIMER,
    }


def audit_export_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# API client (requests-based)
# ---------------------------------------------------------------------------


class ApiClientError(Exception):
    """Safe API error — never contains tokens or patient data."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class StrokeApiClient:
    """Thin requests wrapper around the local triage API."""

    def __init__(self, base_url: Optional[str] = None, session=None,
                 timeout: float = 30.0):
        self.base_url = (
            base_url or os.environ.get("API_BASE_URL") or API_BASE_URL_DEFAULT
        ).rstrip("/")
        self._session = session or requests.Session()
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            resp = self._session.request(
                method, self.base_url + path, timeout=timeout, **kwargs
            )
        except requests.RequestException:
            raise ApiClientError(
                0, "API unreachable — start uvicorn app.api:app --port 8000"
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text or "Request failed"
            raise ApiClientError(resp.status_code, str(detail))
        if not resp.content:
            return {}
        try:
            return resp.json()
        except Exception:
            raise ApiClientError(resp.status_code, "Invalid JSON from API")

    # Core triage ---------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def upload_studies(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        multipart = [
            ("files", (f["name"], f["bytes"], f.get("content_type",
                                                     "application/dicom")))
            for f in files
        ]
        return self._request("POST", "/studies/upload", files=multipart)

    def study_status(self, study_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/studies/{study_id}/status")

    def study_result(self, study_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/studies/{study_id}/result")

    def eligibility(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eligibility", json=payload)

    def audit_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/audit", json=payload)

    def audit_list(self, study_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/audit/{study_id}")

    # Grad-CAM explainability --------------------------------------------
    def explain_gradcam(
        self,
        filename: str,
        content: bytes,
        content_type: str = "image/png",
        model: str = "a",
        target_class: Optional[str] = None,
        timeout: float = 90.0,
    ) -> Dict[str, Any]:
        files = {"file": (filename, content, content_type or "image/png")}
        params: Dict[str, Any] = {"model": model}
        if target_class:
            params["target_class"] = target_class
        return self._request(
            "POST", "/explain/gradcam", files=files, params=params,
            timeout=timeout,
        )

    # Eka clinical context ------------------------------------------------
    def eka_start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/assessment/start", json=payload)

    def eka_continue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/assessment/continue", json=payload)

    def eka_submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/assessment/submit", json=payload)

    def eka_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/context", json=payload)

    def eka_upload_document(
        self, filename: str, content: bytes, content_type: str,
        study_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        files = {"file": (filename, content, content_type)}
        data = {"study_id": study_id} if study_id else {}
        return self._request(
            "POST", "/eka/documents/upload", files=files, data=data
        )

    def eka_document_status(self, document_id: str) -> Dict[str, Any]:
        return self._request(
            "GET", f"/eka/documents/{document_id}/status"
        )

    def eka_document_parsed(self, document_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/eka/documents/{document_id}/parsed")
