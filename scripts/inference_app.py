from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from PyQt6.QtCore import QDate, QEvent, QPoint, QPointF, QRectF, QSize, Qt, QTime, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette, QPen, QPixmap, QPolygonF, QTextCursor, QStandardItem
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from utils.config_utils import load_yaml, resolve_path, save_yaml
from utils.inference_utils import (
    camera_capture_can_read,
    open_capture,
    open_video_file_capture,
    resolve_security_mode,
    scan_available_cameras,
    should_raise_alert,
)
from utils.model_utils import load_yolo_model
from utils.runtime_env_utils import ensure_windows_compile_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EVENT_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"}
QIMAGE_BGR888 = getattr(QImage.Format, "Format_BGR888", None)
DEFAULT_DAY_SEG_MODEL_NAME = "yolo26s-seg.pt"
DAY_SEG_MODEL_SUGGESTIONS = ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt"]
UNIFORM_TOP_DEFAULT = "#2F5FA8"
UNIFORM_BOTTOM_DEFAULT = "#1F2430"
UNIFORM_COLOR_TOLERANCE_DEFAULT = 42.0
UNIFORM_COLOR_TOLERANCE_MIN = 0.0
UNIFORM_COLOR_TOLERANCE_MAX = 100.0
UNIFORM_MIN_MASK_PIXELS_DEFAULT = 180


@dataclass
class PersonDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    track_id: int | None = None
    is_intruder: bool = False
    uniform_match: bool | None = None
    upper_match: bool | None = None
    lower_match: bool | None = None
    upper_color_hex: str = ""
    lower_color_hex: str = ""
    has_segmentation: bool = False
    label: str = ""
    visible_section_count: int = 0
    uniform_cached_decision: bool = False


def _normalize_hex_color(value: Any, fallback: str) -> str:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) != 6:
        return fallback.upper()
    try:
        int(raw, 16)
    except Exception:  # noqa: BLE001
        return fallback.upper()
    return f"#{raw.upper()}"


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    normalized = _normalize_hex_color(value, "#000000")
    red = int(normalized[1:3], 16)
    green = int(normalized[3:5], 16)
    blue = int(normalized[5:7], 16)
    return blue, green, red


def _bgr_to_hex(color: tuple[int, int, int] | np.ndarray) -> str:
    blue, green, red = [int(max(0, min(255, channel))) for channel in color]
    return f"#{red:02X}{green:02X}{blue:02X}"


def _lab_color_distance(color_a: tuple[int, int, int], color_b: tuple[int, int, int]) -> float:
    sample = np.array([[list(color_a), list(color_b)]], dtype=np.uint8)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    first = lab[0, 0].astype(np.float32)
    second = lab[0, 1].astype(np.float32)
    return float(np.linalg.norm(first - second))


def _bgr_hsv(color: tuple[int, int, int]) -> tuple[int, int, int]:
    sample = np.array([[list(color)]], dtype=np.uint8)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _hue_distance(hue_a: int, hue_b: int) -> int:
    diff = abs(int(hue_a) - int(hue_b))
    return min(diff, 180 - diff)


def _uniform_color_matches(
    sample_bgr: tuple[int, int, int],
    target_bgr: tuple[int, int, int],
    tolerance: float,
) -> bool:
    distance = _lab_color_distance(sample_bgr, target_bgr)
    if distance <= tolerance:
        return True

    target_hue, target_sat, target_value = _bgr_hsv(target_bgr)
    sample_hue, sample_sat, sample_value = _bgr_hsv(sample_bgr)

    if target_sat > 28 or sample_sat > 64:
        if target_sat <= 28 or sample_sat <= 18:
            return False

        hue_limit = float(_clamp(4.0 + (float(tolerance) * 0.18), 6.0, 20.0))
        relaxed_tolerance = min(115.0, (float(tolerance) * 1.35) + 10.0)
        saturation_gap = abs(float(sample_sat) - float(target_sat))
        value_gap = abs(float(sample_value) - float(target_value))
        saturation_limit = 70.0 + (float(tolerance) * 0.70)
        value_limit = 70.0 + (float(tolerance) * 1.05)
        return (
            _hue_distance(sample_hue, target_hue) <= hue_limit
            and distance <= relaxed_tolerance
            and saturation_gap <= saturation_limit
            and value_gap <= value_limit
        )

    relaxed_tolerance = min(UNIFORM_COLOR_TOLERANCE_MAX, (float(tolerance) * 1.25) + 6.0)
    if distance > relaxed_tolerance:
        return False
    if target_value >= 180:
        min_brightness = max(130.0, float(target_value) - (float(tolerance) * 2.2))
        return float(sample_value) >= min_brightness
    if target_value <= 55:
        max_brightness = min(110.0, float(target_value) + (float(tolerance) * 1.5) + 18.0)
        return float(sample_value) <= max_brightness
    return True


def _compute_region_color(
    frame: np.ndarray,
    mask: np.ndarray,
    y_start: int,
    y_end: int,
    *,
    center_band_fraction: float = 1.0,
) -> tuple[str, int] | None:
    if frame.ndim != 3 or mask.ndim != 2:
        return None
    height = frame.shape[0]
    y1 = max(0, min(height, int(y_start)))
    y2 = max(y1, min(height, int(y_end)))
    if y2 <= y1:
        return None
    region_mask = mask[y1:y2, :] > 0
    if not np.any(region_mask):
        return None

    # Optionally ignore silhouette edges (hands/held objects) and sample torso/legs center band.
    band_fraction = _clamp(float(center_band_fraction), 0.2, 1.0)
    if band_fraction < 0.999:
        _, xs = np.where(region_mask)
        if xs.size >= 16:
            side_trim = (1.0 - band_fraction) * 50.0
            x_lo = int(np.percentile(xs, side_trim))
            x_hi = int(np.percentile(xs, 100.0 - side_trim))
            if x_hi > x_lo:
                x_coords = np.arange(region_mask.shape[1], dtype=np.int32)
                center_cols = (x_coords >= x_lo) & (x_coords <= x_hi)
                center_mask = region_mask & center_cols[None, :]
                if np.any(center_mask):
                    region_mask = center_mask

    region_pixels = frame[y1:y2, :, :][region_mask]
    if region_pixels.size == 0:
        return None
    median_color = np.median(region_pixels, axis=0)
    color = tuple(int(max(0, min(255, round(float(channel))))) for channel in median_color)
    return _bgr_to_hex(color), int(region_pixels.shape[0])


def _mask_row_percentile_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if getattr(mask, "ndim", 0) != 2 or mask.size == 0:
        return None
    row_counts = np.count_nonzero(mask, axis=1)
    total = int(row_counts.sum())
    if total <= 0:
        return None

    cumulative = np.cumsum(row_counts, dtype=np.int64)

    def _locate(fraction: float) -> int:
        threshold = max(0, min(total - 1, int(math.floor(float(fraction) * total))))
        return int(np.searchsorted(cumulative, threshold, side="left"))

    upper_start = _locate(0.20)
    upper_end = _locate(0.60)
    lower_start = upper_end
    lower_end = _locate(0.95) + 1

    upper_end = max(upper_start + 1, upper_end)
    lower_start = max(upper_end, lower_start)
    lower_end = max(lower_start + 1, lower_end)
    return upper_start, upper_end, lower_start, lower_end


def _clip_box_to_frame(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    if len(frame_shape) < 2:
        return None
    frame_h = int(frame_shape[0])
    frame_w = int(frame_shape[1])
    left = max(0, min(frame_w, int(x1)))
    top = max(0, min(frame_h, int(y1)))
    right = max(left, min(frame_w, int(x2)))
    bottom = max(top, min(frame_h, int(y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _extract_mask_roi_for_detection(
    seg_mask: np.ndarray | None,
    detection: PersonDetection,
    frame_shape: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    if seg_mask is None or getattr(seg_mask, "ndim", 0) != 2:
        return None

    clipped = _clip_box_to_frame(detection.x1, detection.y1, detection.x2, detection.y2, frame_shape)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped

    mask_h, mask_w = seg_mask.shape[:2]
    frame_h = max(1, int(frame_shape[0]))
    frame_w = max(1, int(frame_shape[1]))
    scale_x = float(mask_w) / float(frame_w)
    scale_y = float(mask_h) / float(frame_h)

    mx1 = max(0, min(mask_w, int(math.floor(x1 * scale_x))))
    my1 = max(0, min(mask_h, int(math.floor(y1 * scale_y))))
    mx2 = max(mx1 + 1, min(mask_w, int(math.ceil(x2 * scale_x))))
    my2 = max(my1 + 1, min(mask_h, int(math.ceil(y2 * scale_y))))
    if mx2 <= mx1 or my2 <= my1:
        return None

    roi_mask = seg_mask[my1:my2, mx1:mx2]
    if roi_mask.size == 0:
        return None

    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    if roi_mask.shape[1] != target_w or roi_mask.shape[0] != target_h:
        try:
            roi_mask = cv2.resize(roi_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        except Exception:  # noqa: BLE001
            return None

    if roi_mask.dtype == np.bool_:
        roi_mask = roi_mask.astype(np.uint8) * 255
    elif roi_mask.dtype != np.uint8:
        roi_mask = roi_mask > 0.5 if np.issubdtype(roi_mask.dtype, np.floating) else roi_mask > 0
        roi_mask = roi_mask.astype(np.uint8) * 255
    else:
        roi_mask = np.where(roi_mask > 0, 255, 0).astype(np.uint8, copy=False)
    roi_mask = np.ascontiguousarray(roi_mask)
    if not np.any(roi_mask):
        return None
    return roi_mask, clipped


def _build_day_segmentation_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    base_dir = str(model_cfg.get("local_weights_dir", "models/base"))
    raw_cfg = model_cfg.get("day_segmentation", {}) or {}
    seg_cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
    seg_cfg.setdefault("enabled", True)
    seg_cfg.setdefault("name", DEFAULT_DAY_SEG_MODEL_NAME)
    seg_cfg.setdefault("local_weights_dir", base_dir)
    seg_cfg.setdefault("auto_download", bool(model_cfg.get("auto_download", True)))
    seg_cfg.setdefault("fallback_name", None)
    seg_cfg.setdefault("use_fallback", False)
    seg_cfg.setdefault("force_redownload", False)
    seg_cfg.setdefault("download_url", None)
    seg_cfg.setdefault("selected_model_path", "")
    return seg_cfg

YOLO_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "low": {
        "title": "Low",
        "combo_label": "Low - szybki i lekki",
        "description": "Najlepszy dla slabszego sprzetu lub wielu kamer jednoczesnie.",
        "model_name": "yolo26n.pt",
        "day_seg_model_name": "yolo26n-seg.pt",
        "target_fps": 30.0,
        "conf": 0.24,
        "iou": 0.40,
        "imgsz": 640,
        "max_det": 40,
    },
    "medium": {
        "title": "Medium",
        "combo_label": "Medium - balans",
        "description": "Dobry kompromis miedzy dokladnoscia i plynnoscia na przecietnym GPU.",
        "model_name": "yolo26s.pt",
        "day_seg_model_name": "yolo26s-seg.pt",
        "target_fps": 20.0,
        "conf": 0.27,
        "iou": 0.42,
        "imgsz": 960,
        "max_det": 50,
    },
    "high": {
        "title": "High",
        "combo_label": "High - dokladnosc",
        "description": "Lepsza jakosc detekcji kosztem szybkosci, dobra dla mocniejszego sprzetu.",
        "model_name": "yolo26m.pt",
        "day_seg_model_name": "yolo26m-seg.pt",
        "target_fps": 12.0,
        "conf": 0.30,
        "iou": 0.45,
        "imgsz": 1280,
        "max_det": 60,
    },
}

YOLO_PROFILE_CUSTOM = "custom"
YOLO_IMGSZ_OPTIONS = [512, 640, 736, 960, 1024, 1280, 1536]


def _is_compile_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized not in {"", "0", "false", "none", "off", "no"}


def _safe_name(raw_name: str, fallback: str) -> str:
    candidate = str(raw_name or "").strip()
    if candidate:
        return candidate
    return fallback


def _ensure_unique_name(existing: set[str], base_name: str) -> str:
    if base_name not in existing:
        return base_name
    index = 2
    while True:
        candidate = f"{base_name}_{index}"
        if candidate not in existing:
            return candidate
        index += 1


def _safe_file_part(value: str, fallback: str = "source") -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    allowed = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    cleaned = "".join(allowed).strip("_")
    return cleaned or fallback


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _uniform_color_tolerance(value: Any) -> float:
    try:
        raw = float(value)
    except Exception:  # noqa: BLE001
        raw = UNIFORM_COLOR_TOLERANCE_DEFAULT
    return float(_clamp(raw, UNIFORM_COLOR_TOLERANCE_MIN, UNIFORM_COLOR_TOLERANCE_MAX))


def _ema(previous: float, value: float, alpha: float) -> float:
    if previous <= 0.0:
        return value
    return previous + alpha * (value - previous)


def _quantize_fps(value: float, step: float = 0.5) -> float:
    if value <= 0.0:
        return 0.0
    safe_step = max(0.1, float(step))
    return round(value / safe_step) * safe_step


def _to_relative_or_abs(path_value: Path) -> str:
    try:
        return str(path_value.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:  # noqa: BLE001
        return str(path_value.resolve())


def _scaled_annotation_style(
    frame: np.ndarray,
    *,
    reference_height: float = 720.0,
    label_font_base: float = 0.48,
    status_font_base: float = 0.46,
    min_scale: float = 0.45,
    max_scale: float = 4.0,
) -> dict[str, float | int]:
    frame_h = max(1, int(frame.shape[0]) if getattr(frame, "ndim", 0) >= 2 else 720)
    reference = max(120.0, float(reference_height))
    resolution_scale = float(_clamp(frame_h / reference, float(min_scale), float(max_scale)))
    label_font_scale = float(_clamp(float(label_font_base) * resolution_scale, 0.18, 2.4))
    status_font_scale = float(_clamp(float(status_font_base) * resolution_scale, 0.18, 2.2))
    text_thickness = max(1, int(round(1.25 * resolution_scale)))
    box_thickness = max(1, int(round(1.75 * resolution_scale)))
    pad_x = max(3, int(round(4.0 * resolution_scale)))
    pad_y = max(2, int(round(3.0 * resolution_scale)))
    gap = max(3, int(round(6.0 * resolution_scale)))
    status_x = max(4, int(round(10.0 * resolution_scale)))
    status_y = max(14, int(round(24.0 * resolution_scale)))
    return {
        "label_font_scale": label_font_scale,
        "status_font_scale": status_font_scale,
        "text_thickness": text_thickness,
        "box_thickness": box_thickness,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "gap": gap,
        "status_x": status_x,
        "status_y": status_y,
    }


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    minutes = total // 60
    sec = total % 60
    return f"{minutes:02d}:{sec:02d}"


def _configure_opencv_logging(*, silent: bool) -> None:
    target_name = "SILENT" if silent else "ERROR"
    os.environ["OPENCV_LOG_LEVEL"] = target_name

    # OpenCV Python API differs by version, so try both interfaces.
    level_name = "LOG_LEVEL_SILENT" if silent else "LOG_LEVEL_ERROR"
    level_numeric = 0 if silent else 2

    try:
        if hasattr(cv2, level_name):
            level_numeric = int(getattr(cv2, level_name))
    except Exception:  # noqa: BLE001
        pass

    try:
        utils_logging = getattr(getattr(cv2, "utils", None), "logging", None)
        if utils_logging is not None and hasattr(utils_logging, "setLogLevel"):
            if hasattr(utils_logging, level_name):
                level_numeric = int(getattr(utils_logging, level_name))
            utils_logging.setLogLevel(level_numeric)
    except Exception:  # noqa: BLE001
        pass

    try:
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(level_numeric)
    except Exception:  # noqa: BLE001
        pass


def _resolve_tracker_backend(backend_name: str) -> tuple[type[Any] | None, str | None]:
    if importlib.util.find_spec("lap") is None:
        return None, "missing dependency 'lap' in active environment"

    normalized = str(backend_name or "bytetrack").strip().lower()
    try:
        if normalized == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT as tracker_cls
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker as tracker_cls
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    return tracker_cls, None


def _extract_run_name_from_weight_filename(filename: str) -> str | None:
    for suffix in ("_best.pt", "_last.pt"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def _infer_model_family(name: str) -> str:
    text = str(name or "").lower()
    match = re.search(r"(yolo(?:v?\d+)[nslmx]?)", text)
    if match is not None:
        return str(match.group(1)).lower()
    return "-"


def _parse_model_series_and_size(name: str) -> tuple[str, str]:
    text = str(name or "").strip().lower()
    match = re.search(r"(yolo(?:v?\d+))(?:[-_])?([nslmx])", text)
    if match is None:
        return "", ""
    return str(match.group(1)), str(match.group(2))


def _model_size_rank(size_key: str) -> int:
    ranks = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}
    return int(ranks.get(str(size_key or "").strip().lower(), 99))


def _read_model_meta(path: Path) -> dict[str, Any] | None:
    candidates = [
        Path(f"{path}.meta.json"),
        path.with_suffix(".meta.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(raw, dict):
            return raw
    return None


def _parse_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _read_last_run_metrics(results_csv: Path) -> dict[str, Any] | None:
    if not results_csv.exists():
        return None

    try:
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        return None

    if not rows:
        return None

    best_map50: float | None = None
    best_map5095: float | None = None
    for row in rows:
        map50 = _parse_float(row, ["metrics/mAP50(B)", "metrics/mAP50"])
        if map50 is not None and (best_map50 is None or map50 > best_map50):
            best_map50 = map50

        map5095 = _parse_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
        if map5095 is not None and (best_map5095 is None or map5095 > best_map5095):
            best_map5095 = map5095

    return {
        "map50": best_map50,
        "map5095": best_map5095,
        "updated_ts": results_csv.stat().st_mtime,
    }


def _frame_cache_key(frame: np.ndarray | None) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    if frame is None:
        return None
    try:
        data_ptr = int(frame.__array_interface__["data"][0])
    except Exception:  # noqa: BLE001
        return None
    return data_ptr, tuple(int(dim) for dim in frame.shape), tuple(int(step) for step in frame.strides)


def _frame_to_pixmap(frame: np.ndarray | None) -> QPixmap:
    if frame is None or frame.size == 0:
        return QPixmap()

    source = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)
    height, width = source.shape[:2]
    bytes_per_line = int(source.strides[0])

    try:
        if QIMAGE_BGR888 is not None:
            image = QImage(source.data, width, height, bytes_per_line, QIMAGE_BGR888)
        else:
            rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            image = QImage(
                rgb.data,
                width,
                height,
                int(rgb.strides[0]),
                QImage.Format.Format_RGB888,
            )
        return QPixmap.fromImage(image)
    except Exception:  # noqa: BLE001
        return QPixmap()


def _open_event_video_writer(
    base_path: Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter | None, Path | None]:
    width, height = frame_size
    if width <= 1 or height <= 1:
        return None, None

    target_fps = max(1.0, float(fps))
    writer_candidates = [
        (".mp4", ("mp4v", "avc1", "H264")),
        (".avi", ("XVID", "MJPG")),
    ]

    for suffix, codec_candidates in writer_candidates:
        output_path = base_path.with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for codec in codec_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
            except Exception:  # noqa: BLE001
                writer = None
            if writer is not None and writer.isOpened():
                return writer, output_path
            if writer is not None:
                try:
                    writer.release()
                except Exception:  # noqa: BLE001
                    pass
    return None, None


@dataclass
class SourceRuntime:
    source: dict[str, Any]
    capture: cv2.VideoCapture | None = None
    status: str = "idle"
    fps: float = 0.0
    infer_fps: float = 0.0
    last_tick_ts: float = 0.0
    last_infer_ts: float = 0.0
    source_fps: float = 0.0
    last_capture_frame_ts: float = 0.0
    playback_interval_sec: float = 0.0
    last_frame_due_ts: float = 0.0
    ui_fps: float = 0.0
    smoothed_source_fps: float = 0.0
    smoothed_view_fps: float = 0.0
    smoothed_infer_fps: float = 0.0
    display_source_fps: float = 0.0
    display_view_fps: float = 0.0
    display_infer_fps: float = 0.0
    last_meta_fps_update_ts: float = 0.0
    last_render_ts: float = 0.0
    person_count: int = 0
    intruder_count: int = 0
    alert: bool = False
    mode: str = "day"
    last_boxes: list[PersonDetection] | None = None
    detection_box_memory: list[tuple[PersonDetection, float]] = field(default_factory=list)
    last_tracked_boxes: list[PersonDetection] | None = None
    last_input: np.ndarray | None = None
    last_output: np.ndarray | None = None
    capture_reader_thread: threading.Thread | None = None
    capture_reader_stop_event: threading.Event | None = None
    capture_latest_frame: np.ndarray | None = None
    capture_latest_seq: int = 0
    capture_last_consumed_seq: int = 0
    last_decorated_capture_seq: int = 0
    last_decorated_infer_ts: float = 0.0
    no_frame_refresh_needed: bool = True
    person_visible_since_ts: float = 0.0
    person_visible_duration_sec: float = 0.0
    last_event_capture_ts: float = 0.0
    event_saved_in_streak: bool = False
    event_clip_started_wall_ts: float = 0.0
    event_last_seen_ts: float = 0.0
    event_max_person_count: int = 0
    event_max_intruder_count: int = 0
    event_clip_temp_path: Path | None = None
    event_clip_writer: cv2.VideoWriter | None = None
    event_clip_frame_size: tuple[int, int] | None = None
    event_clip_frames_written: int = 0
    event_clip_last_enqueue_ts: float = 0.0
    event_prebuffer_frames: deque[tuple[float, np.ndarray]] = field(default_factory=deque)
    event_prebuffer_last_store_ts: float = 0.0
    event_clip_generation: int = 0
    event_clip_failed: bool = False
    last_tracker_update_ts: float = 0.0
    camera_settings_rejected: bool = False

    def release(self) -> None:
        if self.capture_reader_stop_event is not None:
            self.capture_reader_stop_event.set()

        if self.capture_reader_thread is not None and self.capture_reader_thread.is_alive():
            self.capture_reader_thread.join(timeout=1.5)

        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:  # noqa: BLE001
                pass
            self.capture = None

        self.capture_reader_thread = None
        self.capture_reader_stop_event = None
        self.last_input = None
        self.last_tracked_boxes = None
        self.last_tracker_update_ts = 0.0
        self.capture_latest_frame = None
        self.capture_latest_seq = 0
        self.capture_last_consumed_seq = 0
        if self.event_clip_writer is not None:
            try:
                self.event_clip_writer.release()
            except Exception:  # noqa: BLE001
                pass
        if self.event_clip_temp_path is not None:
            try:
                if self.event_clip_temp_path.exists():
                    self.event_clip_temp_path.unlink()
            except Exception:  # noqa: BLE001
                pass
        self.event_clip_writer = None
        self.event_clip_temp_path = None
        self.event_clip_frame_size = None
        self.event_clip_frames_written = 0
        self.event_clip_last_enqueue_ts = 0.0
        self.event_prebuffer_frames.clear()
        self.event_prebuffer_last_store_ts = 0.0
        self.event_clip_started_wall_ts = 0.0
        self.event_max_person_count = 0
        self.event_max_intruder_count = 0


@dataclass
class AsyncInferenceResult:
    infer_ts: float
    person_count: int
    intruder_count: int
    mode: str
    alert: bool
    boxes: list[PersonDetection]


class VideoCanvas(QLabel):
    clicked = pyqtSignal(str)
    right_clicked = pyqtSignal(str)
    zoom_delta = pyqtSignal(str, int)
    pan_delta = pyqtSignal(str, float, float)

    def __init__(self, source_name: str) -> None:
        super().__init__()
        self.source_name = source_name
        self._frame: np.ndarray | None = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._expand_mode = False
        self._drag_active = False
        self._drag_last_pos: QPoint | None = None
        self._base_pixmap = QPixmap()
        self._frame_cache_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None
        self._last_render_signature: tuple[Any, ...] | None = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(64, 36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #0b0d11; color: #888; border: 0px;")
        self.setText(f"{source_name}\\nBrak klatki")

    def set_expand_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._expand_mode == enabled:
            return
        self._expand_mode = enabled
        self._refresh_pixmap()

    def set_frame(
        self,
        frame: np.ndarray | None,
        *,
        zoom: float = 1.0,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
    ) -> None:
        frame_key = _frame_cache_key(frame)
        if frame_key != self._frame_cache_key:
            self._frame_cache_key = frame_key
            self._base_pixmap = _frame_to_pixmap(frame)
            self._last_render_signature = None
        self._frame = frame
        self._zoom = _clamp(float(zoom), 1.0, 8.0)
        self._pan_x = _clamp(float(pan_x), -1.0, 1.0)
        self._pan_y = _clamp(float(pan_y), -1.0, 1.0)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._frame is None:
            self._last_render_signature = None
            self.setPixmap(QPixmap())
            self.setText(f"{self.source_name}\\nBrak klatki")
            return

        pixmap = self._base_pixmap
        if pixmap.isNull():
            self._last_render_signature = None
            return

        self.setText("")
        if self._expand_mode:
            transform = Qt.TransformationMode.SmoothTransformation
        else:
            # In grid mode prioritize throughput over visual smoothing.
            transform = Qt.TransformationMode.FastTransformation

        view_w = max(1, int(self.width()))
        view_h = max(1, int(self.height()))
        zoom = max(1.0, float(self._zoom))
        render_signature = (
            self._frame_cache_key,
            view_w,
            view_h,
            round(zoom, 3),
            round(float(self._pan_x), 4),
            round(float(self._pan_y), 4),
            self._expand_mode,
        )
        if render_signature == self._last_render_signature:
            return

        if zoom <= 1.01:
            self.setScaledContents(False)
            scaled = pixmap.scaled(
                QSize(view_w, view_h),
                Qt.AspectRatioMode.KeepAspectRatio,
                transform,
            )
            self.setPixmap(scaled)
            self._last_render_signature = render_signature
            return

        target_w = max(1, int(view_w * zoom))
        target_h = max(1, int(view_h * zoom))
        scaled = pixmap.scaled(
            QSize(target_w, target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            transform,
        )

        if scaled.width() <= view_w and scaled.height() <= view_h:
            self.setScaledContents(False)
            self.setPixmap(scaled)
            self._last_render_signature = render_signature
            return

        max_shift_x = max(0, int((scaled.width() - view_w) / 2))
        max_shift_y = max(0, int((scaled.height() - view_h) / 2))
        center_x = int((scaled.width() / 2) + _clamp(self._pan_x, -1.0, 1.0) * max_shift_x)
        center_y = int((scaled.height() / 2) + _clamp(self._pan_y, -1.0, 1.0) * max_shift_y)

        x0 = int(_clamp(center_x - (view_w / 2), 0, max(0, scaled.width() - view_w)))
        y0 = int(_clamp(center_y - (view_h / 2), 0, max(0, scaled.height() - view_h)))

        cropped = scaled.copy(x0, y0, view_w, view_h)
        self.setScaledContents(False)
        self.setPixmap(cropped)
        self._last_render_signature = render_signature

    def resizeEvent(self, event: Any) -> None:  # noqa: ANN401
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event: Any) -> None:  # noqa: ANN401
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.source_name)
            self._drag_active = True
            self._drag_last_pos = event.position().toPoint()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self.source_name)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: ANN401
        if self._drag_active and self._drag_last_pos is not None:
            point = event.position().toPoint()
            dx = point.x() - self._drag_last_pos.x()
            dy = point.y() - self._drag_last_pos.y()
            if dx != 0 or dy != 0:
                self.pan_delta.emit(self.source_name, float(dx), float(dy))
                self._drag_last_pos = point
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: ANN401
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            self._drag_last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: ANN401
        self._drag_active = False
        self._drag_last_pos = None
        super().leaveEvent(event)

    def wheelEvent(self, event: Any) -> None:  # noqa: ANN401
        self.zoom_delta.emit(self.source_name, int(event.angleDelta().y()))
        event.accept()


class VideoTile(QWidget):
    clicked = pyqtSignal(str)
    right_clicked = pyqtSignal(str)
    zoom_delta = pyqtSignal(str, int)
    pan_delta = pyqtSignal(str, float, float)

    def __init__(self, source_name: str) -> None:
        super().__init__()
        self.source_name = source_name
        self._is_focused = False
        self._is_alert = False
        self._header_visible = True
        self._border_width = 5
        self._last_meta_text = "idle"
        self.setObjectName("videoTile")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.title_label = QLabel(source_name)
        self.title_label.setStyleSheet("color: #f1f4fa; font-weight: 700; padding-left: 6px;")

        self.meta_label = QLabel("idle")
        self.meta_label.setStyleSheet("color: #d0d7e4; padding-right: 6px;")

        self.canvas = VideoCanvas(source_name)
        self.canvas.clicked.connect(self.clicked.emit)
        self.canvas.right_clicked.connect(self.right_clicked.emit)
        self.canvas.zoom_delta.connect(self.zoom_delta.emit)
        self.canvas.pan_delta.connect(self.pan_delta.emit)

        self.header_widget = QWidget(self)
        self.header_widget.setObjectName("videoHeader")
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.addWidget(self.title_label)
        top_row.addStretch(1)
        top_row.addWidget(self.meta_label)
        self.header_widget.setLayout(top_row)
        self.header_widget.setFixedHeight(30)
        self.header_widget.setStyleSheet(
            "QWidget#videoHeader {"
            "background: #11151c;"
            "border: 0px;"
            "}"
        )

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)
        self.root_layout.addWidget(self.header_widget)
        self.root_layout.addWidget(self.canvas, stretch=1)

        self._refresh_style()

    def _sync_header_visibility(self) -> None:
        self.header_widget.setVisible(self._header_visible and not self._is_focused)

    def set_focus_state(self, focused: bool) -> None:
        focused = bool(focused)
        if self._is_focused == focused:
            return
        self._is_focused = focused
        self.canvas.set_expand_mode(self._is_focused)
        self._sync_header_visibility()
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)
        self._refresh_style()

    def set_alert_state(self, alert: bool) -> None:
        alert = bool(alert)
        if self._is_alert == alert:
            return
        self._is_alert = alert
        self._refresh_style()

    def set_header_visibility(self, visible: bool) -> None:
        visible = bool(visible)
        if self._header_visible == visible:
            return
        self._header_visible = visible
        self._sync_header_visibility()

    def _refresh_style(self) -> None:
        border_width = self._border_width
        if self._is_alert:
            border_color = "#e53935"
        elif self._is_focused:
            border_color = "#4ea7ff"
        else:
            border_color = "#2e3643"
        self.setStyleSheet(
            "QWidget#videoTile {"
            "background: #0b0d11;"
            f"border: {border_width}px solid {border_color};"
            "border-radius: 6px;"
            "}"
        )

    def update_view(
        self,
        frame: np.ndarray | None,
        *,
        meta_text: str,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> None:
        if meta_text != self._last_meta_text:
            self._last_meta_text = meta_text
            self.meta_label.setText(meta_text)
        self.canvas.set_frame(frame, zoom=zoom, pan_x=pan_x, pan_y=pan_y)


class MaskEditorWidget(QWidget):
    def __init__(self, frame: np.ndarray, poly_norm: list[list[tuple[float, float]]] | list[tuple[float, float]] | None) -> None:
        super().__init__()
        self._polys_norm: list[list[tuple[float, float]]] = []
        if poly_norm:
            if poly_norm and isinstance(poly_norm[0], (list, tuple)) and len(poly_norm) > 0:
                # Either list of points or list of polygons; normalize.
                first = poly_norm[0]  # type: ignore[index]
                if first and isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
                    self._polys_norm = [list(poly_norm)]  # type: ignore[list-item]
                else:
                    self._polys_norm = [list(poly) for poly in poly_norm]  # type: ignore[arg-type]
        self._current_poly: list[tuple[float, float]] = []
        self._dragging = False
        self._drag_start: tuple[float, float] | None = None
        self._last_point: tuple[float, float] | None = None
        self._pixmap: QPixmap | None = None
        self._img_w = 0
        self._img_h = 0
        self.setMinimumSize(480, 270)
        self.set_frame(frame)

    def set_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image)
        self._img_w = width
        self._img_h = height
        self.update()

    def clear_rect(self) -> None:
        self._polys_norm = []
        self._current_poly = []
        self.update()

    def get_rect(self) -> list[list[tuple[float, float]]]:
        return [list(poly) for poly in self._polys_norm]

    def _display_rect(self) -> tuple[float, float, float, float, float]:
        if self._pixmap is None or self._pixmap.isNull():
            return 0.0, 0.0, 0.0, 0.0, 1.0
        w = max(1, self.width())
        h = max(1, self.height())
        scale = min(w / float(self._img_w), h / float(self._img_h))
        disp_w = self._img_w * scale
        disp_h = self._img_h * scale
        offset_x = (w - disp_w) / 2.0
        offset_y = (h - disp_h) / 2.0
        return offset_x, offset_y, disp_w, disp_h, scale

    def _widget_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        offset_x, offset_y, disp_w, disp_h, scale = self._display_rect()
        if disp_w <= 0 or disp_h <= 0:
            return None
        clamped_x = _clamp(x, offset_x, offset_x + disp_w)
        clamped_y = _clamp(y, offset_y, offset_y + disp_h)
        img_x = (clamped_x - offset_x) / scale
        img_y = (clamped_y - offset_y) / scale
        return _clamp(img_x, 0.0, float(self._img_w)), _clamp(img_y, 0.0, float(self._img_h))

    def paintEvent(self, event: Any) -> None:  # noqa: ANN401
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        offset_x, offset_y, disp_w, disp_h, _ = self._display_rect()
        target = QRectF(offset_x, offset_y, disp_w, disp_h)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if self._polys_norm:
            painter.setPen(QPen(QColor(255, 80, 80), 2))
            painter.setBrush(QColor(255, 80, 80, 40))
            for poly_norm in self._polys_norm:
                if len(poly_norm) < 3:
                    continue
                points = [QPointF(offset_x + x * disp_w, offset_y + y * disp_h) for x, y in poly_norm]
                painter.drawPolygon(QPolygonF(points))

        if self._current_poly and len(self._current_poly) >= 2:
            painter.setPen(QPen(QColor(255, 120, 120), 2))
            painter.setBrush(QColor(255, 120, 120, 20))
            points = [QPointF(offset_x + x * disp_w, offset_y + y * disp_h) for x, y in self._current_poly]
            painter.drawPolyline(QPolygonF(points))

    def mousePressEvent(self, event: Any) -> None:  # noqa: ANN401
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position().toPoint()
        img_pos = self._widget_to_image(float(point.x()), float(point.y()))
        if img_pos is None:
            return
        self._dragging = True
        self._drag_start = img_pos
        self._last_point = img_pos
        x0, y0 = img_pos
        self._current_poly = [
            (_clamp(x0 / self._img_w, 0.0, 1.0), _clamp(y0 / self._img_h, 0.0, 1.0))
        ]

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: ANN401
        if not self._dragging or self._drag_start is None:
            return
        point = event.position().toPoint()
        img_pos = self._widget_to_image(float(point.x()), float(point.y()))
        if img_pos is None:
            return
        if self._last_point is not None:
            dx = img_pos[0] - self._last_point[0]
            dy = img_pos[1] - self._last_point[1]
            if (dx * dx + dy * dy) < 9.0:
                return
        self._last_point = img_pos
        x1, y1 = img_pos
        self._current_poly.append(
            (_clamp(x1 / self._img_w, 0.0, 1.0), _clamp(y1 / self._img_h, 0.0, 1.0))
        )
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: ANN401
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = False
        self._drag_start = None
        self._last_point = None
        if len(self._current_poly) >= 3:
            self._polys_norm.append(list(self._current_poly))
        self._current_poly = []
        self.update()


class ModelDownloadProgressDialog(QDialog):
    def __init__(self, title: str, model_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(520)

        self.status_label = QLabel(f"Przygotowanie pobierania modelu: {model_name}")
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.percent_label = QLabel("Trwa pobieranie...")
        self.percent_label.setStyleSheet("color: #bfc9da;")

        self.ok_button = QPushButton("OK")
        self.ok_button.setEnabled(False)
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.percent_label)
        layout.addWidget(self.ok_button, alignment=Qt.AlignmentFlag.AlignRight)

    def _pump_ui(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def update_from_progress(self, status: str, downloaded: int | None, total: int | None) -> None:
        self.status_label.setText(str(status or "Pobieranie modelu..."))

        if total is not None and total > 0 and downloaded is not None:
            percent = int(max(0, min(100, round((float(downloaded) * 100.0) / float(total)))))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)
            self.percent_label.setText(f"{percent}%")
        else:
            self.progress_bar.setRange(0, 0)
            self.percent_label.setText("Trwa pobieranie...")

        self._pump_ui()

    def mark_complete(self, message: str = "Pobieranie zakonczone.") -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText(message)
        self.percent_label.setText("100%")
        self.ok_button.setEnabled(True)
        self._pump_ui()


class UniformPreviewWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._top_color = QColor(UNIFORM_TOP_DEFAULT)
        self._bottom_color = QColor(UNIFORM_BOTTOM_DEFAULT)
        self.setMinimumSize(150, 220)

    def set_colors(self, top_hex: str, bottom_hex: str) -> None:
        self._top_color = QColor(_normalize_hex_color(top_hex, UNIFORM_TOP_DEFAULT))
        self._bottom_color = QColor(_normalize_hex_color(bottom_hex, UNIFORM_BOTTOM_DEFAULT))
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: ANN401
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.fillRect(rect, QColor("#121822"))
        painter.setPen(QPen(QColor("#344153"), 2))
        painter.drawRoundedRect(rect, 14, 14)

        center_x = rect.center().x()
        top_y = rect.top() + 18
        head_radius = max(14, rect.width() // 10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f1d3bd"))
        painter.drawEllipse(QPointF(center_x, top_y + head_radius), head_radius, head_radius)

        torso_top = top_y + head_radius * 2 + 10
        torso_width = max(34, rect.width() // 3)
        torso_height = max(56, rect.height() // 3)
        torso_rect = QRectF(center_x - torso_width / 2, torso_top, torso_width, torso_height)
        painter.setBrush(self._top_color)
        painter.drawRoundedRect(torso_rect, 12, 12)

        arm_width = max(10, torso_width // 4)
        arm_height = max(48, torso_height - 10)
        painter.drawRoundedRect(QRectF(torso_rect.left() - arm_width + 4, torso_rect.top() + 8, arm_width, arm_height), 8, 8)
        painter.drawRoundedRect(QRectF(torso_rect.right() - 4, torso_rect.top() + 8, arm_width, arm_height), 8, 8)

        waist_y = torso_rect.bottom() - 4
        leg_gap = max(8, torso_width // 7)
        leg_width = max(14, torso_width // 3)
        leg_height = max(64, rect.bottom() - waist_y - 26)
        painter.setBrush(self._bottom_color)
        painter.drawRoundedRect(
            QRectF(center_x - leg_gap / 2 - leg_width, waist_y, leg_width, leg_height),
            9,
            9,
        )
        painter.drawRoundedRect(
            QRectF(center_x + leg_gap / 2, waist_y, leg_width, leg_height),
            9,
            9,
        )

        shoe_width = max(18, leg_width + 6)
        shoe_height = 10
        painter.setBrush(QColor("#d8dde8"))
        painter.drawRoundedRect(
            QRectF(center_x - leg_gap / 2 - shoe_width, waist_y + leg_height - 4, shoe_width, shoe_height),
            5,
            5,
        )
        painter.drawRoundedRect(
            QRectF(center_x + leg_gap / 2 - 2, waist_y + leg_height - 4, shoe_width, shoe_height),
            5,
            5,
        )


class MaskEditorDialog(QDialog):
    def __init__(
        self,
        frame: np.ndarray,
        poly_norm: list[list[tuple[float, float]]] | list[tuple[float, float]] | None,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ustaw strefe niewidoczna dla AI")
        self.editor = MaskEditorWidget(frame, poly_norm)
        self.setMinimumSize(900, 520)
        self.resize(1000, 640)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowFullScreen)

        info = QLabel(
            "Przytrzymaj LPM i rysuj ksztalt. Poza obrazem punkt przyciagnie sie do krawedzi. "
            "Mozesz narysowac kilka ksztaltow."
        )
        info.setStyleSheet("color: #bfc9da;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        back_btn = QPushButton("Cofnij")
        buttons.addButton(back_btn, QDialogButtonBox.ButtonRole.ActionRole)
        clear_btn = QPushButton("Wyczysc")
        buttons.addButton(clear_btn, QDialogButtonBox.ButtonRole.ResetRole)

        back_btn.clicked.connect(self.reject)
        clear_btn.clicked.connect(self.editor.clear_rect)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.editor, stretch=1)
        layout.addWidget(buttons)

    def get_rect(self) -> list[list[tuple[float, float]]]:
        return self.editor.get_rect()


class FullscreenVideoWindow(QWidget):
    request_close = pyqtSignal()
    zoom_delta = pyqtSignal(int)
    pan_delta = pyqtSignal(float, float)

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background: #000;")

        self.canvas = VideoCanvas("fullscreen")
        self.canvas.setMinimumSize(1, 1)
        self.canvas.set_expand_mode(True)
        self.canvas.clicked.connect(lambda _name: None)
        self.canvas.right_clicked.connect(lambda _name: self.request_close.emit())
        self.canvas.zoom_delta.connect(lambda _name, delta: self.zoom_delta.emit(delta))
        self.canvas.pan_delta.connect(lambda _name, dx, dy: self.pan_delta.emit(dx, dy))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)

        self.source_label = QLabel("", self)
        self.source_label.setStyleSheet(
            "QLabel {"
            "background: rgba(20, 24, 31, 210);"
            "color: #ecf2ff;"
            "border: 1px solid #4f6078;"
            "border-radius: 8px;"
            "padding: 4px 10px;"
            "font-weight: 600;"
            "}"
        )

        self.close_btn = QPushButton("✕", self)
        self.close_btn.setFixedSize(42, 34)
        self.close_btn.clicked.connect(self.request_close.emit)
        self.close_btn.setStyleSheet(
            "QPushButton {"
            "background: rgba(160, 38, 38, 220);"
            "color: #fff;"
            "border: 1px solid #7a2a2a;"
            "border-radius: 8px;"
            "font-size: 18px;"
            "font-weight: 700;"
            "}"
            "QPushButton:hover { background: rgba(190, 46, 46, 240); }"
        )

    def set_source_name(self, source_name: str) -> None:
        self.source_label.setText(source_name)
        self.source_label.adjustSize()
        self._position_overlay()

    def set_frame(self, frame: np.ndarray | None, *, zoom: float, pan_x: float, pan_y: float) -> None:
        self.canvas.set_frame(frame, zoom=zoom, pan_x=pan_x, pan_y=pan_y)

    def _position_overlay(self) -> None:
        margin = 16
        self.close_btn.move(self.width() - self.close_btn.width() - margin, margin)
        self.source_label.move(margin, margin)
        self.close_btn.raise_()
        self.source_label.raise_()

    def resizeEvent(self, event: Any) -> None:  # noqa: ANN401
        super().resizeEvent(event)
        self._position_overlay()

    def keyPressEvent(self, event: Any) -> None:  # noqa: ANN401
        if event.key() in {Qt.Key.Key_Escape, Qt.Key.Key_F11}:
            self.request_close.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: ANN401
        if event.button() == Qt.MouseButton.LeftButton:
            self.request_close.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class InferenceWindow(QMainWindow):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config_path = Path(config_path).resolve()
        self.config = load_yaml(self.config_path)

        self.app_root_dir = resolve_path("logs/app")
        self.app_settings_dir = self.app_root_dir / "settings"
        self.app_config_path = self.app_settings_dir / "config.yaml"
        self._load_app_config_overlay()
        self.sources_settings_path = self.app_settings_dir / "sources.yaml"
        self.sources = self._load_sources_config()
        self.source_by_name: dict[str, dict[str, Any]] = {}
        self._rebuild_source_lookup()

        self.model_cfg = dict(self.config.get("model", {}) or {})
        self.inference_cfg = dict(self.config.get("inference", {}) or {})
        self.security_cfg = dict(self.config.get("security", {}) or {})
        self.uniform_cfg = dict(self.config.get("uniform", {}) or {})
        self.runtime_cfg = dict(self.config.get("runtime", {}) or {})
        self.tracker_cfg = dict(self.config.get("tracker", {}) or {})
        self.events_cfg = dict(self.config.get("events", {}) or {})
        self.debug_cfg = dict(self.config.get("debug", {}) or {})
        self.runtime_cfg.setdefault("annotation_reference_height", 720)
        self.runtime_cfg.setdefault("annotation_label_font_scale", 0.48)
        self.runtime_cfg.setdefault("annotation_status_font_scale", 0.46)
        self.runtime_cfg.setdefault("annotation_min_resolution_scale", 0.45)
        self.runtime_cfg.setdefault("annotation_max_resolution_scale", 4.0)
        self.runtime_cfg.setdefault("camera_backend", "msmf")
        self.runtime_cfg.setdefault("uniform_recheck_interval_sec", 5.0)
        self.runtime_cfg.setdefault("uniform_worker_hold_sec", 5.0)
        self.runtime_cfg.setdefault("uniform_memory_ttl_sec", 8.0)
        self.runtime_cfg.setdefault("detection_box_hold_sec", 0.5)
        self.debug_cfg.setdefault("enabled", False)
        self.debug_cfg.setdefault("console", True)
        self.debug_cfg.setdefault("profiling", True)
        self.debug_cfg.setdefault("profile_interval_sec", 5.0)
        self.console_logs_enabled = bool(self.runtime_cfg.get("console_logs", False))
        self.suppress_opencv_warnings = bool(self.runtime_cfg.get("suppress_opencv_warnings", True))
        self.auto_scan_cameras_on_startup = bool(self.runtime_cfg.get("auto_scan_cameras_on_startup", False))
        self.auto_start_live = bool(self.runtime_cfg.get("auto_start_live", True))
        self.debug_enabled = bool(self.debug_cfg.get("enabled", False))
        self.debug_console_enabled = bool(self.debug_cfg.get("console", True))
        self.debug_profiling_enabled = bool(self.debug_cfg.get("profiling", True))
        self.debug_profile_interval_sec = max(0.5, float(self.debug_cfg.get("profile_interval_sec", 5.0)))
        _configure_opencv_logging(silent=self.suppress_opencv_warnings)
        self.tracker_enabled = bool(self.tracker_cfg.get("enabled", True))
        self.tracker_backend_name = str(self.tracker_cfg.get("backend", "botsort")).strip().lower() or "botsort"
        self.tracker_backend_cls: type[Any] | None = None
        self._tracker_disabled_reason: str | None = None
        if self.tracker_enabled:
            self.tracker_backend_cls, tracker_error = _resolve_tracker_backend(self.tracker_backend_name)
            if self.tracker_backend_cls is None:
                self.tracker_enabled = False
                backend_label = self.tracker_backend_name or "tracker"
                self._tracker_disabled_reason = f"{backend_label} disabled: {tracker_error or 'backend unavailable'}"

        self.model: YOLO | None = None
        self.model_reference = ""
        self.current_model_path: Path | None = None
        self.day_seg_model: YOLO | None = None
        self.day_seg_model_reference = ""
        self.current_day_seg_model_path: Path | None = None
        self.predict_kwargs: dict[str, Any] = {}
        self.day_seg_predict_kwargs: dict[str, Any] = {}
        self.compile_requested = False
        self.compile_enabled = False
        self.compile_fallback_applied = False
        self.manual_compile_active = False
        self._last_saved_config_signature = ""

        self.runtimes: dict[str, SourceRuntime] = {}
        self.trackers: dict[str, Any] = {}
        self.tiles: dict[str, VideoTile] = {}
        self.zoom_levels: dict[str, float] = {}
        self.pan_offsets: dict[str, tuple[float, float]] = {}
        self.focused_source: str | None = None
        self.fullscreen_window: FullscreenVideoWindow | None = None

        self.live_running = False
        self.frame_interval_ms = int(self.runtime_cfg.get("frame_interval_ms", 16))
        self.view_target_fps = float(_clamp(float(self.runtime_cfg.get("view_target_fps", 60.0)), 1.0, 60.0))
        self.model_target_fps = max(1, int(round(float(self.runtime_cfg.get("model_target_fps", 6.0)))))
        self.max_infer_per_tick = max(1, int(self.runtime_cfg.get("max_infer_per_tick", 2)))
        self.loop_videos = bool(self.runtime_cfg.get("loop_videos", True))
        self.live_tile_spacing = max(0, int(self.runtime_cfg.get("live_tile_spacing", 4)))
        self.live_tile_header_visible = bool(self.runtime_cfg.get("show_live_tile_headers", True))
        self.camera_backend = str(self.runtime_cfg.get("camera_backend", "msmf")).strip().lower() or "msmf"
        self._detection_box_hold_sec = float(
            _clamp(float(self.runtime_cfg.get("detection_box_hold_sec", 0.5)), 0.0, 5.0)
        )
        self.annotation_reference_height = max(120.0, float(self.runtime_cfg.get("annotation_reference_height", 720.0)))
        self.annotation_label_font_scale = float(
            _clamp(float(self.runtime_cfg.get("annotation_label_font_scale", 0.48)), 0.20, 1.20)
        )
        self.annotation_status_font_scale = float(
            _clamp(float(self.runtime_cfg.get("annotation_status_font_scale", 0.46)), 0.20, 1.20)
        )
        self.annotation_min_resolution_scale = float(
            _clamp(float(self.runtime_cfg.get("annotation_min_resolution_scale", 0.45)), 0.10, 1.50)
        )
        self.annotation_max_resolution_scale = float(
            _clamp(float(self.runtime_cfg.get("annotation_max_resolution_scale", 4.0)), 1.0, 8.0)
        )
        # Always start with the navigation tabs visible, even if the previous session hid them.
        self.navigation_tabs_visible = True

        self._model_lock = threading.Lock()
        self._capture_lock = threading.RLock()
        self._infer_lock = threading.RLock()
        self._infer_stop_event = threading.Event()
        self._infer_has_work_event = threading.Event()
        self._infer_thread: threading.Thread | None = None
        self._infer_pending_frames: dict[str, np.ndarray] = {}
        self._infer_results: dict[str, AsyncInferenceResult] = {}
        self._infer_last_submit_ts: dict[str, float] = {}
        self._infer_worker_error: str | None = None
        self._infer_notices: list[str] = []
        self._infer_worker_rr_cursor = 0
        self._infer_batch_coalesce_ms = float(
            _clamp(float(self.runtime_cfg.get("infer_batch_coalesce_ms", 4.0)), 0.0, 15.0)
        )
        self._tracker_warn_last_ts: dict[str, float] = {}
        self._debug_profile_lock = threading.Lock()
        self._debug_profile_stats: dict[str, dict[str, float]] = {}
        self._debug_uniform_samples_ms: list[float] = []
        self._debug_batch_size_total = 0.0
        self._debug_batch_size_count = 0
        self._debug_profile_last_flush_ts = time.perf_counter()
        self._ignore_mask_cache: dict[str, tuple[tuple[int, int], str, np.ndarray]] = {}
        self._uniform_track_memory: dict[str, dict[int, dict[str, Any]]] = {}
        self._uniform_memory_decay = float(_clamp(float(self.runtime_cfg.get("uniform_memory_decay", 0.88)), 0.5, 0.99))
        self._uniform_memory_alpha = float(_clamp(float(self.runtime_cfg.get("uniform_memory_alpha", 0.30)), 0.05, 0.95))
        self._uniform_memory_min_worker_score = float(
            _clamp(float(self.runtime_cfg.get("uniform_memory_min_worker_score", 0.55)), 0.2, 0.95)
        )
        self._uniform_memory_max_bad_streak = max(1, int(self.runtime_cfg.get("uniform_memory_max_bad_streak", 4)))
        self._uniform_worker_hold_sec = max(0.0, float(self.runtime_cfg.get("uniform_worker_hold_sec", 5.0)))
        self._uniform_memory_ttl_sec = max(
            self._uniform_worker_hold_sec + 1.0,
            float(self.runtime_cfg.get("uniform_memory_ttl_sec", 8.0)),
        )
        self._uniform_recheck_interval_sec = max(0.0, float(self.runtime_cfg.get("uniform_recheck_interval_sec", 5.0)))
        self._uniform_recheck_iou = float(_clamp(float(self.runtime_cfg.get("uniform_recheck_iou", 0.85)), 0.2, 0.99))
        self._uniform_max_fresh_per_cycle = max(1, int(self.runtime_cfg.get("uniform_max_fresh_per_cycle", 2)))
        self._event_writer_cond = threading.Condition()
        self._event_writer_queue: deque[tuple[str, int]] = deque()
        self._event_writer_latest_frames: dict[tuple[str, int], np.ndarray] = {}
        self._event_writer_prebuffer_frames: dict[tuple[str, int], deque[np.ndarray]] = {}
        self._event_writer_queued_keys: set[tuple[str, int]] = set()
        self._event_writer_inflight_keys: set[tuple[str, int]] = set()
        self._event_writer_stop = False
        self._event_writer_thread: threading.Thread | None = None
        self._load_shed_level = 0.0
        self._load_shed_last_pressure_ts = 0.0
        self._load_shed_last_log_ts = 0.0
        self._load_shed_decay_block_until_ts = 0.0
        self._load_shed_initial_level = float(_clamp(float(self.runtime_cfg.get("load_shed_initial_level", 0.70)), 0.10, 1.0))
        self._load_shed_min_view_fps = max(5.0, float(self.runtime_cfg.get("load_shed_min_view_fps", 10.0)))
        self._load_shed_min_model_fps = max(1.0, float(self.runtime_cfg.get("load_shed_min_model_fps", 4.0)))
        self._load_shed_hold_seconds = max(1.0, float(self.runtime_cfg.get("load_shed_hold_seconds", 6.0)))
        self._load_shed_sticky_seconds = max(1.0, float(self.runtime_cfg.get("load_shed_sticky_seconds", 18.0)))
        self._load_shed_decay_per_sec = max(0.05, float(self.runtime_cfg.get("load_shed_decay_per_sec", 0.20)))
        self._live_timer_interval_ms = int(max(1, self.frame_interval_ms))
        self._live_timer_last_adjust_ts = 0.0
        self.events_enabled = bool(self.events_cfg.get("enabled", True))
        self.events_min_visible_seconds = max(0.1, float(self.events_cfg.get("min_visible_seconds", 3.0)))
        self.events_cooldown_seconds = max(0.0, float(self.events_cfg.get("cooldown_seconds", 10.0)))
        self.events_linger_seconds = max(0.0, float(self.events_cfg.get("linger_seconds", 1.5)))
        self.events_min_person_count = max(1, int(self.events_cfg.get("min_person_count", 1)))
        self.events_clip_fps = max(1.0, float(self.events_cfg.get("clip_fps", 30.0)))
        self.events_prebuffer_seconds = max(0.0, float(self.events_cfg.get("prebuffer_seconds", 2.0)))
        self.events_save_annotated = bool(self.events_cfg.get("save_annotated_frame", True))
        self.events_once_per_streak = bool(self.events_cfg.get("once_per_streak", True))
        self.events_max_saved = max(0, int(self.events_cfg.get("max_saved_events", 300)))
        self.events_output_dir_raw = str(self.events_cfg.get("output_dir", "logs/app/events")).strip()
        self.events_output_dir = resolve_path(self.events_output_dir_raw or "logs/app/events")
        self.events_index_path = self.events_output_dir / "events_index.json"
        self.event_entries: list[dict[str, Any]] = []
        self._event_table_updating = False

        self._table_updating = False
        self._log_entries: list[str] = []
        self._pending_log_lines: list[str] = []
        self._log_flush_interval_ms = max(20, int(self.runtime_cfg.get("log_flush_interval_ms", 120)))
        self._log_flush_batch_size = max(1, int(self.runtime_cfg.get("log_flush_batch_size", 200)))
        self._settings_autosave_delay_ms = max(100, int(self.runtime_cfg.get("settings_autosave_delay_ms", 400)))
        self.settings_dirty = False
        self._settings_change_tracking_ready = False
        self._last_main_tab_index = 0
        self._suppress_main_tab_change_handler = False
        self._settings_tab_index = -1
        if self._tracker_disabled_reason:
            self._log(self._tracker_disabled_reason)
        elif self.tracker_enabled:
            self._log(f"Tracker enabled: {self.tracker_backend_name}.")

        self.model_catalog: list[dict[str, Any]] = []
        self.model_table_row_map: list[int | None] = []
        self._online_model_suggestions_cache: set[str] | None = None
        self._online_model_suggestions_error_logged = False
        self._source_table_min_visible_rows = 8
        self._source_table_max_visible_rows = 15

        self.recording_capture: cv2.VideoCapture | None = None
        self.recording_playing = False
        self.recording_frame_count = 0
        self.recording_fps = 25.0
        self.recording_duration_sec = 0.0
        self.recording_current_frame = 0
        self.recording_slider_internal = False
        self.recording_slider_user_drag = False
        self.recording_zoom = 1.0
        self.recording_pan_x = 0.0
        self.recording_pan_y = 0.0
        self._suppress_setting_autosave = True
        self._load_event_entries()

        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._tick_live)

        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self._tick_recording)

        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.timeout.connect(self._flush_logs_widget)

        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.timeout.connect(self._commit_pending_settings)

        self._start_event_writer_thread()

        self._load_model()
        self._build_ui()
        self._sync_runtimes_with_sources()
        self._rebuild_source_table()
        self._rebuild_live_layout()
        self._refresh_model_catalog()
        self._settings_change_tracking_ready = True
        self._clear_settings_dirty()

        self._log("Application started.")
        if self.auto_start_live:
            QTimer.singleShot(200, self._auto_start_live_if_possible)

    # ---------- logging ----------
    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self._log_entries.append(line)
        if self.console_logs_enabled:
            print(line)
        if hasattr(self, "logs_text") and self.logs_text is not None:
            self._pending_log_lines.append(line)
            if not self._log_flush_timer.isActive():
                self._log_flush_timer.start(self._log_flush_interval_ms)

    def _debug_console(self, message: str) -> None:
        if not self.debug_enabled or not self.debug_console_enabled:
            return
        timestamp = time.strftime("%H:%M:%S")
        print(f"[debug {timestamp}] {message}")

    def _profile_add(self, stage: str, elapsed_sec: float, *, units: int = 1) -> None:
        if not self.debug_enabled or not self.debug_profiling_enabled:
            return
        safe_units = max(1, int(units))
        safe_elapsed = max(0.0, float(elapsed_sec))
        with self._debug_profile_lock:
            entry = self._debug_profile_stats.setdefault(
                stage,
                {"total_sec": 0.0, "count": 0.0, "max_ms": 0.0},
            )
            entry["total_sec"] += safe_elapsed
            entry["count"] += float(safe_units)
            entry["max_ms"] = max(float(entry.get("max_ms", 0.0)), (safe_elapsed * 1000.0) / safe_units)
            if stage == "uniform":
                self._debug_uniform_samples_ms.append((safe_elapsed * 1000.0) / safe_units)

    def _profile_batch_size(self, batch_size: int) -> None:
        if not self.debug_enabled or not self.debug_profiling_enabled:
            return
        with self._debug_profile_lock:
            self._debug_batch_size_total += float(max(0, int(batch_size)))
            self._debug_batch_size_count += 1

    def _maybe_flush_debug_profile(self) -> None:
        if not self.debug_enabled or not self.debug_profiling_enabled:
            return
        now_ts = time.perf_counter()
        if (now_ts - self._debug_profile_last_flush_ts) < self.debug_profile_interval_sec:
            return

        with self._debug_profile_lock:
            snapshot = dict(self._debug_profile_stats)
            self._debug_profile_stats.clear()
            uniform_samples = sorted(float(value) for value in self._debug_uniform_samples_ms)
            self._debug_uniform_samples_ms.clear()
            batch_size_total = float(self._debug_batch_size_total)
            batch_size_count = int(self._debug_batch_size_count)
            self._debug_batch_size_total = 0.0
            self._debug_batch_size_count = 0
            self._debug_profile_last_flush_ts = now_ts

        if not snapshot:
            return

        parts: list[str] = []
        for stage in sorted(snapshot.keys()):
            entry = snapshot[stage]
            count = max(1.0, float(entry.get("count", 0.0)))
            total_ms = float(entry.get("total_sec", 0.0)) * 1000.0
            avg_ms = total_ms / count
            max_ms = float(entry.get("max_ms", 0.0))
            parts.append(f"{stage}: avg={avg_ms:.2f}ms max={max_ms:.2f}ms n={int(round(count))}")

        if batch_size_count > 0:
            avg_batch_size = batch_size_total / float(batch_size_count)
            parts.append(f"batch: avg_size={avg_batch_size:.2f} n={batch_size_count}")

        if uniform_samples:
            p95_index = min(len(uniform_samples) - 1, max(0, int(math.ceil(len(uniform_samples) * 0.95)) - 1))
            parts.append(f"uniform_p95={uniform_samples[p95_index]:.2f}ms")

        self._debug_console("profile | " + " | ".join(parts))

    def _flush_logs_widget(self) -> None:
        if not hasattr(self, "logs_text") or self.logs_text is None:
            return
        if not self._pending_log_lines:
            return

        chunk = self._pending_log_lines[: self._log_flush_batch_size]
        del self._pending_log_lines[: len(chunk)]
        cursor = self.logs_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("".join(f"{line}\n" for line in chunk))
        self.logs_text.setTextCursor(cursor)
        self.logs_text.ensureCursorVisible()

        if self._pending_log_lines:
            self._log_flush_timer.start(self._log_flush_interval_ms)

    def _auto_start_live_if_possible(self) -> None:
        if self.live_running:
            return
        if not self._get_enabled_sources():
            self._log("Auto-start skipped: no enabled sources.")
            return
        self.start_live()

    # ---------- model ----------
    def _resolve_canonical_trained_model_path(self, model_cfg: dict[str, Any]) -> Path | None:
        model_name = str(model_cfg.get("name", "")).strip()
        if not model_name:
            return None
        if not model_name.lower().endswith(".pt"):
            model_name = f"{model_name}.pt"
        candidate = resolve_path(Path("models/weights") / model_name)
        if candidate.exists():
            return candidate.resolve()
        return None

    def _resolve_trained_weights_path(self, model_cfg: dict[str, Any]) -> Path | None:
        trained_cfg = model_cfg.get("trained_weights", {}) or {}
        if not bool(trained_cfg.get("enabled", False)):
            return None

        weights_dir = resolve_path(trained_cfg.get("dir", "models/weights/latest"))
        preferred = str(trained_cfg.get("preferred", "best")).strip().lower()
        order = ("last.pt", "best.pt") if preferred == "last" else ("best.pt", "last.pt")

        for filename in order:
            candidate = (weights_dir / filename).resolve()
            if candidate.exists():
                return candidate
        return None

    def _resolve_model_reference_for_cfg(self, model_cfg: dict[str, Any]) -> tuple[YOLO, str, Path | None]:
        selected_model_path = str(model_cfg.get("selected_model_path", "")).strip()
        if selected_model_path:
            selected_path = resolve_path(selected_model_path)
            if selected_path.exists():
                return YOLO(str(selected_path)), str(selected_path), selected_path

        trained_cfg = model_cfg.get("trained_weights", {}) or {}
        prefer_canonical = bool(trained_cfg.get("prefer_canonical", True))
        if prefer_canonical:
            canonical_trained_path = self._resolve_canonical_trained_model_path(model_cfg)
            if canonical_trained_path is not None:
                return YOLO(str(canonical_trained_path)), str(canonical_trained_path), canonical_trained_path

        trained_path = self._resolve_trained_weights_path(model_cfg)
        fallback_to_base = bool(trained_cfg.get("fallback_to_base", True))

        if trained_path is not None:
            return YOLO(str(trained_path)), str(trained_path), trained_path

        if fallback_to_base:
            model, reference = load_yolo_model(model_cfg)
            reference_path = Path(reference).resolve() if Path(reference).exists() else None
            return model, reference, reference_path

        raise FileNotFoundError(
            "No model available. Set model.selected_model_path or keep trained_weights fallback enabled."
        )

    def _resolve_model_reference(self) -> tuple[YOLO, str, Path | None]:
        return self._resolve_model_reference_for_cfg(self.model_cfg)

    def _rebuild_predict_kwargs(self) -> None:
        self.predict_kwargs = {
            "conf": float(self.inference_cfg.get("conf", 0.35)),
            "iou": float(self.inference_cfg.get("iou", 0.45)),
            "imgsz": int(self.inference_cfg.get("imgsz", 960)),
            "max_det": int(self.inference_cfg.get("max_det", 100)),
            "verbose": False,
        }

        classes = self.inference_cfg.get("classes")
        if isinstance(classes, list) and classes:
            self.predict_kwargs["classes"] = [int(item) for item in classes]

        device = self.inference_cfg.get("device")
        if device not in (None, "", "auto"):
            self.predict_kwargs["device"] = device

        half = self.inference_cfg.get("half")
        if half is not None:
            self.predict_kwargs["half"] = bool(half)

        if self.compile_enabled and not self.manual_compile_active:
            self.predict_kwargs["compile"] = True
        else:
            self.predict_kwargs.pop("compile", None)

        self.day_seg_predict_kwargs = dict(self.predict_kwargs)
        self.day_seg_predict_kwargs.pop("compile", None)

    def _uniform_detection_enabled(self) -> bool:
        return bool(self.uniform_cfg.get("enabled", True))

    def _day_segmentation_model_cfg(self) -> dict[str, Any]:
        return _build_day_segmentation_cfg(self.model_cfg)

    def _load_optional_day_segmentation_model(self) -> None:
        self.day_seg_model = None
        self.day_seg_model_reference = ""
        self.current_day_seg_model_path = None

        if not self._uniform_detection_enabled():
            return

        seg_cfg = self._day_segmentation_model_cfg()
        if not bool(seg_cfg.get("enabled", True)):
            return

        try:
            model, reference, reference_path = self._resolve_model_reference_for_cfg(seg_cfg)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[warn] Nie udalo sie zaladowac modelu segmentacji dziennej: {exc}")
            return

        self.day_seg_model = model
        self.day_seg_model_reference = reference
        self.current_day_seg_model_path = reference_path
        self.model_cfg["day_segmentation"] = dict(seg_cfg)
        if reference_path is not None and reference_path.exists():
            self.model_cfg["day_segmentation"]["selected_model_path"] = _to_relative_or_abs(reference_path)

    def _is_compile_runtime_error(self, exc: Exception) -> bool:
        message = str(exc)
        normalized = message.lower()
        return (
            "autobackend does not support len()" in normalized
            or "does not support len()" in normalized
            or "backendcompilerfailed" in normalized
            or "torch.utils._sympy" in normalized
            or "pow_by_natural" in normalized
            or "failed while executing" in normalized
            or "torch._inductor" in normalized
            or "inductor" in normalized
            or "triton" in normalized
        )

    def _compile_status_text(self) -> str:
        if self.manual_compile_active:
            return "enabled (manual backend compile)"
        if self.compile_enabled:
            return "enabled (ultralytics compile arg)"
        if self.compile_requested:
            return "disabled (fallback to eager)"
        return "disabled"

    def _snapshot_settings_state(self) -> dict[str, dict[str, Any]]:
        return {
            "model": dict(self.model_cfg),
            "inference": dict(self.inference_cfg),
            "tracker": dict(self.tracker_cfg),
            "security": dict(self.security_cfg),
            "uniform": dict(self.uniform_cfg),
            "events": dict(self.events_cfg),
            "runtime": dict(self.runtime_cfg),
        }

    def _format_setting_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, dict, tuple, set)):
            try:
                return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
            except Exception:  # noqa: BLE001
                return str(value)
        return str(value)

    def _collect_setting_changes(
        self,
        before: dict[str, dict[str, Any]],
        after: dict[str, dict[str, Any]],
    ) -> list[str]:
        changes: list[str] = []
        sections = sorted(set(before.keys()) | set(after.keys()))
        for section in sections:
            before_map = dict(before.get(section, {}))
            after_map = dict(after.get(section, {}))
            keys = sorted(set(before_map.keys()) | set(after_map.keys()))
            for key in keys:
                old_value = before_map.get(key)
                new_value = after_map.get(key)
                if old_value == new_value:
                    continue
                changes.append(
                    f"{section}.{key}: {self._format_setting_value(old_value)} -> {self._format_setting_value(new_value)}"
                )
        return changes

    def _log_active_model_runtime(self, *, reason: str) -> None:
        model_path_text = self.model_reference or "-"
        classes = self.predict_kwargs.get("classes")
        classes_text = "all" if not classes else str(classes)
        self._log(
            "Model runtime "
            f"[{reason}] "
            f"path={model_path_text} | "
            f"conf={self.predict_kwargs.get('conf')} iou={self.predict_kwargs.get('iou')} "
            f"imgsz={self.predict_kwargs.get('imgsz')} max_det={self.predict_kwargs.get('max_det')} "
            f"device={self.predict_kwargs.get('device', 'auto')} half={self.predict_kwargs.get('half', False)} "
            f"classes={classes_text} | "
            f"compile_requested={self.compile_requested} compile_status={self._compile_status_text()} | "
            f"day_seg={self.day_seg_model_reference or '-'}"
        )

    def _load_model_with_progress_dialog(
        self,
        model_cfg: dict[str, Any],
        *,
        dialog_title: str,
        model_name: str,
    ) -> tuple[YOLO, str]:
        dialog = ModelDownloadProgressDialog(dialog_title, model_name, self)
        dialog.show()
        dialog.update_from_progress("Start pobierania...", 0, None)

        def _progress_cb(status: str, downloaded: int | None, total: int | None) -> None:
            dialog.update_from_progress(status, downloaded, total)

        try:
            model, reference = load_yolo_model(model_cfg, base_dir=Path.cwd(), progress_callback=_progress_cb)
            dialog.mark_complete("Model pobrany i gotowy do uzycia.")
            dialog.exec()
            return model, reference
        except Exception:
            dialog.reject()
            raise

    def _preflight_compile_predict(self) -> None:
        if self.model is None or not self.compile_enabled:
            return

        probe_imgsz = int(self.predict_kwargs.get("imgsz", self.inference_cfg.get("imgsz", 960)))
        probe_imgsz = max(64, min(2048, probe_imgsz))
        probe_frame = np.zeros((probe_imgsz, probe_imgsz, 3), dtype=np.uint8)

        with self._model_lock:
            try:
                _ = self.model.predict([probe_frame], **self.predict_kwargs)
            except Exception as exc:  # noqa: BLE001
                if not self._is_compile_runtime_error(exc):
                    raise

                self.compile_enabled = False
                self.inference_cfg["compile"] = False
                self.predict_kwargs.pop("compile", None)
                try:
                    self.model.predictor = None
                except Exception:  # noqa: BLE001
                    pass

                self._log(
                    "compile preflight failed, fallback to compile=False "
                    f"({exc.__class__.__name__}: {exc})"
                )

    def _try_manual_backend_compile(self) -> bool:
        if self.model is None or not self.compile_enabled:
            return False
        if not hasattr(torch, "compile"):
            return False

        compile_value = self.inference_cfg.get("compile", True)
        if isinstance(compile_value, bool):
            compile_mode = "default"
        else:
            compile_mode = str(compile_value).strip() or "default"

        probe_imgsz = int(self.predict_kwargs.get("imgsz", self.inference_cfg.get("imgsz", 960)))
        probe_imgsz = max(64, min(2048, probe_imgsz))
        probe_frame = np.zeros((probe_imgsz, probe_imgsz, 3), dtype=np.uint8)
        probe_kwargs = dict(self.predict_kwargs)
        probe_kwargs.pop("compile", None)

        with self._model_lock:
            try:
                _ = self.model.predict([probe_frame], **probe_kwargs)
                predictor = getattr(self.model, "predictor", None)
                auto_backend = getattr(predictor, "model", None)
                backend = getattr(auto_backend, "backend", None)
                backend_model = getattr(backend, "model", None)
                if backend_model is None or not isinstance(backend_model, torch.nn.Module):
                    return False

                compiled_model = torch.compile(backend_model, mode=compile_mode, backend="inductor")
                backend.model = compiled_model

                # Run one tiny eager call through predictor API so the compiled graph is built early.
                _ = self.model.predict([probe_frame], **probe_kwargs)

                self.manual_compile_active = True
                self.compile_enabled = False
                self.predict_kwargs.pop("compile", None)
                self._log(f"manual backend compile enabled (mode={compile_mode})")
                return True
            except Exception as exc:  # noqa: BLE001
                self._log(
                    "manual backend compile failed, fallback to eager "
                    f"({exc.__class__.__name__}: {exc})"
                )
                self.manual_compile_active = False
                return False

    def _load_model(self) -> None:
        ensure_windows_compile_env(self.inference_cfg, compile_value=self.inference_cfg.get("compile", False))
        self.compile_requested = _is_compile_enabled(self.inference_cfg.get("compile", False))
        self.compile_enabled = self.compile_requested

        model, reference, reference_path = self._resolve_model_reference()
        self.model = model
        self.model_reference = reference
        self.current_model_path = reference_path
        self.compile_fallback_applied = False
        self.manual_compile_active = False

        self._rebuild_predict_kwargs()
        self._load_optional_day_segmentation_model()
        if self.compile_enabled and not self._try_manual_backend_compile():
            self._preflight_compile_predict()
        self._log(f"Model loaded: {reference}")
        if self.day_seg_model is not None:
            self._log(f"Day segmentation model loaded: {self.day_seg_model_reference}")
        self._log_active_model_runtime(reason="load")

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self.setWindowTitle(str(self.runtime_cfg.get("window_title", "Intrusion Detection")))
        self.resize(1600, 960)

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.main_tabs = QTabWidget()
        self.main_tabs.installEventFilter(self)
        self.main_tabs.tabBar().installEventFilter(self)
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tabs.addTab(self._build_preview_tab(), "Podglad kamer")
        camera_config_page = self._build_camera_config_tab()
        events_page = self._build_events_tab()
        self.main_tabs.addTab(self._wrap_main_tab_page(camera_config_page), "Konfiguracja kamer")
        self.main_tabs.addTab(self._wrap_main_tab_page(events_page), "Wykryty ruch")
        self.settings_tab_page = self._build_settings_tab()
        self._settings_tab_index = self.main_tabs.addTab(
            self._wrap_main_tab_page(self.settings_tab_page), "Ustawienia"
        )
        logs_page = self._build_logs_tab()
        self.main_tabs.addTab(self._wrap_main_tab_page(logs_page), "Logi")
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
        layout.addWidget(self.main_tabs)

        self.navigation_corner_widget = QWidget(self.main_tabs)
        navigation_corner_layout = QHBoxLayout(self.navigation_corner_widget)
        navigation_corner_layout.setContentsMargins(0, 0, 0, 0)
        navigation_corner_layout.setSpacing(0)

        self.navigation_toggle_btn = QPushButton(self.navigation_corner_widget)
        self.navigation_toggle_btn.setFixedSize(34, 30)
        self.navigation_toggle_btn.clicked.connect(self._toggle_navigation_tabs_visibility)
        self.navigation_toggle_btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgba(20, 24, 31, 225);"
            "color: #e7edf8;"
            "border: 1px solid #4a5568;"
            "border-radius: 6px;"
            "font-size: 16px;"
            "font-weight: 700;"
            "}"
            "QPushButton:hover { background-color: rgba(32, 38, 48, 235); }"
        )
        navigation_corner_layout.addWidget(self.navigation_toggle_btn)
        self.main_tabs.setCornerWidget(self.navigation_corner_widget, Qt.Corner.TopLeftCorner)

        self.navigation_overlay_btn = QPushButton(root)
        self.navigation_overlay_btn.setFixedSize(34, 30)
        self.navigation_overlay_btn.clicked.connect(self._toggle_navigation_tabs_visibility)
        self.navigation_overlay_btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgba(20, 24, 31, 225);"
            "color: #e7edf8;"
            "border: 1px solid #4a5568;"
            "border-radius: 6px;"
            "font-size: 16px;"
            "font-weight: 700;"
            "}"
            "QPushButton:hover { background-color: rgba(32, 38, 48, 235); }"
        )
        self.navigation_overlay_btn.hide()

        self._update_navigation_toggle_button()

        self.exit_app_btn = QPushButton("Exit", root)
        self.exit_app_btn.setFixedSize(86, 34)
        self.exit_app_btn.clicked.connect(self.close)
        self.exit_app_btn.setToolTip("Zamknij aplikacje")
        self.exit_app_btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgba(178, 45, 45, 235);"
            "color: white;"
            "border: 1px solid #702525;"
            "border-radius: 0px;"
            "font-weight: 600;"
            "}"
            "QPushButton:hover { background-color: rgba(212, 54, 54, 245); }"
        )
        self.exit_app_btn.raise_()

        self._set_controls_from_config()
        self._bind_setting_autosave()
        self.main_tabs.setCurrentIndex(0)
        self._last_main_tab_index = 0
        self.preview_tabs.setCurrentIndex(0)
        self._apply_theme()
        self._set_navigation_tabs_visibility(self.navigation_tabs_visible, persist=False)
        self._position_overlay_controls()

    def _wrap_main_tab_page(self, page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(page)
        return scroll

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#111417"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#d9dee7"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#10151d"))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#171c26"))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#171c26"))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#d9dee7"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#d9dee7"))
            palette.setColor(QPalette.ColorRole.Button, QColor("#1d2430"))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#d9dee7"))
            palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Highlight, QColor("#2f81f7"))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Light, QColor("#2b3342"))
            palette.setColor(QPalette.ColorRole.Midlight, QColor("#222835"))
            palette.setColor(QPalette.ColorRole.Dark, QColor("#0d1117"))
            palette.setColor(QPalette.ColorRole.Mid, QColor("#303743"))
            palette.setColor(QPalette.ColorRole.Shadow, QColor("#000000"))
            palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#94a2b8"))
            app.setPalette(palette)

        self.setStyleSheet(
            "QMainWindow { background: #111417; }"
            "QWidget { color: #d9dee7; font-size: 13px; }"
            "QTabWidget::pane { border: 1px solid #2a2f38; top: 0px; background: #151922; }"
            "QTabBar::tab {"
            "background: #222835;"
            "color: #b9c2d0;"
            "padding: 9px 16px;"
            "margin-right: 1px;"
            "border-top-left-radius: 0px;"
            "border-top-right-radius: 0px;"
            "}"
            "QTabBar::tab:selected {"
            "background: #2f81f7;"
            "color: #ffffff;"
            "font-weight: 600;"
            "}"
            "QTabBar::tab:hover:!selected { background: #2b3342; }"
            "QToolBox { background: #151b24; border: 1px solid #303743; }"
            "QToolBox > QWidget { background: #151b24; border: 1px solid #303743; }"
            "QToolBox::tab {"
            "background: #1b2230;"
            "color: #c8d0de;"
            "border: 1px solid #313a4a;"
            "border-radius: 4px;"
            "padding: 7px 10px;"
            "}"
            "QToolBox::tab:selected { background: #2f81f7; color: #ffffff; font-weight: 600; }"
            "QToolBox::tab:hover:!selected { background: #273142; }"
            "QGroupBox {"
            "border: 1px solid #303743;"
            "border-radius: 8px;"
            "margin-top: 10px;"
            "padding: 8px;"
            "background: #171c26;"
            "}"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #c8d0de; }"
            "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {"
            "background: #10151d;"
            "border: 1px solid #3a4452;"
            "border-radius: 6px;"
            "padding: 4px;"
            "selection-background-color: #2f81f7;"
            "selection-color: #ffffff;"
            "}"
            "QTableWidget { gridline-color: #2e3644; }"
            "QTableWidget::item { background: #10151d; color: #dfe6f3; }"
            "QTableWidget::item:selected { background: #2b313a; color: #ffffff; }"
            "QPushButton {"
            "background: #2f81f7;"
            "color: white;"
            "border: 1px solid #2363c0;"
            "border-radius: 7px;"
            "padding: 6px 12px;"
            "font-weight: 600;"
            "}"
            "QPushButton:hover { background: #3f8cff; }"
            "QPushButton:pressed { background: #2363c0; }"
            "QPushButton:disabled { background: #374457; color: #94a2b8; border: 1px solid #4b5b72; }"
            "QHeaderView::section {"
            "background: #1d2430;"
            "color: #d2d9e6;"
            "border: 1px solid #313a4a;"
            "padding: 5px;"
            "}"
            "QHeaderView::section:horizontal { background: #1d2430; color: #d2d9e6; }"
            "QHeaderView::section:vertical { background: #1d2430; color: #d2d9e6; }"
            "QTableCornerButton::section { background: #1d2430; border: 1px solid #313a4a; }"
            "QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #53647f; background: #0f1520; border-radius: 3px; }"
            "QCheckBox::indicator:checked {"
            "background: #2aa96b;"
            "border: 1px solid #1f8a55;"
            "image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 10'><path fill='%23ffffff' d='M4.5 7.5L1.5 4.5L0 6L4.5 10.5L12 3L10.5 1.5z'/></svg>\");"
            "}"
            "QScrollArea { background: #10141b; border: 1px solid #2a303c; }"
            "QScrollBar:vertical { background: #161c27; width: 12px; margin: 0px; border: 1px solid #2b3341; }"
            "QScrollBar::handle:vertical { background: #3a4659; min-height: 18px; border-radius: 5px; }"
            "QScrollBar::handle:vertical:hover { background: #4b5c73; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
            "QScrollBar:horizontal { background: #161c27; height: 12px; margin: 0px; border: 1px solid #2b3341; }"
            "QScrollBar::handle:horizontal { background: #3a4659; min-width: 18px; border-radius: 5px; }"
            "QScrollBar::handle:horizontal:hover { background: #4b5c73; }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }"
            "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }"
        )

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        root_layout = QVBoxLayout(page)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        header_actions = QHBoxLayout()
        header_actions.setContentsMargins(0, 0, 0, 0)
        header_actions.setSpacing(8)
        self.settings_unsaved_label = QLabel("Niezapisane zmiany")
        self.settings_unsaved_label.setStyleSheet("color: #ffb74d; font-weight: 700;")
        self.settings_unsaved_label.setVisible(False)
        self.settings_reset_btn = QPushButton("Reset do domyslnych")
        self.settings_reset_btn.clicked.connect(self._reset_settings_to_defaults)
        self.settings_load_preset_btn = QPushButton("Wczytaj preset")
        self.settings_load_preset_btn.clicked.connect(self._load_settings_preset)
        self.settings_save_preset_btn = QPushButton("Zapisz preset")
        self.settings_save_preset_btn.clicked.connect(self._save_settings_preset)
        self.settings_apply_btn = QPushButton("Potwierdz i zapisz")
        self.settings_apply_btn.clicked.connect(self._confirm_and_save_settings)
        header_actions.addStretch(1)
        header_actions.addWidget(self.settings_unsaved_label)
        header_actions.addWidget(self.settings_reset_btn)
        header_actions.addWidget(self.settings_load_preset_btn)
        header_actions.addWidget(self.settings_save_preset_btn)
        header_actions.addWidget(self.settings_apply_btn)
        layout.addLayout(header_actions)

        settings_box_style = (
            "QGroupBox {"
            "font-size: 16px;"
            "font-weight: 700;"
            "color: #e3e9f4;"
            "border: 1px solid #2b3341;"
            "border-radius: 8px;"
            "margin-top: 14px;"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "subcontrol-position: top left;"
            "padding: 2px 10px;"
            "background: #151a23;"
            "border-radius: 6px;"
            "}"
        )

        def _make_toggle_row(label: str) -> tuple[QWidget, QPushButton]:
            btn = QPushButton("")
            btn.setCheckable(True)
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(
                "QPushButton {"
                "background: #1a202b;"
                "border: 1px solid #3b4657;"
                "border-radius: 4px;"
                "color: #ffffff;"
                "padding: 0px;"
                "font-size: 14px;"
                "font-weight: 800;"
                "}"
                "QPushButton:checked {"
                "background: #2aa96b;"
                "border: 1px solid #1f8a55;"
                "}"
            )
            btn.toggled.connect(lambda checked, b=btn: b.setText("✓" if checked else ""))
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            row_layout.addWidget(btn)
            row_layout.addWidget(QLabel(label))
            row_layout.addStretch(1)
            return row, btn

        top_settings_row = QHBoxLayout()
        top_settings_row.setSpacing(10)

        security_box = QGroupBox("Reguly alarmu (dzien / noc)")
        security_box.setStyleSheet(settings_box_style)
        security_grid = QGridLayout(security_box)

        self.security_mode_combo = QComboBox()
        self.security_mode_combo.addItems(["auto", "day", "night"])

        self.night_start_spin = QSpinBox()
        self.night_start_spin.setRange(0, 23)
        self.night_end_spin = QSpinBox()
        self.night_end_spin.setRange(0, 23)

        self.day_threshold_spin = QSpinBox()
        self.day_threshold_spin.setRange(1, 99)
        self.night_threshold_spin = QSpinBox()
        self.night_threshold_spin.setRange(1, 99)

        security_grid.addWidget(QLabel("Tryb pracy:"), 0, 0)
        security_grid.addWidget(self.security_mode_combo, 0, 1)
        security_grid.addWidget(QLabel("Noc od (godzina):"), 1, 0)
        security_grid.addWidget(self.night_start_spin, 1, 1)
        security_grid.addWidget(QLabel("Noc do (godzina):"), 2, 0)
        security_grid.addWidget(self.night_end_spin, 2, 1)
        security_grid.addWidget(QLabel("Ilosc intruzow do uruchomienia alarmu w dzien:"), 3, 0)
        security_grid.addWidget(self.day_threshold_spin, 3, 1)
        security_grid.addWidget(QLabel("Ilosc osob do uruchomienia alarmu w nocy:"), 4, 0)
        security_grid.addWidget(self.night_threshold_spin, 4, 1)
        top_settings_row.addWidget(security_box, stretch=1)

        inference_box = QGroupBox("Profil detekcji YOLO")
        inference_box.setStyleSheet(settings_box_style)
        inference_layout = QHBoxLayout(inference_box)
        inference_layout.setContentsMargins(12, 12, 12, 12)
        inference_layout.setSpacing(16)

        self.yolo_profile_combo = QComboBox()
        self.yolo_profile_combo.setMaxVisibleItems(len(YOLO_PROFILE_PRESETS) + 1)
        for profile_key, preset in YOLO_PROFILE_PRESETS.items():
            self.yolo_profile_combo.addItem(str(preset["combo_label"]), profile_key)
        self.yolo_profile_combo.addItem("Custom - reczne ustawienia", YOLO_PROFILE_CUSTOM)

        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.setMaxVisibleItems(20)
        self.yolo_day_seg_model_combo = QComboBox()
        self.yolo_day_seg_model_combo.setMaxVisibleItems(20)

        self.model_target_fps_spin = QSpinBox()
        self.model_target_fps_spin.setRange(1, 60)
        self.model_target_fps_spin.setSingleStep(1)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.99)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.setMaxVisibleItems(len(YOLO_IMGSZ_OPTIONS))
        for imgsz_value in YOLO_IMGSZ_OPTIONS:
            self.imgsz_combo.addItem(f"{imgsz_value} px", imgsz_value)

        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)

        self.device_edit = QLineEdit()
        self.device_edit.setPlaceholderText("auto / 0 / cpu")
        half_row, self.half_checkbox = _make_toggle_row("FP16 (half precision)")
        compile_row, self.compile_checkbox = _make_toggle_row("torch.compile (jesli stabilne)")
        startmax_row, self.start_maximized_checkbox = _make_toggle_row("Start aplikacji w trybie zmaksymalizowanym")
        self.yolo_profile_help_label = QLabel("")
        self.yolo_profile_help_label.setWordWrap(True)
        self.yolo_profile_help_label.setStyleSheet("color: #9fb0c9;")
        self.yolo_profile_summary_label = QLabel("")
        self.yolo_profile_summary_label.setWordWrap(True)
        self.yolo_profile_summary_label.setStyleSheet("color: #d8d8d8;")

        inference_left_widget = QWidget()
        inference_left_grid = QGridLayout(inference_left_widget)
        inference_left_grid.setContentsMargins(0, 0, 0, 0)
        inference_left_grid.setHorizontalSpacing(10)
        inference_left_grid.setVerticalSpacing(8)
        inference_left_grid.addWidget(QLabel("Preset:"), 0, 0)
        inference_left_grid.addWidget(self.yolo_profile_combo, 0, 1)
        inference_left_grid.addWidget(QLabel("Model YOLO (noc):"), 1, 0)
        inference_left_grid.addWidget(self.yolo_model_combo, 1, 1)
        inference_left_grid.addWidget(QLabel("Model YOLO (dzien):"), 2, 0)
        inference_left_grid.addWidget(self.yolo_day_seg_model_combo, 2, 1)
        inference_left_grid.addWidget(QLabel("FPS modelu:"), 3, 0)
        inference_left_grid.addWidget(self.model_target_fps_spin, 3, 1)
        inference_left_grid.addWidget(QLabel("Urzadzenie:"), 4, 0)
        inference_left_grid.addWidget(self.device_edit, 4, 1)
        inference_left_grid.addWidget(self.yolo_profile_help_label, 5, 0, 1, 2)
        inference_left_grid.addWidget(self.yolo_profile_summary_label, 6, 0, 1, 2)
        inference_left_grid.addWidget(half_row, 7, 0, 1, 2)
        inference_left_grid.addWidget(compile_row, 8, 0, 1, 2)
        inference_left_grid.addWidget(startmax_row, 9, 0, 1, 2)

        inference_right_widget = QWidget()
        inference_right_grid = QGridLayout(inference_right_widget)
        inference_right_grid.setContentsMargins(0, 0, 0, 0)
        inference_right_grid.setHorizontalSpacing(10)
        inference_right_grid.setVerticalSpacing(8)
        inference_right_grid.addWidget(QLabel("Rozdzielczosc modelu:"), 0, 0)
        inference_right_grid.addWidget(self.imgsz_combo, 0, 1)
        inference_right_grid.addWidget(QLabel("Prog pewnosci (conf):"), 1, 0)
        inference_right_grid.addWidget(self.conf_spin, 1, 1)
        inference_right_grid.addWidget(QLabel("Prog IOU (NMS):"), 2, 0)
        inference_right_grid.addWidget(self.iou_spin, 2, 1)
        inference_right_grid.addWidget(QLabel("Maks. liczba detekcji:"), 3, 0)
        inference_right_grid.addWidget(self.max_det_spin, 3, 1)
        inference_right_grid.setRowStretch(4, 1)

        inference_layout.addWidget(inference_left_widget, stretch=11)
        inference_layout.addWidget(inference_right_widget, stretch=9)

        events_box = QGroupBox("Archiwizacja zdarzen")
        events_box.setStyleSheet(settings_box_style)
        events_grid = QGridLayout(events_box)

        events_enabled_row, self.events_enabled_checkbox = _make_toggle_row(
            "Zapisz klip wideo, gdy osoba jest widoczna dluzej niz prog"
        )
        self.events_min_visible_spin = QDoubleSpinBox()
        self.events_min_visible_spin.setRange(0.3, 120.0)
        self.events_min_visible_spin.setSingleStep(0.2)
        self.events_min_visible_spin.setDecimals(1)

        self.events_cooldown_spin = QDoubleSpinBox()
        self.events_cooldown_spin.setRange(0.0, 3600.0)
        self.events_cooldown_spin.setSingleStep(0.5)
        self.events_cooldown_spin.setDecimals(1)

        self.events_linger_spin = QDoubleSpinBox()
        self.events_linger_spin.setRange(0.0, 30.0)
        self.events_linger_spin.setSingleStep(0.2)
        self.events_linger_spin.setDecimals(1)

        self.events_min_person_spin = QSpinBox()
        self.events_min_person_spin.setRange(1, 20)

        self.events_clip_fps_spin = QSpinBox()
        self.events_clip_fps_spin.setRange(1, 120)
        self.events_clip_fps_spin.setSingleStep(1)

        self.events_prebuffer_spin = QDoubleSpinBox()
        self.events_prebuffer_spin.setRange(0.0, 10.0)
        self.events_prebuffer_spin.setSingleStep(0.5)
        self.events_prebuffer_spin.setDecimals(1)

        self.events_max_saved_spin = QSpinBox()
        self.events_max_saved_spin.setRange(0, 20000)
        self.events_max_saved_spin.setSpecialValueText("0 (bez limitu)")

        save_annotated_row, self.events_save_annotated_checkbox = _make_toggle_row(
            "Zapisuj klip z boxami i opisem"
        )
        once_row, self.events_once_per_streak_checkbox = _make_toggle_row(
            "Tylko jeden zapis na ciagla sekwencje wykrycia"
        )

        self.events_output_dir_edit = QLineEdit()
        self.events_output_dir_edit.setPlaceholderText("logs/app/events")
        self.events_output_dir_browse_btn = QPushButton("Browse")
        self.events_output_dir_browse_btn.clicked.connect(self._browse_events_output_dir)
        output_row = QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.addWidget(self.events_output_dir_edit, stretch=1)
        output_row.addWidget(self.events_output_dir_browse_btn)
        output_row_widget = QWidget()
        output_row_widget.setLayout(output_row)

        events_grid.addWidget(events_enabled_row, 0, 0, 1, 2)
        events_grid.addWidget(QLabel("Minimalny czas widocznosci (s):"), 1, 0)
        events_grid.addWidget(self.events_min_visible_spin, 1, 1)
        events_grid.addWidget(QLabel("Cooldown miedzy zapisami (s):"), 2, 0)
        events_grid.addWidget(self.events_cooldown_spin, 2, 1)
        events_grid.addWidget(QLabel("Dodatkowy czas po zaniku (s):"), 3, 0)
        events_grid.addWidget(self.events_linger_spin, 3, 1)
        events_grid.addWidget(QLabel("Min. liczba osob:"), 4, 0)
        events_grid.addWidget(self.events_min_person_spin, 4, 1)
        events_grid.addWidget(QLabel("FPS zapisu klipu:"), 5, 0)
        events_grid.addWidget(self.events_clip_fps_spin, 5, 1)
        events_grid.addWidget(QLabel("Pre-event buffer (s):"), 6, 0)
        events_grid.addWidget(self.events_prebuffer_spin, 6, 1)
        events_grid.addWidget(QLabel("Maks. liczba zapisanych zdarzen:"), 7, 0)
        events_grid.addWidget(self.events_max_saved_spin, 7, 1)
        events_grid.addWidget(save_annotated_row, 8, 0, 1, 2)
        events_grid.addWidget(once_row, 9, 0, 1, 2)
        events_grid.addWidget(QLabel("Folder zapisu zdarzen:"), 10, 0)
        events_grid.addWidget(output_row_widget, 10, 1)

        top_settings_row.addWidget(events_box, stretch=1)
        layout.addLayout(top_settings_row)
        layout.addWidget(inference_box)

        uniform_box = QGroupBox("Wzorzec ubioru dziennego")
        uniform_box.setStyleSheet(settings_box_style)
        uniform_layout = QHBoxLayout(uniform_box)
        uniform_layout.setContentsMargins(12, 12, 12, 12)
        uniform_layout.setSpacing(18)

        uniform_left_widget = QWidget()
        uniform_left_grid = QGridLayout(uniform_left_widget)
        uniform_left_grid.setContentsMargins(0, 0, 0, 0)
        uniform_left_grid.setHorizontalSpacing(10)
        uniform_left_grid.setVerticalSpacing(8)

        uniform_enabled_row, self.uniform_enabled_checkbox = _make_toggle_row(
            "W dzien wykrywaj intruza po niezgodnym ubiorze"
        )

        self.uniform_tolerance_spin = QDoubleSpinBox()
        self.uniform_tolerance_spin.setRange(UNIFORM_COLOR_TOLERANCE_MIN, UNIFORM_COLOR_TOLERANCE_MAX)
        self.uniform_tolerance_spin.setSingleStep(1.0)
        self.uniform_tolerance_spin.setDecimals(1)

        self.uniform_min_pixels_spin = QSpinBox()
        self.uniform_min_pixels_spin.setRange(20, 50000)
        self.uniform_min_pixels_spin.setSingleStep(20)

        self.uniform_top_color_btn = QPushButton()
        self.uniform_top_color_btn.clicked.connect(lambda: self._choose_uniform_color("top"))
        self.uniform_bottom_color_btn = QPushButton()
        self.uniform_bottom_color_btn.clicked.connect(lambda: self._choose_uniform_color("bottom"))

        self.uniform_help_label = QLabel(
            "W trybie dziennym aplikacja uzywa modelu segmentacji osoby, "
            "wycina sylwetke maska i porownuje kolor gornej oraz dolnej czesci ubioru z wzorcem. "
            "Tolerancja 0-100: im wieksza, tym wiecej odcieni uznaje za zgodne; biel i czern sa "
            "dopasowywane z lekkim luzem na cien i ekspozycje kamery."
        )
        self.uniform_help_label.setWordWrap(True)
        self.uniform_help_label.setStyleSheet("color: #9fb0c9;")

        uniform_left_grid.addWidget(uniform_enabled_row, 0, 0, 1, 2)
        uniform_left_grid.addWidget(QLabel("Kolor gornej czesci:"), 1, 0)
        uniform_left_grid.addWidget(self.uniform_top_color_btn, 1, 1)
        uniform_left_grid.addWidget(QLabel("Kolor dolnej czesci:"), 2, 0)
        uniform_left_grid.addWidget(self.uniform_bottom_color_btn, 2, 1)
        uniform_left_grid.addWidget(QLabel("Tolerancja koloru (LAB):"), 3, 0)
        uniform_left_grid.addWidget(self.uniform_tolerance_spin, 3, 1)
        uniform_left_grid.addWidget(QLabel("Min. pikseli maski na sekcje:"), 4, 0)
        uniform_left_grid.addWidget(self.uniform_min_pixels_spin, 4, 1)
        uniform_left_grid.addWidget(self.uniform_help_label, 5, 0, 1, 2)

        uniform_right_widget = QWidget()
        uniform_right_layout = QVBoxLayout(uniform_right_widget)
        uniform_right_layout.setContentsMargins(0, 0, 0, 0)
        uniform_right_layout.setSpacing(8)
        self.uniform_preview_widget = UniformPreviewWidget()
        self.uniform_preview_title = QLabel("Podglad zaprogramowanego stroju")
        self.uniform_preview_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.uniform_preview_title.setStyleSheet("color: #d8d8d8; font-weight: 600;")
        uniform_right_layout.addWidget(self.uniform_preview_title)
        uniform_right_layout.addWidget(self.uniform_preview_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        uniform_right_layout.addStretch(1)

        uniform_layout.addWidget(uniform_left_widget, stretch=12)
        uniform_layout.addWidget(uniform_right_widget, stretch=7)
        layout.addWidget(uniform_box)

        model_box = QGroupBox("Zaawansowany wybor modelu")
        model_box.setStyleSheet(settings_box_style)
        model_layout = QVBoxLayout(model_box)

        self.model_help_label = QLabel(
            "Na co dzien uzywaj presetu wyzej. "
            "Tutaj mozesz recznie wymusic konkretny model. "
            "Base = surowe wagi (np. yolo26n.pt), "
            "trained/latest = ostatni best/last z treningu, "
            "trained/final = najlepszy model utrwalony per architektura."
        )
        self.model_help_label.setWordWrap(True)
        self.model_help_label.setStyleSheet("color: #9fb0c9;")
        model_layout.addWidget(self.model_help_label)

        self.model_table = QTableWidget(0, 7)
        self.model_table.setHorizontalHeaderLabels(
            [
                "Dostepny",
                "Model",
                "Zrodlo",
                "Arch",
                "Size(MB)",
                "Run",
                "Path",
            ]
        )
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.verticalHeader().setDefaultSectionSize(30)
        self.model_table.setMinimumHeight(420)
        self.model_table.itemDoubleClicked.connect(self._apply_selected_model)

        header = self.model_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        model_layout.addWidget(self.model_table)

        model_buttons = QHBoxLayout()
        refresh_models_btn = QPushButton("Odswiez liste modeli")
        refresh_models_btn.clicked.connect(self._refresh_model_catalog)
        download_model_btn = QPushButton("Pobierz wybrany model")
        download_model_btn.clicked.connect(self._download_selected_model)
        apply_model_btn = QPushButton("Zaladuj wybrany model")
        apply_model_btn.clicked.connect(self._apply_selected_model)
        for btn in (refresh_models_btn, download_model_btn, apply_model_btn):
            btn.setFixedWidth(280)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        model_buttons.addWidget(refresh_models_btn)
        model_buttons.addWidget(download_model_btn)
        model_buttons.addWidget(apply_model_btn)
        model_buttons.addStretch(1)
        model_layout.addLayout(model_buttons)

        layout.addWidget(model_box)

        layout.addStretch(1)
        scroll.setWidget(content)
        root_layout.addWidget(scroll, stretch=1)
        return page

    def _build_camera_config_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        table_box = QGroupBox("Sources")
        table_layout = QVBoxLayout(table_box)
        table_box.setStyleSheet(
            "QGroupBox {"
            "font-size: 16px;"
            "font-weight: 700;"
            "color: #e3e9f4;"
            "border: 1px solid #2b3341;"
            "border-radius: 8px;"
            "margin-top: 14px;"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "subcontrol-position: top left;"
            "padding: 2px 10px;"
            "background: #151a23;"
            "border-radius: 6px;"
            "}"
        )

        self.source_table = QTableWidget(0, 6)
        self.source_table.setHorizontalHeaderLabels(["Name", "Type", "Value", "Mask", "Random", "Enabled"])
        self.source_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.source_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.source_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.source_table.verticalHeader().setVisible(False)
        self.source_table.verticalHeader().setDefaultSectionSize(30)
        self.source_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.source_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.source_table.setStyleSheet(
            "QTableWidget { font-size: 12px; }"
            "QHeaderView::section {"
            "font-size: 13px;"
            "font-weight: 600;"
            "padding: 6px;"
            "}"
        )

        header = self.source_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._update_source_table_height()

        self.source_table.itemChanged.connect(self._on_source_item_changed)
        table_layout.addWidget(self.source_table)

        source_buttons = QHBoxLayout()
        remove_source_btn = QPushButton("Remove selected source")
        remove_source_btn.clicked.connect(self._remove_selected_source)
        edit_mask_btn = QPushButton("Edit mask")
        edit_mask_btn.clicked.connect(self._open_source_mask_editor)
        save_source_btn = QPushButton("Save camera config")
        save_source_btn.clicked.connect(lambda: self._persist_config(show_message=True))
        source_buttons.addWidget(remove_source_btn)
        source_buttons.addWidget(edit_mask_btn)
        source_buttons.addWidget(save_source_btn)
        table_layout.addLayout(source_buttons)

        layout.addWidget(table_box)

        add_box = QGroupBox("Dodaj zrodlo")
        add_layout = QVBoxLayout(add_box)
        add_box.setStyleSheet(
            "QGroupBox {"
            "font-size: 16px;"
            "font-weight: 700;"
            "color: #e3e9f4;"
            "border: 1px solid #2b3341;"
            "border-radius: 8px;"
            "margin-top: 14px;"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "subcontrol-position: top left;"
            "padding: 2px 10px;"
            "background: #151a23;"
            "border-radius: 6px;"
            "}"
        )

        selector_row = QHBoxLayout()
        selector_label = QLabel("Typ zrodla:")
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(
            [
                "Kamera",
                "Plik wideo",
                "Strumien (RTSP/HTTP)",
            ]
        )
        selector_row.addWidget(selector_label)
        selector_row.addWidget(self.source_type_combo, stretch=1)
        add_layout.addLayout(selector_row)

        self.source_add_stack = QStackedWidget()
        self.source_add_stack.addWidget(self._build_add_camera_page())
        self.source_add_stack.addWidget(self._build_add_video_page())
        self.source_add_stack.addWidget(self._build_add_stream_page())
        add_layout.addWidget(self.source_add_stack)

        self.source_type_combo.currentIndexChanged.connect(self.source_add_stack.setCurrentIndex)
        layout.addWidget(add_box)

        layout.addStretch(1)
        return page

    def _build_add_camera_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        row = QHBoxLayout()
        self.camera_combo = QComboBox()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_camera_list)
        row.addWidget(self.camera_combo, stretch=1)
        row.addWidget(refresh_btn)
        layout.addLayout(row)

        self.camera_name_edit = QLineEdit()
        self.camera_name_edit.setPlaceholderText("Name (optional)")
        layout.addWidget(self.camera_name_edit)

        add_btn = QPushButton("Add camera")
        add_btn.clicked.connect(self._add_camera_source)
        add_btn.setFixedWidth(240)
        add_row = QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        layout.addStretch(1)
        if self.auto_scan_cameras_on_startup:
            self._refresh_camera_list()
        else:
            self.camera_combo.clear()
            self.camera_combo.addItem("Click Refresh to scan cameras", -1)
        return page

    def _build_add_video_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Path to file")
        self.video_path_edit.textChanged.connect(self._update_video_duplicate_hint)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_video_file)
        row.addWidget(self.video_path_edit, stretch=1)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        self.video_name_edit = QLineEdit()
        self.video_name_edit.setPlaceholderText("Name (optional)")
        layout.addWidget(self.video_name_edit)

        self.video_random_start_checkbox = QCheckBox("Startuj od losowego momentu")
        layout.addWidget(self.video_random_start_checkbox)

        self.video_duplicate_label = QLabel("")
        self.video_duplicate_label.setStyleSheet("color: #f0b429;")
        self.video_duplicate_label.setVisible(False)
        layout.addWidget(self.video_duplicate_label)

        add_btn = QPushButton("Add video source")
        add_btn.clicked.connect(self._add_video_source)
        add_btn.setFixedWidth(240)
        add_row = QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        layout.addStretch(1)
        return page

    def _build_add_stream_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.stream_url_edit = QLineEdit()
        self.stream_url_edit.setPlaceholderText("rtsp://... or http://...")
        layout.addWidget(self.stream_url_edit)

        self.stream_name_edit = QLineEdit()
        self.stream_name_edit.setPlaceholderText("Name (optional)")
        layout.addWidget(self.stream_name_edit)

        add_btn = QPushButton("Add stream source")
        add_btn.clicked.connect(self._add_stream_source)
        add_btn.setFixedWidth(240)
        add_row = QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        layout.addStretch(1)
        return page

    def _build_preview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.preview_tabs = QTabWidget()
        self.preview_tabs.installEventFilter(self)
        self.preview_tabs.tabBar().installEventFilter(self)
        self.preview_tabs.currentChanged.connect(self._on_preview_subtab_changed)
        self.preview_tabs.addTab(self._build_live_tab(), "Live")
        self.preview_tabs.addTab(self._build_recordings_tab(), "Nagrania")
        layout.addWidget(self.preview_tabs)
        return page

    def _build_live_tab(self) -> QWidget:
        page = QWidget()
        self.live_tab_page = page
        layout = QVBoxLayout(page)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.live_view_container = QWidget(page)
        self.live_view_container.installEventFilter(self)
        self.live_view_layout = QVBoxLayout(self.live_view_container)
        self.live_view_layout.setSpacing(0)
        self.live_view_layout.setContentsMargins(0, 0, 0, 0)

        self.live_placeholder = QLabel("No enabled sources. Add or enable sources in Kamera config.")
        self.live_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_placeholder.setStyleSheet("background:#111; color:#8a8a8a; border:1px dashed #444;")

        self.live_scroll = QScrollArea()
        self.live_scroll.setWidgetResizable(True)
        self.live_scroll.viewport().installEventFilter(self)

        self.live_grid_widget = QWidget()
        self.live_grid_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.live_grid_layout = QGridLayout(self.live_grid_widget)
        self.live_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.live_grid_layout.setHorizontalSpacing(self.live_tile_spacing)
        self.live_grid_layout.setVerticalSpacing(self.live_tile_spacing)
        self.live_scroll.setWidget(self.live_grid_widget)

        self.live_view_layout.addWidget(self.live_placeholder, stretch=1)
        self.live_view_layout.addWidget(self.live_scroll, stretch=10)
        self.live_scroll.hide()


        self.live_header_toggle_btn = QPushButton(self.preview_tabs)
        self.live_header_toggle_btn.setFixedSize(34, 30)
        self.live_header_toggle_btn.clicked.connect(self._toggle_live_tile_header_visibility)
        self.live_header_toggle_btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgba(20, 24, 31, 225);"
            "color: #e7edf8;"
            "border: 1px solid #4a5568;"
            "border-radius: 6px;"
            "font-size: 16px;"
            "font-weight: 700;"
            "}"
            "QPushButton:hover { background-color: rgba(32, 38, 48, 235); }"
        )
        self._update_live_header_toggle_button()

        self._update_live_overlay_margin()
        self._on_preview_subtab_changed(0)

        layout.addWidget(self.live_view_container, stretch=1)
        return page

    def _build_recordings_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        self.recording_path_edit = QLineEdit()
        self.recording_path_edit.setPlaceholderText("Path to recording file")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_recording_file)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_recording_from_input)
        top_row.addWidget(self.recording_path_edit, stretch=1)
        top_row.addWidget(browse_btn)
        top_row.addWidget(load_btn)
        layout.addLayout(top_row)

        controls = QHBoxLayout()
        self.recording_play_btn = QPushButton("Play")
        self.recording_pause_btn = QPushButton("Pause")
        self.recording_stop_btn = QPushButton("Stop")
        self.recording_zoom_out_btn = QPushButton("-")
        self.recording_zoom_in_btn = QPushButton("+")
        self.recording_zoom_reset_btn = QPushButton("Reset zoom")
        self.recording_time_label = QLabel("00:00 / 00:00")

        self.recording_play_btn.clicked.connect(self._recording_play)
        self.recording_pause_btn.clicked.connect(self._recording_pause)
        self.recording_stop_btn.clicked.connect(self._recording_stop)
        self.recording_zoom_out_btn.clicked.connect(lambda: self._change_recording_zoom(-1))
        self.recording_zoom_in_btn.clicked.connect(lambda: self._change_recording_zoom(1))
        self.recording_zoom_reset_btn.clicked.connect(self._reset_recording_zoom)

        controls.addWidget(self.recording_play_btn)
        controls.addWidget(self.recording_pause_btn)
        controls.addWidget(self.recording_stop_btn)
        controls.addSpacing(12)
        controls.addWidget(self.recording_zoom_out_btn)
        controls.addWidget(self.recording_zoom_in_btn)
        controls.addWidget(self.recording_zoom_reset_btn)
        controls.addStretch(1)
        controls.addWidget(self.recording_time_label)
        layout.addLayout(controls)

        self.recording_canvas = VideoCanvas("recording")
        self.recording_canvas.clicked.connect(self._noop_click)
        self.recording_canvas.right_clicked.connect(self._noop_click)
        self.recording_canvas.zoom_delta.connect(self._on_recording_zoom_delta)
        self.recording_canvas.pan_delta.connect(self._on_recording_pan_delta)
        self.recording_canvas.setText("Load recording to start preview.")
        layout.addWidget(self.recording_canvas, stretch=12)

        self.recording_slider = QSlider(Qt.Orientation.Horizontal)
        self.recording_slider.setRange(0, 0)
        self.recording_slider.sliderPressed.connect(self._on_recording_slider_pressed)
        self.recording_slider.sliderReleased.connect(self._on_recording_slider_released)
        self.recording_slider.valueChanged.connect(self._on_recording_slider_changed)
        layout.addWidget(self.recording_slider)

        help_label = QLabel("Recording controls: drag slider to seek, wheel to zoom, hold left and drag to pan.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #bfbfbf;")
        layout.addWidget(help_label)

        return page

    def _build_events_tab(self) -> QWidget:
        page = QWidget()
        self.events_tab_page = page
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        filters_box = QGroupBox("Filtry wykrytego ruchu")
        filters_layout = QGridLayout(filters_box)

        self.events_period_combo = QComboBox()
        self.events_camera_combo = QComboBox()
        self.events_mode_combo = QComboBox()
        self.events_mode_combo.addItem("Wszystkie tryby", "all")
        self.events_mode_combo.addItem("Tylko dzien", "day")
        self.events_mode_combo.addItem("Tylko noc", "night")

        self.events_exact_day_checkbox = QCheckBox("Konkretny dzien")
        self.events_exact_day_edit = QDateEdit()
        self.events_exact_day_edit.setCalendarPopup(True)
        self.events_exact_day_edit.setDisplayFormat("yyyy-MM-dd")
        self.events_exact_day_edit.setDate(QDate.currentDate())
        self.events_exact_day_edit.setKeyboardTracking(False)
        self.events_hour_filter_checkbox = QCheckBox("Konkretne godziny")
        self.events_hour_from_edit = QTimeEdit()
        self.events_hour_from_edit.setDisplayFormat("HH:mm")
        self.events_hour_from_edit.setTime(QTime(0, 0))
        self.events_hour_from_edit.setKeyboardTracking(False)
        self.events_hour_to_edit = QTimeEdit()
        self.events_hour_to_edit.setDisplayFormat("HH:mm")
        self.events_hour_to_edit.setTime(QTime(23, 59))
        self.events_hour_to_edit.setKeyboardTracking(False)

        filters_layout.addWidget(QLabel("Okres:"), 0, 0)
        filters_layout.addWidget(self.events_period_combo, 0, 1)
        filters_layout.addWidget(QLabel("Kamera:"), 0, 2)
        filters_layout.addWidget(self.events_camera_combo, 0, 3)
        filters_layout.addWidget(QLabel("Tryb:"), 1, 0)
        filters_layout.addWidget(self.events_mode_combo, 1, 1)
        filters_layout.addWidget(self.events_exact_day_checkbox, 1, 2)
        filters_layout.addWidget(self.events_exact_day_edit, 1, 3)
        filters_layout.addWidget(self.events_hour_filter_checkbox, 2, 0)
        filters_layout.addWidget(self.events_hour_from_edit, 2, 1)
        filters_layout.addWidget(QLabel("do"), 2, 2)
        filters_layout.addWidget(self.events_hour_to_edit, 2, 3)

        layout.addWidget(filters_box)

        self.events_table = QTableWidget(0, 7)
        self.events_table.setHorizontalHeaderLabels(["Data", "Godzina", "Kamera", "Tryb", "Osoby", "Widocznosc[s]", "Plik"])
        self.events_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.events_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.events_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.itemSelectionChanged.connect(self._on_event_table_selection_changed)
        self.events_table.setStyleSheet(
            "QHeaderView::section {"
            "background-color: #202838;"
            "color: #eef3fb;"
            "font-size: 15px;"
            "font-weight: 700;"
            "padding: 8px 6px;"
            "border: 1px solid #334155;"
            "}"
        )
        events_header = self.events_table.horizontalHeader()
        events_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        events_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        events_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        events_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        events_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        events_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        events_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.events_table.setColumnWidth(0, 118)
        self.events_table.setColumnWidth(1, 92)
        self.events_table.setColumnWidth(2, 170)
        self.events_table.setColumnWidth(3, 78)
        self.events_table.setColumnWidth(4, 68)
        self.events_table.setColumnWidth(5, 140)

        self._events_filter_timer = QTimer(self)
        self._events_filter_timer.setSingleShot(True)
        self._events_filter_timer.timeout.connect(self._refresh_events_table)

        self.events_preview = VideoCanvas("event")
        self.events_preview.clicked.connect(self._noop_click)
        self.events_preview.right_clicked.connect(self._noop_click)
        self.events_preview.zoom_delta.connect(lambda _name, _delta: None)
        self.events_preview.pan_delta.connect(lambda _name, _dx, _dy: None)
        self.events_preview.setText("Brak zapisanych zdarzen.")

        self.events_status_label = QLabel("Saved events: 0")
        self.events_status_label.setStyleSheet("color: #bfc9da;")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.events_table)
        splitter.addWidget(self.events_preview)
        splitter.setSizes([340, 280])

        controls = QHBoxLayout()
        refresh_btn = QPushButton("Refresh list")
        refresh_btn.clicked.connect(self._refresh_events_table)
        open_btn = QPushButton("Open selected file")
        open_btn.clicked.connect(self._open_selected_event_file)
        clear_btn = QPushButton("Wyczysc wszystkie zapisane zdarzenia")
        clear_btn.clicked.connect(self._clear_all_events)
        controls.addWidget(refresh_btn)
        controls.addWidget(open_btn)
        controls.addWidget(clear_btn)
        controls.addStretch(1)
        controls.addWidget(self.events_status_label)

        self.events_period_combo.currentIndexChanged.connect(self._schedule_events_table_refresh)
        self.events_camera_combo.currentIndexChanged.connect(self._schedule_events_table_refresh)
        self.events_mode_combo.currentIndexChanged.connect(self._schedule_events_table_refresh)
        self.events_exact_day_checkbox.toggled.connect(self._schedule_events_table_refresh)
        self.events_exact_day_edit.dateChanged.connect(self._schedule_events_table_refresh)
        self.events_exact_day_edit.editingFinished.connect(self._schedule_events_table_refresh)
        self.events_hour_filter_checkbox.toggled.connect(self._schedule_events_table_refresh)
        self.events_hour_from_edit.timeChanged.connect(self._schedule_events_table_refresh)
        self.events_hour_to_edit.timeChanged.connect(self._schedule_events_table_refresh)
        self.events_hour_from_edit.editingFinished.connect(self._schedule_events_table_refresh)
        self.events_hour_to_edit.editingFinished.connect(self._schedule_events_table_refresh)

        layout.addWidget(splitter, stretch=1)
        layout.addLayout(controls)
        self._refresh_events_filter_controls()
        self._refresh_events_table()
        return page

    def _build_logs_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        layout.addWidget(self.logs_text, stretch=1)

        button_row = QHBoxLayout()
        clear_btn = QPushButton("Clear logs")
        clear_btn.clicked.connect(self._clear_logs)
        export_btn = QPushButton("Export logs")
        export_btn.clicked.connect(self._export_logs)
        button_row.addWidget(clear_btn)
        button_row.addWidget(export_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        if self._log_entries:
            self.logs_text.setPlainText("\n".join(self._log_entries))
            cursor = self.logs_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.logs_text.setTextCursor(cursor)

        return page

    def _infer_yolo_profile_key(self) -> str:
        model_name = str(self.model_cfg.get("name", "")).strip()
        target_fps = float(self.runtime_cfg.get("model_target_fps", self.model_target_fps))
        imgsz_value = int(self.inference_cfg.get("imgsz", 960))
        conf_value = float(self.inference_cfg.get("conf", 0.35))
        iou_value = float(self.inference_cfg.get("iou", 0.45))
        max_det_value = int(self.inference_cfg.get("max_det", 100))
        return self._match_yolo_profile(
            model_name=model_name,
            target_fps=target_fps,
            conf_value=conf_value,
            iou_value=iou_value,
            imgsz_value=imgsz_value,
            max_det_value=max_det_value,
        )

    def _match_yolo_profile(
        self,
        *,
        model_name: str,
        target_fps: float,
        conf_value: float,
        iou_value: float,
        imgsz_value: int,
        max_det_value: int,
    ) -> str:
        normalized_model = str(model_name or "").strip().lower()
        for profile_key, preset in YOLO_PROFILE_PRESETS.items():
            if (
                normalized_model == str(preset["model_name"]).strip().lower()
                and abs(float(target_fps) - float(preset["target_fps"])) < 0.0001
                and abs(float(conf_value) - float(preset["conf"])) < 0.0001
                and abs(float(iou_value) - float(preset["iou"])) < 0.0001
                and int(imgsz_value) == int(preset["imgsz"])
                and int(max_det_value) == int(preset["max_det"])
            ):
                return profile_key
        return YOLO_PROFILE_CUSTOM

    def _set_combo_to_data(self, combo: QComboBox, data_value: Any, *, fallback_index: int = 0) -> None:
        index = combo.findData(data_value)
        combo.setCurrentIndex(fallback_index if index < 0 else index)

    def _build_model_selection(self, entry: dict[str, Any]) -> dict[str, str]:
        model_name = str(entry.get("model_name", entry.get("name", ""))).strip()
        selected_model_path = ""
        try:
            path_value = Path(entry.get("path", ""))
            if path_value.exists():
                selected_model_path = _to_relative_or_abs(path_value.resolve())
        except Exception:  # noqa: BLE001
            selected_model_path = str(entry.get("path_display", "")).strip()
        return {
            "model_name": model_name,
            "selected_model_path": selected_model_path,
        }

    def _entry_is_segmentation_model(self, entry: dict[str, Any]) -> bool:
        model_name = str(entry.get("model_name", entry.get("name", ""))).strip().lower()
        task_name = str(entry.get("task", "")).strip().lower()
        return model_name.endswith("-seg.pt") or task_name in {"segment", "seg"}

    def _current_yolo_model_selection(self) -> dict[str, str]:
        data_value = self.yolo_model_combo.currentData()
        if isinstance(data_value, dict):
            return {
                "model_name": str(data_value.get("model_name", "")).strip(),
                "selected_model_path": str(data_value.get("selected_model_path", "")).strip(),
            }
        return {
            "model_name": str(self.model_cfg.get("name", "")).strip(),
            "selected_model_path": str(self.model_cfg.get("selected_model_path", "")).strip(),
        }

    def _current_day_seg_model_selection(self) -> dict[str, str]:
        seg_cfg = self._day_segmentation_model_cfg()
        if hasattr(self, "yolo_day_seg_model_combo") and self.yolo_day_seg_model_combo is not None:
            data_value = self.yolo_day_seg_model_combo.currentData()
            if isinstance(data_value, dict):
                return {
                    "model_name": str(data_value.get("model_name", "")).strip(),
                    "selected_model_path": str(data_value.get("selected_model_path", "")).strip(),
                }
        return {
            "model_name": str(seg_cfg.get("name", DEFAULT_DAY_SEG_MODEL_NAME)).strip(),
            "selected_model_path": str(seg_cfg.get("selected_model_path", "")).strip(),
        }

    def _populate_yolo_model_combo(self) -> None:
        if not hasattr(self, "yolo_model_combo") or self.yolo_model_combo is None:
            return

        current_selection = self._current_yolo_model_selection()
        combo_model = self.yolo_model_combo.model()
        self.yolo_model_combo.clear()

        installed_entries = [
            entry
            for entry in self.model_catalog
            if not bool(entry.get("missing", False)) and not self._entry_is_segmentation_model(entry)
        ]
        recommended_entries: list[dict[str, Any]] = []
        older_entries: list[dict[str, Any]] = []
        for entry in installed_entries:
            if self._is_recommended_model_entry(entry):
                recommended_entries.append(entry)
            else:
                older_entries.append(entry)

        recommended_entries.sort(key=self._model_catalog_sort_key)
        older_entries.sort(key=self._model_catalog_sort_key)

        def _add_section(title: str, entries: list[dict[str, Any]]) -> None:
            if not entries:
                return
            self.yolo_model_combo.addItem(title)
            header_index = self.yolo_model_combo.count() - 1
            if hasattr(combo_model, "item"):
                header_item = combo_model.item(header_index)
                if isinstance(header_item, QStandardItem):
                    header_item.setEnabled(False)
                    header_item.setSelectable(False)
                    header_item.setForeground(QColor("#7d889a"))
            for entry in entries:
                label = f"{entry.get('display_name', entry.get('name', '-'))} [{entry.get('kind', 'custom')}]"
                selection = self._build_model_selection(entry)
                self.yolo_model_combo.addItem(label, selection)

        _add_section("Zalecane (najnowsze)", recommended_entries)
        _add_section("Starsze", older_entries)

        fallback_selection = current_selection
        if not fallback_selection["model_name"] and installed_entries:
            fallback_selection = self._build_model_selection(installed_entries[0])
        self._set_model_combo_value(fallback_selection)

    def _populate_day_seg_model_combo(self) -> None:
        if not hasattr(self, "yolo_day_seg_model_combo") or self.yolo_day_seg_model_combo is None:
            return

        current_selection = self._current_day_seg_model_selection()
        combo_model = self.yolo_day_seg_model_combo.model()
        self.yolo_day_seg_model_combo.clear()

        installed_seg_entries = [
            entry
            for entry in self.model_catalog
            if self._entry_is_segmentation_model(entry) and not bool(entry.get("missing", False))
        ]
        installed_seg_entries.sort(key=self._model_catalog_sort_key)

        if installed_seg_entries:
            self.yolo_day_seg_model_combo.addItem("Segmentacja - zainstalowane")
            header_index = self.yolo_day_seg_model_combo.count() - 1
            if hasattr(combo_model, "item"):
                header_item = combo_model.item(header_index)
                if isinstance(header_item, QStandardItem):
                    header_item.setEnabled(False)
                    header_item.setSelectable(False)
                    header_item.setForeground(QColor("#7d889a"))
            for entry in installed_seg_entries:
                label = f"{entry.get('display_name', entry.get('name', '-'))} [{entry.get('kind', 'custom')}]"
                self.yolo_day_seg_model_combo.addItem(label, self._build_model_selection(entry))
        else:
            self.yolo_day_seg_model_combo.addItem("Brak lokalnych modeli segm - pobierz z tabeli na dole")
            header_index = self.yolo_day_seg_model_combo.count() - 1
            if hasattr(combo_model, "item"):
                header_item = combo_model.item(header_index)
                if isinstance(header_item, QStandardItem):
                    header_item.setEnabled(False)
                    header_item.setSelectable(False)
                    header_item.setForeground(QColor("#7d889a"))

        fallback_selection = current_selection
        first_available = installed_seg_entries[0] if installed_seg_entries else None
        if not fallback_selection["model_name"] and first_available is not None:
            fallback_selection = self._build_model_selection(first_available)
        self._set_day_seg_model_combo_value(fallback_selection)

    def _set_model_combo_value(self, selection: dict[str, str] | str) -> None:
        if isinstance(selection, str):
            target_model_name = str(selection).strip()
            target_path = ""
        else:
            target_model_name = str(selection.get("model_name", "")).strip()
            target_path = str(selection.get("selected_model_path", "")).strip()

        for index in range(self.yolo_model_combo.count()):
            data_value = self.yolo_model_combo.itemData(index)
            if not isinstance(data_value, dict):
                continue
            candidate_model_name = str(data_value.get("model_name", "")).strip()
            candidate_path = str(data_value.get("selected_model_path", "")).strip()
            if target_path and candidate_path == target_path:
                self.yolo_model_combo.setCurrentIndex(index)
                return
            if candidate_model_name == target_model_name and (not target_path or candidate_path == target_path):
                self.yolo_model_combo.setCurrentIndex(index)
                return

        for index in range(self.yolo_model_combo.count()):
            if isinstance(self.yolo_model_combo.itemData(index), dict):
                self.yolo_model_combo.setCurrentIndex(index)
                return

    def _set_day_seg_model_combo_value(self, selection: dict[str, str] | str) -> None:
        if not hasattr(self, "yolo_day_seg_model_combo") or self.yolo_day_seg_model_combo is None:
            return
        if isinstance(selection, str):
            target_model_name = str(selection).strip()
            target_path = ""
        else:
            target_model_name = str(selection.get("model_name", "")).strip()
            target_path = str(selection.get("selected_model_path", "")).strip()

        for index in range(self.yolo_day_seg_model_combo.count()):
            data_value = self.yolo_day_seg_model_combo.itemData(index)
            if not isinstance(data_value, dict):
                continue
            candidate_model_name = str(data_value.get("model_name", "")).strip()
            candidate_path = str(data_value.get("selected_model_path", "")).strip()
            if target_path and candidate_path == target_path:
                self.yolo_day_seg_model_combo.setCurrentIndex(index)
                return
            if candidate_model_name == target_model_name and (not target_path or candidate_path == target_path):
                self.yolo_day_seg_model_combo.setCurrentIndex(index)
                return

        for index in range(self.yolo_day_seg_model_combo.count()):
            if isinstance(self.yolo_day_seg_model_combo.itemData(index), dict):
                self.yolo_day_seg_model_combo.setCurrentIndex(index)
                return

    def _set_imgsz_combo_value(self, imgsz_value: int) -> None:
        label = f"{int(imgsz_value)} px"
        index = self.imgsz_combo.findData(int(imgsz_value))
        if index < 0:
            self.imgsz_combo.addItem(f"{label} (custom)", int(imgsz_value))
            index = self.imgsz_combo.count() - 1
        self.imgsz_combo.setCurrentIndex(index)

    def _current_imgsz_value(self) -> int:
        data_value = self.imgsz_combo.currentData()
        if data_value is None:
            return 960
        return int(data_value)

    def _sync_yolo_profile_combo_from_controls(self) -> None:
        model_name = self._current_yolo_model_selection()["model_name"]
        matched_profile = self._match_yolo_profile(
            model_name=model_name,
            target_fps=float(int(self.model_target_fps_spin.value())),
            conf_value=float(self.conf_spin.value()),
            iou_value=float(self.iou_spin.value()),
            imgsz_value=self._current_imgsz_value(),
            max_det_value=int(self.max_det_spin.value()),
        )
        self._suppress_setting_autosave = True
        try:
            self._set_combo_to_data(self.yolo_profile_combo, matched_profile)
        finally:
            self._suppress_setting_autosave = False

    def _on_yolo_manual_control_changed(self, *_args: Any) -> None:
        if self._suppress_setting_autosave:
            return
        self._sync_yolo_profile_combo_from_controls()
        self._update_yolo_profile_summary()
        self._on_setting_changed()

    def _apply_yolo_profile_to_controls(self, profile_key: str) -> None:
        preset = YOLO_PROFILE_PRESETS.get(profile_key, YOLO_PROFILE_PRESETS["medium"])
        self._set_model_combo_value({"model_name": str(preset["model_name"]), "selected_model_path": ""})
        self._set_day_seg_model_combo_value(
            {"model_name": str(preset.get("day_seg_model_name", DEFAULT_DAY_SEG_MODEL_NAME)), "selected_model_path": ""}
        )
        self.model_target_fps_spin.setValue(int(round(float(preset["target_fps"]))))
        self.conf_spin.setValue(float(preset["conf"]))
        self.iou_spin.setValue(float(preset["iou"]))
        self._set_imgsz_combo_value(int(preset["imgsz"]))
        self.max_det_spin.setValue(int(preset["max_det"]))

    def _update_yolo_profile_summary(self) -> None:
        if not hasattr(self, "yolo_profile_combo") or self.yolo_profile_combo is None:
            return

        profile_key = str(self.yolo_profile_combo.currentData() or self._infer_yolo_profile_key())
        preset = YOLO_PROFILE_PRESETS.get(profile_key)
        if preset is None:
            self.yolo_profile_help_label.setText(
                "Custom: recznie dopasowany zestaw ustawien. "
                "Mozesz laczyc lepszy model z nizsza rozdzielczoscia albo mniejszym max_det."
            )
        else:
            self.yolo_profile_help_label.setText(
                f"{preset['title']}: {preset['description']} Domyslny model: {preset['model_name']}, "
                f"segm: {preset.get('day_seg_model_name', DEFAULT_DAY_SEG_MODEL_NAME)}."
            )
        current_model_name = self._current_yolo_model_selection()["model_name"]
        day_seg_model_name = self._current_day_seg_model_selection()["model_name"]
        self.yolo_profile_summary_label.setText(
            "Aplikacja ustawi: "
            f"model={current_model_name} | conf={self.conf_spin.value():.2f} | "
            f"IOU={self.iou_spin.value():.2f} | imgsz={self._current_imgsz_value()} | "
            f"max_det={int(self.max_det_spin.value())} | fps_modelu={int(self.model_target_fps_spin.value())} | "
            f"seg_dzien={day_seg_model_name or '-'}"
        )

    def _update_uniform_color_button(self, button: QPushButton, color_hex: str, label: str) -> None:
        normalized = _normalize_hex_color(color_hex, UNIFORM_TOP_DEFAULT)
        rgb = QColor(normalized)
        brightness = (rgb.red() * 299 + rgb.green() * 587 + rgb.blue() * 114) / 1000.0
        text_color = "#0E1117" if brightness >= 150 else "#F7FAFF"
        button.setText(f"{label}: {normalized}")
        button.setStyleSheet(
            "QPushButton {"
            f"background: {normalized};"
            f"color: {text_color};"
            "font-weight: 700;"
            "border: 1px solid #3b4657;"
            "border-radius: 6px;"
            "padding: 6px 10px;"
            "}"
        )

    def _sync_uniform_preview(self) -> None:
        top_hex = str(getattr(self, "_selected_uniform_top_color", self.uniform_cfg.get("top_color", UNIFORM_TOP_DEFAULT)))
        bottom_hex = str(
            getattr(self, "_selected_uniform_bottom_color", self.uniform_cfg.get("bottom_color", UNIFORM_BOTTOM_DEFAULT))
        )
        self._update_uniform_color_button(self.uniform_top_color_btn, top_hex, "Gora")
        self._update_uniform_color_button(self.uniform_bottom_color_btn, bottom_hex, "Dol")
        self.uniform_preview_widget.set_colors(top_hex, bottom_hex)

    def _update_uniform_controls_enabled(self) -> None:
        enabled = bool(self.uniform_enabled_checkbox.isChecked())
        for widget in (
            self.yolo_day_seg_model_combo,
            self.uniform_top_color_btn,
            self.uniform_bottom_color_btn,
            self.uniform_tolerance_spin,
            self.uniform_min_pixels_spin,
            self.uniform_preview_widget,
        ):
            widget.setEnabled(enabled)

    def _choose_uniform_color(self, section: str) -> None:
        current_value = (
            getattr(self, "_selected_uniform_top_color", self.uniform_cfg.get("top_color", UNIFORM_TOP_DEFAULT))
            if section == "top"
            else getattr(self, "_selected_uniform_bottom_color", self.uniform_cfg.get("bottom_color", UNIFORM_BOTTOM_DEFAULT))
        )
        picked = QColorDialog.getColor(QColor(_normalize_hex_color(current_value, UNIFORM_TOP_DEFAULT)), self, "Wybierz kolor ubioru")
        if not picked.isValid():
            return
        normalized = picked.name(QColor.NameFormat.HexRgb).upper()
        if section == "top":
            self._selected_uniform_top_color = normalized
        else:
            self._selected_uniform_bottom_color = normalized
        self._sync_uniform_preview()
        self._on_setting_changed()

    def _on_day_seg_manual_control_changed(self, *_args: Any) -> None:
        if self._suppress_setting_autosave:
            return
        self._update_yolo_profile_summary()
        self._on_setting_changed()

    def _on_uniform_enabled_toggled(self, checked: bool) -> None:
        _ = checked
        self._update_uniform_controls_enabled()
        self._update_yolo_profile_summary()
        self._on_setting_changed()

    def _on_yolo_profile_changed(self) -> None:
        if self._suppress_setting_autosave:
            return

        profile_key = str(self.yolo_profile_combo.currentData() or "").strip().lower()
        if profile_key == YOLO_PROFILE_CUSTOM:
            self._update_yolo_profile_summary()
            self._on_setting_changed()
            return
        if profile_key not in YOLO_PROFILE_PRESETS:
            return

        self._suppress_setting_autosave = True
        try:
            self._apply_yolo_profile_to_controls(profile_key)
        finally:
            self._suppress_setting_autosave = False

        preset = YOLO_PROFILE_PRESETS[profile_key]
        self.runtime_cfg["yolo_profile"] = profile_key
        self.model_cfg["name"] = str(preset["model_name"])
        self.model_cfg["selected_model_path"] = ""
        day_seg_cfg = self._day_segmentation_model_cfg()
        day_seg_cfg["name"] = str(preset.get("day_seg_model_name", DEFAULT_DAY_SEG_MODEL_NAME))
        day_seg_cfg["selected_model_path"] = ""
        self.model_cfg["day_segmentation"] = dict(day_seg_cfg)

        if not self._ensure_configured_model_available(segmentation=False):
            self._update_yolo_profile_summary()
            return
        if not self._ensure_configured_model_available(segmentation=True):
            self._update_yolo_profile_summary()
            return

        self._suppress_setting_autosave = True
        try:
            self._set_model_combo_value(
                {
                    "model_name": str(self.model_cfg.get("name", "")).strip(),
                    "selected_model_path": str(self.model_cfg.get("selected_model_path", "")).strip(),
                }
            )
            self._set_day_seg_model_combo_value(self._day_segmentation_model_cfg())
        finally:
            self._suppress_setting_autosave = False

        self._update_yolo_profile_summary()
        self._on_setting_changed()

    def _reload_model_from_config(self) -> None:
        previous_model_reference = self.model_reference
        was_running = self.live_running
        if was_running:
            self.stop_live()

        try:
            self._load_model()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model", f"Nie mozna zaladowac modelu:\n{exc}")
            if was_running:
                self.start_live()
            raise

        self._refresh_model_catalog()
        self._update_current_model_label()
        self._update_yolo_profile_summary()
        if previous_model_reference != self.model_reference:
            self._log(f"Model changed by config: {previous_model_reference} -> {self.model_reference}")
        else:
            self._log("Model reloaded due to config update (same model path).")
        self._log_active_model_runtime(reason="config-reload")

        if was_running:
            self.start_live()

    def _find_model_catalog_entry(
        self,
        selection: dict[str, str],
        *,
        segmentation: bool,
    ) -> dict[str, Any] | None:
        target_name = str(selection.get("model_name", "")).strip().lower()
        target_path = str(selection.get("selected_model_path", "")).strip()
        resolved_target_path = ""
        if target_path:
            try:
                resolved_target_path = str(resolve_path(target_path).resolve())
            except Exception:  # noqa: BLE001
                resolved_target_path = target_path

        for entry in self.model_catalog:
            if self._entry_is_segmentation_model(entry) != segmentation:
                continue
            entry_path = str(Path(entry["path"]).resolve())
            entry_name = str(entry.get("model_name", entry.get("name", ""))).strip().lower()
            if resolved_target_path and entry_path == resolved_target_path:
                return entry
            if target_name and entry_name == target_name:
                return entry
        return None

    def _closest_available_model_entry(self, target_model_name: str, *, segmentation: bool) -> dict[str, Any] | None:
        available_entries = [
            entry
            for entry in self.model_catalog
            if self._entry_is_segmentation_model(entry) == segmentation and not bool(entry.get("missing", False))
        ]
        if not available_entries:
            return None

        target_series, target_size = _parse_model_series_and_size(target_model_name)
        target_size_rank = _model_size_rank(target_size)

        def _score(entry: dict[str, Any]) -> tuple[int, int, tuple[tuple[int, int], int, int, int, float, str]]:
            entry_model_name = str(entry.get("model_name", entry.get("name", ""))).strip()
            entry_series, entry_size = _parse_model_series_and_size(entry_model_name)
            entry_size_rank = _model_size_rank(entry_size)

            series_penalty = 0
            if target_series and entry_series != target_series:
                series_penalty = 100

            size_penalty = 50
            if target_size_rank < 99 and entry_size_rank < 99:
                size_penalty = abs(entry_size_rank - target_size_rank)

            return (series_penalty, size_penalty, self._model_catalog_sort_key(entry))

        return min(available_entries, key=_score)

    def _restore_loaded_model_selection(self, *, segmentation: bool) -> None:
        self._suppress_setting_autosave = True
        try:
            if segmentation:
                seg_cfg = self._day_segmentation_model_cfg()
                if self.current_day_seg_model_path is not None and self.current_day_seg_model_path.exists():
                    seg_cfg["selected_model_path"] = _to_relative_or_abs(self.current_day_seg_model_path)
                    seg_cfg["name"] = self.current_day_seg_model_path.name
                elif self.day_seg_model_reference:
                    seg_cfg["selected_model_path"] = ""
                    seg_cfg["name"] = Path(str(self.day_seg_model_reference)).name
                self.model_cfg["day_segmentation"] = dict(seg_cfg)
                self._populate_day_seg_model_combo()
                self._set_day_seg_model_combo_value(self.model_cfg["day_segmentation"])
                return

            if self.current_model_path is not None and self.current_model_path.exists():
                self.model_cfg["selected_model_path"] = _to_relative_or_abs(self.current_model_path)
                self.model_cfg["name"] = self.current_model_path.name
            elif self.model_reference:
                self.model_cfg["selected_model_path"] = ""
                self.model_cfg["name"] = Path(str(self.model_reference)).name
            self._populate_yolo_model_combo()
            self._set_model_combo_value(
                {
                    "model_name": str(self.model_cfg.get("name", "")),
                    "selected_model_path": str(self.model_cfg.get("selected_model_path", "")).strip(),
                }
            )
        finally:
            self._suppress_setting_autosave = False

    def _ensure_configured_model_available(self, *, segmentation: bool) -> bool:
        if segmentation:
            target_cfg = self._day_segmentation_model_cfg()
            selection = {
                "model_name": str(target_cfg.get("name", DEFAULT_DAY_SEG_MODEL_NAME)).strip(),
                "selected_model_path": str(target_cfg.get("selected_model_path", "")).strip(),
            }
            dialog_title = "Model segmentacji"
        else:
            target_cfg = dict(self.model_cfg)
            selection = {
                "model_name": str(target_cfg.get("name", "")).strip(),
                "selected_model_path": str(target_cfg.get("selected_model_path", "")).strip(),
            }
            dialog_title = "Model"

        selected_path = str(selection.get("selected_model_path", "")).strip()
        if selected_path:
            try:
                if resolve_path(selected_path).exists():
                    return True
            except Exception:  # noqa: BLE001
                pass

        entry = self._find_model_catalog_entry(selection, segmentation=segmentation)
        if entry is not None and not bool(entry.get("missing", False)):
            return True

        model_name = str(selection.get("model_name", "")).strip() or (
            DEFAULT_DAY_SEG_MODEL_NAME if segmentation else str(self.model_cfg.get("name", ""))
        )
        reply = QMessageBox.question(
            self,
            dialog_title,
            f"Model {model_name} nie istnieje lokalnie. Pobierac teraz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            fallback_entry = self._closest_available_model_entry(model_name, segmentation=segmentation)
            if fallback_entry is None:
                self._restore_loaded_model_selection(segmentation=segmentation)
                self._update_yolo_profile_summary()
                return False

            fallback_path = Path(fallback_entry["path"])
            fallback_selection = {
                "model_name": str(fallback_entry.get("model_name", fallback_entry.get("name", ""))).strip(),
                "selected_model_path": _to_relative_or_abs(fallback_path),
            }

            if segmentation:
                day_seg_cfg = self._day_segmentation_model_cfg()
                day_seg_cfg["name"] = fallback_selection["model_name"]
                day_seg_cfg["selected_model_path"] = fallback_selection["selected_model_path"]
                self.model_cfg["day_segmentation"] = dict(day_seg_cfg)
                self._suppress_setting_autosave = True
                try:
                    self._set_day_seg_model_combo_value(fallback_selection)
                finally:
                    self._suppress_setting_autosave = False
            else:
                self.model_cfg["name"] = fallback_selection["model_name"]
                self.model_cfg["selected_model_path"] = fallback_selection["selected_model_path"]
                self._suppress_setting_autosave = True
                try:
                    self._set_model_combo_value(fallback_selection)
                finally:
                    self._suppress_setting_autosave = False

            QMessageBox.information(
                self,
                dialog_title,
                "Wybrany model nie zostal pobrany. "
                f"Ustawiono najblizszy dostepny model: {fallback_selection['model_name']}."
            )
            self._update_yolo_profile_summary()
            return True

        try:
            download_cfg = dict(target_cfg)
            download_cfg["name"] = model_name
            download_cfg["selected_model_path"] = ""
            model, reference = self._load_model_with_progress_dialog(
                download_cfg,
                dialog_title=dialog_title,
                model_name=model_name,
            )
            del model
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, dialog_title, f"Nie mozna pobrac modelu:\n{exc}")
            self._restore_loaded_model_selection(segmentation=segmentation)
            self._update_yolo_profile_summary()
            return False

        resolved_reference = Path(reference)
        if resolved_reference.exists():
            download_cfg["selected_model_path"] = _to_relative_or_abs(resolved_reference.resolve())

        if segmentation:
            self.model_cfg["day_segmentation"] = dict(download_cfg)
        else:
            self.model_cfg.update(download_cfg)

        self._refresh_model_catalog()
        self._suppress_setting_autosave = True
        try:
            if segmentation:
                self._populate_day_seg_model_combo()
                self._set_day_seg_model_combo_value(self.model_cfg["day_segmentation"])
            else:
                self._populate_yolo_model_combo()
                self._set_model_combo_value(
                    {
                        "model_name": str(self.model_cfg.get("name", "")),
                        "selected_model_path": str(self.model_cfg.get("selected_model_path", "")).strip(),
                    }
                )
        finally:
            self._suppress_setting_autosave = False
        self._update_yolo_profile_summary()
        return True

    def _model_config_requires_reload(self) -> bool:
        desired_path_raw = str(self.model_cfg.get("selected_model_path", "")).strip()
        current_loaded_path = self.current_model_path.resolve() if self.current_model_path and self.current_model_path.exists() else None
        if desired_path_raw:
            try:
                desired_path = resolve_path(desired_path_raw).resolve()
            except Exception:  # noqa: BLE001
                return True
            return current_loaded_path is None or desired_path != current_loaded_path

        desired_name = str(self.model_cfg.get("name", "")).strip().lower()
        current_loaded_name = ""
        if current_loaded_path is not None:
            current_loaded_name = current_loaded_path.name.strip().lower()
        elif self.model_reference:
            current_loaded_name = Path(str(self.model_reference)).name.strip().lower()
        primary_changed = bool(desired_name) and desired_name != current_loaded_name
        if primary_changed:
            return True

        seg_cfg = self._day_segmentation_model_cfg()
        seg_enabled = self._uniform_detection_enabled() and bool(seg_cfg.get("enabled", True))
        if not seg_enabled:
            return self.day_seg_model is not None

        desired_seg_path_raw = str(seg_cfg.get("selected_model_path", "")).strip()
        current_seg_path = (
            self.current_day_seg_model_path.resolve()
            if self.current_day_seg_model_path and self.current_day_seg_model_path.exists()
            else None
        )
        if desired_seg_path_raw:
            try:
                desired_seg_path = resolve_path(desired_seg_path_raw).resolve()
            except Exception:  # noqa: BLE001
                return True
            return current_seg_path is None or desired_seg_path != current_seg_path

        desired_seg_name = str(seg_cfg.get("name", "")).strip().lower()
        current_seg_name = ""
        if current_seg_path is not None:
            current_seg_name = current_seg_path.name.strip().lower()
        elif self.day_seg_model_reference:
            current_seg_name = Path(str(self.day_seg_model_reference)).name.strip().lower()
        return bool(desired_seg_name) and desired_seg_name != current_seg_name

    def _set_controls_from_config(self) -> None:
        self.security_mode_combo.setCurrentText(str(self.security_cfg.get("mode", "auto")))
        self.night_start_spin.setValue(int(self.security_cfg.get("night_start_hour", 22)))
        self.night_end_spin.setValue(int(self.security_cfg.get("night_end_hour", 6)))
        self.day_threshold_spin.setValue(int(self.security_cfg.get("day_person_threshold", 1)))
        self.night_threshold_spin.setValue(int(self.security_cfg.get("night_person_threshold", 1)))

        self.conf_spin.setValue(float(self.inference_cfg.get("conf", 0.35)))
        self.iou_spin.setValue(float(self.inference_cfg.get("iou", 0.45)))
        self._populate_yolo_model_combo()
        self._set_model_combo_value(
            {
                "model_name": str(self.model_cfg.get("name", "yolo26s.pt")),
                "selected_model_path": str(self.model_cfg.get("selected_model_path", "")).strip(),
            }
        )
        self._populate_day_seg_model_combo()
        day_seg_cfg = self._day_segmentation_model_cfg()
        self._set_day_seg_model_combo_value(
            {
                "model_name": str(day_seg_cfg.get("name", DEFAULT_DAY_SEG_MODEL_NAME)),
                "selected_model_path": str(day_seg_cfg.get("selected_model_path", "")).strip(),
            }
        )
        self._set_imgsz_combo_value(int(self.inference_cfg.get("imgsz", 960)))
        self.max_det_spin.setValue(int(self.inference_cfg.get("max_det", 100)))
        self.device_edit.setText(str(self.inference_cfg.get("device", "0")))
        self.model_target_fps_spin.setValue(
            int(round(float(self.runtime_cfg.get("model_target_fps", self.model_target_fps))))
        )
        self.half_checkbox.setChecked(bool(self.inference_cfg.get("half", True)))
        self.compile_checkbox.setChecked(_is_compile_enabled(self.inference_cfg.get("compile", False)))
        self.start_maximized_checkbox.setChecked(bool(self.runtime_cfg.get("start_maximized", True)))
        self.events_enabled_checkbox.setChecked(bool(self.events_cfg.get("enabled", True)))
        self.events_min_visible_spin.setValue(float(self.events_cfg.get("min_visible_seconds", 3.0)))
        self.events_cooldown_spin.setValue(float(self.events_cfg.get("cooldown_seconds", 10.0)))
        self.events_linger_spin.setValue(float(self.events_cfg.get("linger_seconds", 1.5)))
        self.events_min_person_spin.setValue(int(self.events_cfg.get("min_person_count", 1)))
        self.events_clip_fps_spin.setValue(int(round(float(self.events_cfg.get("clip_fps", 30.0)))))
        self.events_prebuffer_spin.setValue(float(self.events_cfg.get("prebuffer_seconds", 2.0)))
        self.events_max_saved_spin.setValue(int(self.events_cfg.get("max_saved_events", 300)))
        self.events_save_annotated_checkbox.setChecked(bool(self.events_cfg.get("save_annotated_frame", True)))
        self.events_once_per_streak_checkbox.setChecked(bool(self.events_cfg.get("once_per_streak", True)))
        self.events_output_dir_edit.setText(str(self.events_cfg.get("output_dir", "logs/app/events")))
        self.uniform_enabled_checkbox.setChecked(bool(self.uniform_cfg.get("enabled", True)))
        self.uniform_tolerance_spin.setValue(
            _uniform_color_tolerance(self.uniform_cfg.get("color_tolerance", UNIFORM_COLOR_TOLERANCE_DEFAULT))
        )
        self.uniform_min_pixels_spin.setValue(int(self.uniform_cfg.get("min_mask_pixels", UNIFORM_MIN_MASK_PIXELS_DEFAULT)))
        self._selected_uniform_top_color = _normalize_hex_color(
            self.uniform_cfg.get("top_color", UNIFORM_TOP_DEFAULT),
            UNIFORM_TOP_DEFAULT,
        )
        self._selected_uniform_bottom_color = _normalize_hex_color(
            self.uniform_cfg.get("bottom_color", UNIFORM_BOTTOM_DEFAULT),
            UNIFORM_BOTTOM_DEFAULT,
        )
        self._sync_uniform_preview()
        self._update_uniform_controls_enabled()

        detected_profile_key = self._infer_yolo_profile_key()
        self._set_combo_to_data(self.yolo_profile_combo, detected_profile_key)
        self._update_current_model_label()
        self._update_yolo_profile_summary()

        last_recording = str(self.runtime_cfg.get("last_recording_path", "")).strip()
        if last_recording:
            self.recording_path_edit.setText(last_recording)

        self._suppress_setting_autosave = False

    def _bind_setting_autosave(self) -> None:
        self.security_mode_combo.currentTextChanged.connect(self._on_setting_changed)
        self.night_start_spin.valueChanged.connect(self._on_setting_changed)
        self.night_end_spin.valueChanged.connect(self._on_setting_changed)
        self.day_threshold_spin.valueChanged.connect(self._on_setting_changed)
        self.night_threshold_spin.valueChanged.connect(self._on_setting_changed)
        self.yolo_profile_combo.currentIndexChanged.connect(self._on_yolo_profile_changed)
        self.model_target_fps_spin.valueChanged.connect(self._on_setting_changed)
        self.model_target_fps_spin.valueChanged.connect(self._update_yolo_profile_summary)
        self.yolo_model_combo.currentIndexChanged.connect(self._on_yolo_manual_control_changed)
        self.conf_spin.valueChanged.connect(self._on_yolo_manual_control_changed)
        self.iou_spin.valueChanged.connect(self._on_yolo_manual_control_changed)
        self.imgsz_combo.currentIndexChanged.connect(self._on_yolo_manual_control_changed)
        self.max_det_spin.valueChanged.connect(self._on_yolo_manual_control_changed)
        self.yolo_day_seg_model_combo.currentIndexChanged.connect(self._on_day_seg_manual_control_changed)
        self.device_edit.textChanged.connect(self._on_setting_changed)
        self.half_checkbox.toggled.connect(self._on_setting_changed)
        self.compile_checkbox.toggled.connect(self._on_setting_changed)
        self.start_maximized_checkbox.toggled.connect(self._on_setting_changed)
        self.uniform_enabled_checkbox.toggled.connect(self._on_uniform_enabled_toggled)
        self.uniform_tolerance_spin.valueChanged.connect(self._on_setting_changed)
        self.uniform_min_pixels_spin.valueChanged.connect(self._on_setting_changed)
        self.events_enabled_checkbox.toggled.connect(self._on_setting_changed)
        self.events_min_visible_spin.valueChanged.connect(self._on_setting_changed)
        self.events_cooldown_spin.valueChanged.connect(self._on_setting_changed)
        self.events_linger_spin.valueChanged.connect(self._on_setting_changed)
        self.events_min_person_spin.valueChanged.connect(self._on_setting_changed)
        self.events_clip_fps_spin.valueChanged.connect(self._on_setting_changed)
        self.events_prebuffer_spin.valueChanged.connect(self._on_setting_changed)
        self.events_max_saved_spin.valueChanged.connect(self._on_setting_changed)
        self.events_save_annotated_checkbox.toggled.connect(self._on_setting_changed)
        self.events_once_per_streak_checkbox.toggled.connect(self._on_setting_changed)
        self.events_output_dir_edit.editingFinished.connect(self._on_setting_changed)

    def _on_setting_changed(self, *_args: Any) -> None:
        if self._suppress_setting_autosave:
            return
        if not self._settings_change_tracking_ready:
            return

        self.settings_dirty = True
        if hasattr(self, "settings_unsaved_label") and self.settings_unsaved_label is not None:
            self.settings_unsaved_label.setVisible(True)

    def _clear_settings_dirty(self) -> None:
        self.settings_dirty = False
        if hasattr(self, "settings_unsaved_label") and self.settings_unsaved_label is not None:
            self.settings_unsaved_label.setVisible(False)

    def _confirm_and_save_settings(self) -> None:
        if self._commit_pending_settings():
            self._clear_settings_dirty()
        else:
            QMessageBox.warning(
                self,
                "Ustawienia",
                "Nie udalo sie zapisac ustawien. Sprawdz komunikaty o modelach i sprobuj ponownie.",
            )

    def _discard_pending_settings(self) -> None:
        self._suppress_setting_autosave = True
        try:
            self._set_controls_from_config()
        finally:
            self._suppress_setting_autosave = False
        self._clear_settings_dirty()
        self._log("Niezapisane zmiany ustawien zostaly odrzucone.")

    def _reset_settings_to_defaults(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset ustawien",
            "Przywrocic domyslne ustawienia z pliku konfiguracyjnego?\nZmiany beda wymagaly potwierdzenia zapisu.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            defaults = load_yaml(self.config_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Reset ustawien", f"Nie mozna wczytac domyslnych ustawien:\n{exc}")
            return

        if not isinstance(defaults, dict):
            QMessageBox.critical(self, "Reset ustawien", "Domyslny config ma nieprawidlowy format.")
            return

        for key in ("model", "inference", "tracker", "security", "uniform", "events", "runtime"):
            self.config[key] = dict(defaults.get(key, {}) or {})

        self.model_cfg = dict(self.config.get("model", {}) or {})
        self.inference_cfg = dict(self.config.get("inference", {}) or {})
        self.security_cfg = dict(self.config.get("security", {}) or {})
        self.uniform_cfg = dict(self.config.get("uniform", {}) or {})
        self.runtime_cfg = dict(self.config.get("runtime", {}) or {})
        self.tracker_cfg = dict(self.config.get("tracker", {}) or {})
        self.events_cfg = dict(self.config.get("events", {}) or {})

        self._suppress_setting_autosave = True
        try:
            self._set_controls_from_config()
        finally:
            self._suppress_setting_autosave = False
        self._on_setting_changed()
        self._log("Przywrocono domyslne ustawienia (oczekuja na potwierdzenie zapisu).")

    def _save_settings_preset(self) -> None:
        preset_name, ok = QInputDialog.getText(self, "Zapisz preset", "Nazwa presetu:")
        if not ok:
            return
        preset_name = str(preset_name).strip()
        if not preset_name:
            QMessageBox.information(self, "Preset", "Podaj nazwe presetu.")
            return

        self._apply_controls_to_runtime_state()
        payload = {
            "model": dict(self.model_cfg),
            "inference": dict(self.inference_cfg),
            "tracker": dict(self.tracker_cfg),
            "security": dict(self.security_cfg),
            "uniform": dict(self.uniform_cfg),
            "events": dict(self.events_cfg),
            "runtime": dict(self.runtime_cfg),
        }

        presets_dir = self.app_settings_dir / "presets"
        presets_dir.mkdir(parents=True, exist_ok=True)
        preset_file = presets_dir / f"{_safe_file_part(preset_name, fallback='preset')}.yaml"
        try:
            save_yaml(preset_file, payload)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preset", f"Nie mozna zapisac presetu:\n{exc}")
            return

        previous_settings = self._snapshot_settings_state()
        self.runtime_cfg["active_settings_preset"] = _to_relative_or_abs(preset_file)
        self._persist_config(show_message=False, previous_settings=previous_settings)
        self._clear_settings_dirty()

        self._log(f"Preset zapisany i aktywowany: {preset_file}")
        QMessageBox.information(self, "Preset", f"Zapisano i aktywowano preset:\n{preset_file}")

    def _available_settings_presets(self) -> list[Path]:
        presets_dir = self.app_settings_dir / "presets"
        if not presets_dir.exists():
            return []
        return sorted(presets_dir.glob("*.yaml"), key=lambda path: path.name.lower())

    def _load_settings_preset(self) -> None:
        preset_files = self._available_settings_presets()
        if not preset_files:
            QMessageBox.information(self, "Preset", "Brak zapisanych presetow.")
            return

        options = [preset_file.stem for preset_file in preset_files]
        selected_name, ok = QInputDialog.getItem(
            self,
            "Wczytaj preset",
            "Wybierz preset:",
            options,
            0,
            False,
        )
        if not ok:
            return

        selected_name = str(selected_name).strip()
        selected_file = next((path for path in preset_files if path.stem == selected_name), None)
        if selected_file is None:
            QMessageBox.critical(self, "Preset", "Nie znaleziono wybranego presetu.")
            return

        try:
            payload = load_yaml(selected_file)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preset", f"Nie mozna wczytac presetu:\n{exc}")
            return
        if not isinstance(payload, dict):
            QMessageBox.critical(self, "Preset", "Plik presetu ma nieprawidlowy format.")
            return

        previous_settings = self._snapshot_settings_state()
        for key in ("model", "inference", "tracker", "security", "uniform", "events", "runtime"):
            if key in payload and isinstance(payload.get(key), dict):
                self.config[key] = dict(payload.get(key, {}) or {})

        self.model_cfg = dict(self.config.get("model", {}) or {})
        self.inference_cfg = dict(self.config.get("inference", {}) or {})
        self.security_cfg = dict(self.config.get("security", {}) or {})
        self.uniform_cfg = dict(self.config.get("uniform", {}) or {})
        self.runtime_cfg = dict(self.config.get("runtime", {}) or {})
        self.tracker_cfg = dict(self.config.get("tracker", {}) or {})
        self.events_cfg = dict(self.config.get("events", {}) or {})
        self.runtime_cfg["active_settings_preset"] = _to_relative_or_abs(selected_file)

        self._suppress_setting_autosave = True
        try:
            self._set_controls_from_config()
        finally:
            self._suppress_setting_autosave = False

        if self._commit_pending_settings():
            self._clear_settings_dirty()
            self._log(f"Preset wczytany i aktywowany: {selected_file}")
            QMessageBox.information(self, "Preset", f"Wczytano preset:\n{selected_file}")
        else:
            self._log(f"[warn] Wczytano preset, ale nie udalo sie zapisac zmian: {selected_file}")
            QMessageBox.warning(
                self,
                "Preset",
                "Preset zostal wczytany do formularza, ale zapis nie powiodl sie. "
                "Sprawdz komunikaty o modelach i zapisz ponownie.",
            )

    def _on_main_tab_changed(self, index: int) -> None:
        if self._suppress_main_tab_change_handler:
            self._last_main_tab_index = index
            return
        if self._settings_tab_index < 0:
            self._last_main_tab_index = index
            return

        leaving_settings = self._last_main_tab_index == self._settings_tab_index and index != self._settings_tab_index
        if leaving_settings and self.settings_dirty:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Niezapisane zmiany")
            box.setText("Masz niezapisane zmiany w Ustawieniach.")
            box.setInformativeText("Wybierz, co zrobic przed opuszczeniem zakladki.")
            save_btn = box.addButton("Potwierdz i zapisz", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = box.addButton("Odrzuc zmiany", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = box.addButton("Wroc", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(save_btn)
            box.exec()
            clicked = box.clickedButton()

            if clicked == save_btn:
                self._confirm_and_save_settings()
            elif clicked == discard_btn:
                self._discard_pending_settings()
            else:
                self._suppress_main_tab_change_handler = True
                try:
                    self.main_tabs.setCurrentIndex(self._settings_tab_index)
                finally:
                    self._suppress_main_tab_change_handler = False
                self._last_main_tab_index = self._settings_tab_index
                return

        self._last_main_tab_index = index

    def _commit_pending_settings(self) -> bool:
        previous_settings = self._snapshot_settings_state()
        self._apply_controls_to_runtime_state()
        ensure_windows_compile_env(self.inference_cfg, compile_value=self.inference_cfg.get("compile", False))
        if self._model_config_requires_reload():
            if not self._ensure_configured_model_available(segmentation=False):
                return False
            if self._uniform_detection_enabled() and not self._ensure_configured_model_available(segmentation=True):
                return False
            try:
                self._reload_model_from_config()
            except Exception:  # noqa: BLE001
                return False
        self._rebuild_predict_kwargs()
        self._update_yolo_profile_summary()
        self._persist_config(show_message=False, previous_settings=previous_settings)
        return True

    def _flush_pending_settings(self) -> None:
        if self._settings_save_timer.isActive():
            self._settings_save_timer.stop()

    # ---------- model list ----------
    def _is_recommended_model_entry(self, entry: dict[str, Any]) -> bool:
        latest_major = self._latest_recommended_series_major()
        if latest_major < 0:
            return False
        series_key = self._extract_model_series_key(entry)
        series_major = self._extract_series_major(series_key)
        return series_major == latest_major

    def _extract_model_series_key(self, entry: dict[str, Any]) -> str:
        family = str(entry.get("family", "")).strip().lower()
        model_name = str(entry.get("model_name", entry.get("name", ""))).strip().lower()
        for text in (family, model_name):
            match = re.search(r"(yolo(?:v?\d+))", text)
            if match is not None:
                return str(match.group(1)).lower()
        return ""

    def _extract_series_major(self, series_key: str) -> int:
        match = re.search(r"yolov?(\d+)", str(series_key or "").lower())
        if match is None:
            return -1
        try:
            return int(match.group(1))
        except Exception:  # noqa: BLE001
            return -1

    def _latest_recommended_series_major(self) -> int:
        majors: list[int] = []

        for entry in self.model_catalog:
            if bool(entry.get("missing", False)):
                continue
            series_key = self._extract_model_series_key(entry)
            major = self._extract_series_major(series_key)
            if major >= 0:
                majors.append(major)

        for preset in YOLO_PROFILE_PRESETS.values():
            model_name = str((preset or {}).get("model_name", "")).strip()
            if not model_name:
                continue
            series_key = self._extract_model_series_key({"model_name": model_name})
            major = self._extract_series_major(series_key)
            if major >= 0:
                majors.append(major)

        configured_name = str(self.model_cfg.get("name", "")).strip()
        if configured_name:
            series_key = self._extract_model_series_key({"model_name": configured_name})
            major = self._extract_series_major(series_key)
            if major >= 0:
                majors.append(major)

        if not majors:
            return -1
        return max(majors)

    def _fetch_online_suggested_model_names(self) -> set[str]:
        if self._online_model_suggestions_cache is not None:
            return set(self._online_model_suggestions_cache)

        if not bool(self.runtime_cfg.get("online_model_catalog_enabled", True)):
            self._online_model_suggestions_cache = set()
            return set()

        model_names: set[str] = set()
        url = "https://api.github.com/repos/ultralytics/assets/releases"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "intrusion-detection-app",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=4.0) as response:  # noqa: S310
                payload = response.read().decode("utf-8", errors="ignore")
            parsed = json.loads(payload)
            releases = parsed if isinstance(parsed, list) else []
            for release in releases:
                assets = release.get("assets", []) if isinstance(release, dict) else []
                if not isinstance(assets, list):
                    continue
                for asset in assets:
                    if not isinstance(asset, dict):
                        continue
                    name = str(asset.get("name", "")).strip().lower()
                    if re.fullmatch(r"yolo(?:v?\d+)[nslmx](?:-seg)?\.pt", name):
                        model_names.add(name)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            if not self._online_model_suggestions_error_logged:
                self._log(f"Nie udalo sie pobrac listy modeli online: {exc}")
                self._online_model_suggestions_error_logged = True

        self._online_model_suggestions_cache = set(model_names)
        return set(model_names)

    def _model_version_group_label(self, entry: dict[str, Any]) -> str:
        series_key = self._extract_model_series_key(entry)
        latest_major = self._latest_recommended_series_major()
        series_major = self._extract_series_major(series_key)
        if series_key and series_major == latest_major:
            return f"Modele najnowsze ({series_key.upper()}) - zalecane"
        if series_key:
            return f"Modele {series_key.upper()}"
        if bool(entry.get("missing", False)):
            return "Do pobrania"
        return "Inne modele"

    def _model_version_group_rank(self, entry: dict[str, Any]) -> tuple[int, int]:
        series_key = self._extract_model_series_key(entry)
        latest_major = self._latest_recommended_series_major()
        if series_key:
            major = self._extract_series_major(series_key)
            if major == latest_major:
                return (0, -major)
            if major < 0:
                return (2, 0)
            return (1, -major)
        if bool(entry.get("missing", False)):
            return (3, 0)
        return (2, 0)

    def _model_series_rank(self, family: str) -> int:
        value = str(family or "").strip().lower()
        major = self._extract_series_major(value)
        if major >= 0:
            return 100 - major
        return 999

    def _model_size_rank(self, family: str, model_name: str) -> int:
        text = str(family or model_name or "").strip().lower()
        variant_order = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}
        if not text:
            return 99
        suffix = text[-1]
        return variant_order.get(suffix, 99)

    def _model_catalog_sort_key(self, entry: dict[str, Any]) -> tuple[tuple[int, int], int, int, int, float, str]:
        group_rank = self._model_version_group_rank(entry)
        recommended_rank = 0 if self._is_recommended_model_entry(entry) else 1

        family = str(entry.get("family", "")).strip().lower()
        model_name = str(entry.get("model_name", entry.get("name", ""))).strip().lower()
        series_rank = self._model_series_rank(family)
        size_rank = self._model_size_rank(family, model_name)

        # Najnowsze pliki (po czasie modyfikacji) ida wyzej w tej samej grupie.
        updated_ts = float(entry.get("updated_ts", 0.0) or 0.0)
        display_name = str(entry.get("display_name", entry.get("name", ""))).strip().lower()
        return (group_rank, recommended_rank, series_rank, size_rank, -updated_ts, display_name)

    def _collect_run_metrics(self) -> tuple[dict[str, dict[str, Any]], str | None]:
        logs_dir = resolve_path("logs/train")
        if not logs_dir.exists():
            return {}, None

        metrics_by_run: dict[str, dict[str, Any]] = {}
        latest_run: str | None = None
        latest_ts = -1.0

        for results_path in logs_dir.glob("*/results.csv"):
            run_name = results_path.parent.name
            metrics = _read_last_run_metrics(results_path)
            if metrics is None:
                continue
            metrics_by_run[run_name] = metrics
            if metrics["updated_ts"] > latest_ts:
                latest_ts = metrics["updated_ts"]
                latest_run = run_name

        return metrics_by_run, latest_run

    def _match_metrics_for_model(
        self,
        model_path: Path,
        metrics_by_run: dict[str, dict[str, Any]],
        latest_run: str | None,
    ) -> tuple[float | None, float | None, str | None]:
        filename = model_path.name
        run_name = _extract_run_name_from_weight_filename(filename)

        if run_name and run_name in metrics_by_run:
            metrics = metrics_by_run[run_name]
            return metrics.get("map50"), metrics.get("map5095"), run_name

        if filename in {"best.pt", "last.pt"} and model_path.parent.name == "latest" and latest_run:
            metrics = metrics_by_run.get(latest_run)
            if metrics is not None:
                return metrics.get("map50"), metrics.get("map5095"), latest_run

        stem = model_path.stem.lower()
        best_name: str | None = None
        best_ts = -1.0
        for run, metrics in metrics_by_run.items():
            run_l = run.lower()
            if stem and stem in run_l:
                ts = float(metrics.get("updated_ts", 0.0))
                if ts > best_ts:
                    best_ts = ts
                    best_name = run
        if best_name:
            metrics = metrics_by_run[best_name]
            return metrics.get("map50"), metrics.get("map5095"), best_name

        return None, None, None

    def _refresh_model_catalog(self) -> None:
        metrics_by_run, latest_run = self._collect_run_metrics()

        local_weights_dir = resolve_path(self.model_cfg.get("local_weights_dir", "models/base"))
        trained_dir = resolve_path((self.model_cfg.get("trained_weights", {}) or {}).get("dir", "models/weights/latest"))
        models_root = resolve_path("models/weights")

        candidate_paths: list[Path] = []

        if local_weights_dir.exists():
            candidate_paths.extend(sorted(local_weights_dir.glob("*.pt")))

        if trained_dir.exists():
            candidate_paths.extend(sorted(trained_dir.glob("*.pt")))

        if models_root.exists():
            candidate_paths.extend(sorted(models_root.glob("*.pt")))

        if self.current_model_path is not None and self.current_model_path.exists():
            candidate_paths.append(self.current_model_path)

        unique: dict[str, Path] = {}
        for path in candidate_paths:
            resolved = path.resolve()
            unique[str(resolved)] = resolved

        entries: list[dict[str, Any]] = []
        for path in sorted(unique.values()):
            if path.parent == trained_dir and path.name in {"best.pt", "last.pt"}:
                kind = "trained/latest"
            elif local_weights_dir in path.parents:
                kind = "base"
            elif models_root in path.parents:
                kind = "trained/final"
            else:
                kind = "custom"

            map50, map5095, run_name = self._match_metrics_for_model(path, metrics_by_run, latest_run)
            meta = _read_model_meta(path) or {}
            if isinstance(meta, dict):
                meta_map50 = meta.get("map50")
                meta_map5095 = meta.get("map5095")
                if meta_map50 is not None:
                    try:
                        map50 = float(meta_map50)
                    except Exception:  # noqa: BLE001
                        pass
                if meta_map5095 is not None:
                    try:
                        map5095 = float(meta_map5095)
                    except Exception:  # noqa: BLE001
                        pass
                meta_run = str(meta.get("run_name", "")).strip()
                if meta_run:
                    run_name = meta_run

            size_mb: float | None = None
            updated_ts = 0.0
            try:
                stat_result = path.stat()
                size_mb = stat_result.st_size / (1024 * 1024)
                updated_ts = float(stat_result.st_mtime)
            except Exception:  # noqa: BLE001
                size_mb = None
                updated_ts = 0.0

            params_m = None
            gflops = None
            task = None
            nc = None
            if isinstance(meta, dict):
                for key in ("params_m", "params"):
                    if key in meta and meta.get(key) is not None:
                        try:
                            params_m = float(meta[key])
                        except Exception:  # noqa: BLE001
                            params_m = None
                        break
                for key in ("gflops", "flops"):
                    if key in meta and meta.get(key) is not None:
                        try:
                            gflops = float(meta[key])
                        except Exception:  # noqa: BLE001
                            gflops = None
                        break
                task = str(meta.get("task", "")).strip() or None
                if meta.get("nc") is not None:
                    try:
                        nc = int(meta.get("nc"))
                    except Exception:  # noqa: BLE001
                        nc = None
                elif isinstance(meta.get("classes"), list):
                    nc = len(meta.get("classes"))

            model_name = str(meta.get("model_name", path.name)).strip() if isinstance(meta, dict) else path.name
            family = _infer_model_family(model_name if model_name else path.name)
            if path.name in {"best.pt", "last.pt"} and model_name:
                display_name = f"{path.name} <- {model_name}"
            else:
                display_name = path.name

            entries.append(
                {
                    "name": path.name,
                    "display_name": display_name,
                    "kind": kind,
                    "family": family,
                    "path": path,
                    "path_display": _to_relative_or_abs(path),
                    "map50": map50,
                    "map5095": map5095,
                    "run": run_name or "-",
                    "model_name": model_name or path.name,
                    "params_m": params_m,
                    "gflops": gflops,
                    "size_mb": size_mb,
                    "task": task,
                    "nc": nc,
                    "updated_ts": updated_ts,
                    "missing": False,
                }
            )

        suggested_names_set: set[str] = set()
        for preset in YOLO_PROFILE_PRESETS.values():
            preset_model_name = str((preset or {}).get("model_name", "")).strip()
            if preset_model_name:
                suggested_names_set.add(preset_model_name)
            preset_day_seg_name = str((preset or {}).get("day_seg_model_name", "")).strip()
            if preset_day_seg_name:
                suggested_names_set.add(preset_day_seg_name)

        selected_model_name = str(self.model_cfg.get("name", "")).strip()
        if selected_model_name:
            if not selected_model_name.lower().endswith(".pt"):
                selected_model_name = f"{selected_model_name}.pt"
            suggested_names_set.add(selected_model_name)

        configured_suggestions = self.model_cfg.get("suggested_models", [])
        if isinstance(configured_suggestions, list):
            for item in configured_suggestions:
                model_name = str(item or "").strip()
                if not model_name:
                    continue
                if not model_name.lower().endswith(".pt"):
                    model_name = f"{model_name}.pt"
                suggested_names_set.add(model_name)

        day_seg_cfg = self._day_segmentation_model_cfg()
        day_seg_model_name = str(day_seg_cfg.get("name", DEFAULT_DAY_SEG_MODEL_NAME)).strip()
        if day_seg_model_name:
            if not day_seg_model_name.lower().endswith(".pt"):
                day_seg_model_name = f"{day_seg_model_name}.pt"
            suggested_names_set.add(day_seg_model_name)
        suggested_names_set.update(DAY_SEG_MODEL_SUGGESTIONS)

        suggested_names_set.update(self._fetch_online_suggested_model_names())

        suggested_names = sorted(suggested_names_set)
        existing_names = {entry["name"] for entry in entries}
        for model_name in suggested_names:
            if model_name in existing_names:
                continue
            missing_path = (local_weights_dir / model_name).resolve()
            family = _infer_model_family(model_name)
            entries.append(
                {
                    "name": model_name,
                    "display_name": model_name,
                    "kind": "base",
                    "family": family,
                    "path": missing_path,
                    "path_display": _to_relative_or_abs(missing_path),
                    "map50": None,
                    "map5095": None,
                    "run": "-",
                    "model_name": model_name,
                    "params_m": None,
                    "gflops": None,
                    "size_mb": None,
                    "task": None,
                    "nc": None,
                    "updated_ts": 0.0,
                    "missing": True,
                }
            )

        entries.sort(key=self._model_catalog_sort_key)
        self.model_catalog = entries
        self._populate_model_table()
        if hasattr(self, "yolo_model_combo") and self.yolo_model_combo is not None:
            self._populate_yolo_model_combo()
        if hasattr(self, "yolo_day_seg_model_combo") and self.yolo_day_seg_model_combo is not None:
            self._populate_day_seg_model_combo()

    def _update_current_model_label(self) -> None:
        if not hasattr(self, "current_model_label") or self.current_model_label is None:
            self._update_yolo_profile_summary()
            return
        current_path = self.current_model_path.resolve() if self.current_model_path and self.current_model_path.exists() else None
        if current_path is None:
            self.current_model_label.setText(f"Aktualny model: {self.model_reference}")
            self._update_yolo_profile_summary()
            return

        selected_entry: dict[str, Any] | None = None
        for entry in self.model_catalog:
            try:
                if Path(entry["path"]).resolve() == current_path:
                    selected_entry = entry
                    break
            except Exception:  # noqa: BLE001
                continue

        if selected_entry is None:
            self.current_model_label.setText(f"Aktualny model: {self.model_reference}")
            self._update_yolo_profile_summary()
            return

        map50 = selected_entry.get("map50")
        map5095 = selected_entry.get("map5095")
        map50_text = "-" if map50 is None else f"{float(map50):.4f}"
        map5095_text = "-" if map5095 is None else f"{float(map5095):.4f}"
        run_name = str(selected_entry.get("run", "-"))
        family = str(selected_entry.get("family", "-"))
        kind = str(selected_entry.get("kind", "custom"))
        model_name = str(selected_entry.get("model_name", selected_entry.get("name", "-")))
        self.current_model_label.setText(
            f"Aktualny model: {model_name} | arch={family} | zrodlo={kind} | "
            f"run={run_name} | mAP50={map50_text} | mAP50-95={map5095_text}"
        )
        self._update_yolo_profile_summary()

    def _populate_model_table(self) -> None:
        rows: list[tuple[str, int | None]] = []
        last_group_label: str | None = None
        for catalog_index, entry in enumerate(self.model_catalog):
            group_label = self._model_version_group_label(entry)
            if group_label != last_group_label:
                rows.append(("header", None))
                last_group_label = group_label
            rows.append(("entry", catalog_index))

        self.model_table_row_map = [row_index for _, row_index in rows]
        self.model_table.clearSpans()
        self.model_table.setRowCount(len(rows))

        selected_row = -1
        current_path_str = str(self.current_model_path.resolve()) if self.current_model_path and self.current_model_path.exists() else ""

        for row, (row_type, catalog_index) in enumerate(rows):
            if row_type == "header":
                header_entry: dict[str, Any] | None = None
                if row + 1 < len(rows) and rows[row + 1][1] is not None:
                    header_entry = self.model_catalog[int(rows[row + 1][1])]

                header_text = "Modele"
                if header_entry is not None:
                    header_text = self._model_version_group_label(header_entry)

                self.model_table.setSpan(row, 0, 1, 7)
                header_item = QTableWidgetItem(header_text)
                header_item.setForeground(QColor("#8fb2ff"))
                header_item.setBackground(QColor("#1a2333"))
                header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.model_table.setItem(row, 0, header_item)
                continue

            if catalog_index is None:
                continue

            entry = self.model_catalog[int(catalog_index)]
            size_mb = entry.get("size_mb")

            available = "✓" if not entry.get("missing", False) else "✗"
            available_item = QTableWidgetItem(available)
            available_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.model_table.setItem(row, 0, available_item)

            model_item = QTableWidgetItem(str(entry["display_name"]))
            if self._is_recommended_model_entry(entry):
                model_item.setForeground(QColor("#7ee787"))
            self.model_table.setItem(row, 1, model_item)
            self.model_table.setItem(row, 2, QTableWidgetItem(str(entry["kind"])))
            self.model_table.setItem(row, 3, QTableWidgetItem(str(entry["family"])))
            self.model_table.setItem(row, 4, QTableWidgetItem("-" if size_mb is None else f"{size_mb:.1f}"))
            self.model_table.setItem(row, 5, QTableWidgetItem(str(entry["run"])))
            self.model_table.setItem(row, 6, QTableWidgetItem(str(entry["path_display"])))

            if current_path_str and str(entry["path"].resolve()) == current_path_str:
                selected_row = row

        if selected_row >= 0:
            self.model_table.selectRow(selected_row)

        self._update_current_model_label()

    def _selected_model_table_entry(self) -> dict[str, Any] | None:
        row = self.model_table.currentRow()
        if row < 0 or row >= len(self.model_table_row_map):
            return None
        catalog_index = self.model_table_row_map[row]
        if catalog_index is None or catalog_index < 0 or catalog_index >= len(self.model_catalog):
            return None
        return self.model_catalog[catalog_index]

    def _apply_selected_model(self) -> None:
        entry = self._selected_model_table_entry()
        if entry is None:
            QMessageBox.information(self, "Model", "Najpierw wybierz wiersz modelu.")
            return

        if self._entry_is_segmentation_model(entry):
            model_name = str(entry.get("model_name", entry.get("name", ""))).strip()
            model_path = Path(entry["path"]).resolve()
            missing = bool(entry.get("missing", False)) or not model_path.exists()
            seg_cfg = self._day_segmentation_model_cfg()
            seg_cfg["name"] = model_name or str(seg_cfg.get("name", DEFAULT_DAY_SEG_MODEL_NAME)).strip()
            if missing:
                reply = QMessageBox.question(
                    self,
                    "Model segmentacji",
                    f"Model {model_name or model_path.name} nie istnieje lokalnie. Pobierac teraz?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                try:
                    model, reference = self._load_model_with_progress_dialog(
                        seg_cfg,
                        dialog_title="Model segmentacji",
                        model_name=model_name or model_path.name,
                    )
                    del model
                    resolved_reference = Path(reference)
                    if resolved_reference.exists():
                        model_path = resolved_reference.resolve()
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.critical(self, "Model segmentacji", f"Nie mozna zaladowac modelu segmentacji:\n{exc}")
                    return

            if model_path.exists():
                seg_cfg["selected_model_path"] = _to_relative_or_abs(model_path)
            else:
                seg_cfg["selected_model_path"] = ""
            self.model_cfg["day_segmentation"] = dict(seg_cfg)

            try:
                self._reload_model_from_config()
            except Exception:
                return

            self._suppress_setting_autosave = True
            try:
                self._populate_day_seg_model_combo()
                self._set_day_seg_model_combo_value(self.model_cfg["day_segmentation"])
            finally:
                self._suppress_setting_autosave = False
            self._persist_config(show_message=False)
            self._log(f"Day segmentation model switched to: {model_path}")
            return

        model_path = Path(entry["path"]).resolve()
        missing = bool(entry.get("missing", False)) or not model_path.exists()
        if missing:
            reply = QMessageBox.question(
                self,
                "Model",
                f"Model {entry.get('model_name', entry.get('name'))} nie istnieje lokalnie. Pobierac teraz?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        was_running = self.live_running
        if was_running:
            self.stop_live()

        try:
            if missing:
                self.model_cfg["name"] = str(entry.get("model_name", entry.get("name", ""))).strip() or self.model_cfg.get(
                    "name", ""
                )
                self.model_cfg["selected_model_path"] = ""
                self.model, reference = self._load_model_with_progress_dialog(
                    self.model_cfg,
                    dialog_title="Model",
                    model_name=str(self.model_cfg.get("name", "")).strip() or str(entry.get("model_name", "model")),
                )
                model_path = Path(reference) if reference and Path(reference).exists() else model_path
            else:
                self.model = YOLO(str(model_path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model", f"Nie mozna zaladowac modelu:\n{exc}")
            if was_running:
                self.start_live()
            return

        if model_path.exists():
            self.current_model_path = model_path
            self.model_reference = str(model_path)
            self.model_cfg["selected_model_path"] = _to_relative_or_abs(model_path)
        else:
            self.current_model_path = None
            self.model_reference = str(entry.get("model_name", model_path.name))
        selected_model_name = str(entry.get("model_name", "")).strip()
        if selected_model_name.lower().endswith(".pt"):
            self.model_cfg["name"] = selected_model_name
        self.compile_fallback_applied = False

        self._apply_controls_to_runtime_state()
        self._rebuild_predict_kwargs()
        inferred_profile_key = self._infer_yolo_profile_key()
        self.runtime_cfg["yolo_profile"] = inferred_profile_key
        self._suppress_setting_autosave = True
        try:
            self._set_combo_to_data(self.yolo_profile_combo, inferred_profile_key)
            self._populate_yolo_model_combo()
            self._set_model_combo_value(
                {
                    "model_name": str(self.model_cfg.get("name", "")),
                    "selected_model_path": str(self.model_cfg.get("selected_model_path", "")).strip(),
                }
            )
        finally:
            self._suppress_setting_autosave = False
        self._update_current_model_label()
        self._persist_config(show_message=False)
        self._log(f"Model switched to: {model_path}")

        if was_running:
            self.start_live()

    def _download_selected_model(self) -> None:
        entry = self._selected_model_table_entry()
        if entry is None:
            QMessageBox.information(self, "Model", "Najpierw wybierz wiersz modelu.")
            return

        model_name = str(entry.get("model_name", entry.get("name", ""))).strip()
        model_path = Path(entry["path"]).resolve()
        if model_path.exists() and not entry.get("missing", False):
            QMessageBox.information(self, "Model", "Model jest juz pobrany.")
            return

        reply = QMessageBox.question(
            self,
            "Model",
            f"Pobrac model {model_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            target_cfg = dict(self._day_segmentation_model_cfg()) if self._entry_is_segmentation_model(entry) else dict(self.model_cfg)
            target_cfg["name"] = model_name or str(target_cfg.get("name", "")).strip()
            target_cfg["selected_model_path"] = ""
            model, reference = self._load_model_with_progress_dialog(
                target_cfg,
                dialog_title="Model segmentacji" if self._entry_is_segmentation_model(entry) else "Model",
                model_name=model_name,
            )
            del model
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model", f"Nie mozna pobrac modelu:\n{exc}")
            return

        if self._entry_is_segmentation_model(entry):
            self.model_cfg["day_segmentation"] = dict(target_cfg)
            resolved = Path(reference)
            if resolved.exists():
                self.model_cfg["day_segmentation"]["selected_model_path"] = _to_relative_or_abs(resolved.resolve())
            self._set_day_seg_model_combo_value(self.model_cfg["day_segmentation"])

        self._refresh_model_catalog()
        self._update_yolo_profile_summary()
        self._on_setting_changed()

    def _download_selected_day_seg_model(self) -> None:
        selection = self._current_day_seg_model_selection()
        model_name = str(selection.get("model_name", "")).strip() or DEFAULT_DAY_SEG_MODEL_NAME
        seg_cfg = self._day_segmentation_model_cfg()
        seg_cfg["name"] = model_name
        seg_cfg["selected_model_path"] = ""

        reply = QMessageBox.question(
            self,
            "Model segmentacji",
            f"Pobrac model segmentacji {model_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            model, reference = self._load_model_with_progress_dialog(
                seg_cfg,
                dialog_title="Model segmentacji",
                model_name=model_name,
            )
            del model
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model segmentacji", f"Nie mozna pobrac modelu segmentacji:\n{exc}")
            return

        self.model_cfg["day_segmentation"] = dict(seg_cfg)
        resolved = Path(reference)
        if resolved.exists():
            self.model_cfg["day_segmentation"]["selected_model_path"] = _to_relative_or_abs(resolved.resolve())
        self._refresh_model_catalog()
        self._populate_day_seg_model_combo()
        self._set_day_seg_model_combo_value(self.model_cfg["day_segmentation"])
        self._update_yolo_profile_summary()
        self._on_setting_changed()

    # ---------- sources ----------
    def _normalize_sources_entries(self, raw_sources: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_sources, list):
            return []

        normalized: list[dict[str, Any]] = []
        existing_names: set[str] = set()
        for item in raw_sources:
            if not isinstance(item, dict):
                continue

            source_type = str(item.get("type", "video")).strip().lower() or "video"
            raw_value = item.get("value")
            if source_type == "camera":
                try:
                    value: Any = int(raw_value)
                except Exception:  # noqa: BLE001
                    continue
            else:
                value = str(raw_value or "").strip()
                if not value:
                    continue

            name_hint = str(item.get("name", "")).strip() or f"{source_type}_source"
            source_name = _ensure_unique_name(existing_names, _safe_name(name_hint, f"{source_type}_source"))
            existing_names.add(source_name)
            normalized.append(
                {
                    "name": source_name,
                    "type": source_type,
                    "value": value,
                    "enabled": bool(item.get("enabled", True)),
                }
            )
            if bool(item.get("random_start", False)):
                normalized[-1]["random_start"] = True

            ignore_polys: list[list[tuple[float, float]]] = []
            raw_polys = item.get("ignore_polys")
            if isinstance(raw_polys, list):
                for poly in raw_polys:
                    if not isinstance(poly, list):
                        continue
                    points: list[tuple[float, float]] = []
                    for pt in poly:
                        if isinstance(pt, (list, tuple)) and len(pt) == 2:
                            try:
                                x = float(pt[0])
                                y = float(pt[1])
                            except Exception:  # noqa: BLE001
                                continue
                            points.append((_clamp(x, 0.0, 1.0), _clamp(y, 0.0, 1.0)))
                    if len(points) >= 3:
                        ignore_polys.append(points)

            if not ignore_polys:
                raw_poly = item.get("ignore_poly")
                if isinstance(raw_poly, list):
                    points = []
                    for pt in raw_poly:
                        if isinstance(pt, (list, tuple)) and len(pt) == 2:
                            try:
                                x = float(pt[0])
                                y = float(pt[1])
                            except Exception:  # noqa: BLE001
                                continue
                            points.append((_clamp(x, 0.0, 1.0), _clamp(y, 0.0, 1.0)))
                    if len(points) >= 3:
                        ignore_polys.append(points)

            if not ignore_polys:
                rect = item.get("ignore_rect")
                if isinstance(rect, (list, tuple)) and len(rect) == 4:
                    try:
                        x0, y0, x1, y1 = [float(v) for v in rect]
                        ignore_polys.append(
                            [
                                (_clamp(x0, 0.0, 1.0), _clamp(y0, 0.0, 1.0)),
                                (_clamp(x1, 0.0, 1.0), _clamp(y0, 0.0, 1.0)),
                                (_clamp(x1, 0.0, 1.0), _clamp(y1, 0.0, 1.0)),
                                (_clamp(x0, 0.0, 1.0), _clamp(y1, 0.0, 1.0)),
                            ]
                        )
                    except Exception:  # noqa: BLE001
                        ignore_polys = []

            if ignore_polys:
                normalized[-1]["ignore_polys"] = ignore_polys

        return normalized

    def _load_sources_config(self) -> list[dict[str, Any]]:
        loaded_sources: list[dict[str, Any]] = []
        if self.sources_settings_path.exists():
            try:
                settings_payload = load_yaml(self.sources_settings_path)
            except Exception:  # noqa: BLE001
                settings_payload = {}
            loaded_sources = self._normalize_sources_entries(settings_payload.get("sources", []))
            if loaded_sources:
                return loaded_sources

        legacy_sources = self._normalize_sources_entries(self.config.get("sources", []))
        if legacy_sources:
            self._save_sources_config(legacy_sources)
        return legacy_sources

    def _save_sources_config(self, sources: list[dict[str, Any]] | None = None) -> None:
        payload = {
            "version": 1,
            "sources": [dict(item) for item in (sources if sources is not None else self.sources)],
        }
        save_yaml(self.sources_settings_path, payload)

    def _rebuild_source_lookup(self) -> None:
        self.source_by_name = {
            str(source.get("name", "")): source
            for source in self.sources
            if str(source.get("name", ""))
        }

    def _load_app_config_overlay(self) -> None:
        if not self.app_config_path.exists():
            return
        try:
            overlay = load_yaml(self.app_config_path)
        except Exception:  # noqa: BLE001
            overlay = {}
        if not isinstance(overlay, dict):
            return
        for key in ("model", "inference", "tracker", "security", "uniform", "events", "runtime"):
            if key in overlay:
                self.config[key] = overlay.get(key)

    def _refresh_camera_list(self) -> None:
        max_index = int(self.runtime_cfg.get("scan_max_index", 8))
        self.camera_combo.clear()
        available = list(scan_available_cameras(max_index=max_index, preferred_backend=self.camera_backend))
        if not available:
            self.camera_combo.addItem("Brak dostepnych kamer", -1)
            model = self.camera_combo.model()
            item = model.item(0) if model is not None else None
            if item is not None:
                item.setEnabled(False)
            return

        existing = self._existing_source_values("camera")
        for camera_index in available:
            label = f"Kamera {camera_index}"
            already_added = str(camera_index) in existing
            if already_added:
                label = f"{label} (juz dodana)"
            self.camera_combo.addItem(label, camera_index)
            if already_added:
                row = self.camera_combo.count() - 1
                model = self.camera_combo.model()
                item = model.item(row) if model is not None else None
                if item is not None:
                    item.setEnabled(False)

    def _browse_video_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(resolve_path("data/videos")),
            "Video files (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;All files (*.*)",
        )
        if file_path:
            self.video_path_edit.setText(file_path)
            self._update_video_duplicate_hint()

    def _browse_events_output_dir(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select events output folder",
            str(self.events_output_dir),
        )
        if selected_dir:
            self.events_output_dir_edit.setText(selected_dir)

    def _existing_source_names(self) -> set[str]:
        return {str(source.get("name", "")) for source in self.sources}

    def _existing_source_values(self, source_type: str) -> set[str]:
        values: set[str] = set()
        for source in self.sources:
            if str(source.get("type", "")) == source_type:
                values.add(self._normalize_source_value(source_type, source.get("value", "")))
        return values

    def _normalize_source_value(self, source_type: str, value: Any) -> str:
        raw = str(value or "")
        if source_type == "video":
            try:
                return os.path.normcase(str(Path(raw).resolve()))
            except Exception:  # noqa: BLE001
                return os.path.normcase(raw)
        if source_type == "stream":
            return raw.strip()
        if source_type == "camera":
            return str(raw).strip()
        return raw

    def _add_source(
        self,
        source_type: str,
        value: Any,
        name_hint: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        source_name = _ensure_unique_name(self._existing_source_names(), _safe_name(name_hint, f"{source_type}_source"))
        source = {
            "name": source_name,
            "type": source_type,
            "value": value,
            "enabled": True,
        }
        if extra:
            source.update(extra)
        self.sources.append(source)
        self._rebuild_source_lookup()
        self._sync_runtimes_with_sources()
        self._rebuild_source_table()
        self._rebuild_live_layout()
        self._persist_config(show_message=False)
        self._log(f"Source added: {source_name} ({source_type})")

    def _add_camera_source(self) -> None:
        if self.camera_combo.count() <= 0:
            QMessageBox.warning(self, "Camera", "No camera detected.")
            return

        camera_data = self.camera_combo.currentData()
        if camera_data is None or int(camera_data) < 0:
            QMessageBox.information(self, "Camera", "Use Refresh first to scan available cameras.")
            return

        camera_index = int(camera_data)
        if self._normalize_source_value("camera", camera_index) in self._existing_source_values("camera"):
            QMessageBox.information(self, "Camera", "Ta kamera jest juz dodana.")
            return
        base_name = self.camera_name_edit.text().strip() or f"camera_{camera_index}"
        self._add_source("camera", camera_index, base_name)
        self.camera_name_edit.clear()

    def _add_video_source(self) -> None:
        raw_path = self.video_path_edit.text().strip()
        if not raw_path:
            QMessageBox.warning(self, "Video", "Provide video path first.")
            return

        video_path = resolve_path(raw_path)
        if not video_path.exists():
            QMessageBox.warning(self, "Video", f"File does not exist:\n{video_path}")
            return

        duplicate = self._normalize_source_value("video", video_path) in self._existing_source_values("video")
        if duplicate:
            reply = QMessageBox.question(
                self,
                "Video",
                "To zrodlo wideo jest juz dodane. Czy na pewno chcesz dodac duplikat?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        base_name = self.video_name_edit.text().strip() or video_path.stem
        extra: dict[str, Any] = {}
        if hasattr(self, "video_random_start_checkbox") and self.video_random_start_checkbox.isChecked():
            extra["random_start"] = True
        self._add_source("video", str(video_path), base_name, extra=extra or None)
        self.video_name_edit.clear()
        if hasattr(self, "video_random_start_checkbox"):
            self.video_random_start_checkbox.setChecked(False)
        self._update_video_duplicate_hint()

    def _add_stream_source(self) -> None:
        stream_url = self.stream_url_edit.text().strip()
        if not stream_url:
            QMessageBox.warning(self, "Stream", "Provide stream URL first.")
            return

        if self._normalize_source_value("stream", stream_url) in self._existing_source_values("stream"):
            QMessageBox.information(self, "Stream", "To zrodlo strumienia jest juz dodane.")
            return

        base_name = self.stream_name_edit.text().strip() or "stream_source"
        self._add_source("stream", stream_url, base_name)
        self.stream_name_edit.clear()

    def _get_source_by_name(self, source_name: str) -> dict[str, Any] | None:
        return self.source_by_name.get(source_name)

    def _source_ignore_signature(self, source: dict[str, Any] | None) -> str:
        if source is None:
            return ""
        payload = {
            "ignore_rect": source.get("ignore_rect"),
            "ignore_poly": source.get("ignore_poly"),
            "ignore_polys": source.get("ignore_polys"),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    def _get_or_build_ignore_mask(
        self,
        source_name: str,
        source: dict[str, Any] | None,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        if source is None:
            return None

        h, w = frame.shape[:2]
        signature = self._source_ignore_signature(source)
        cached = self._ignore_mask_cache.get(source_name)
        if cached is not None:
            cached_size, cached_signature, cached_mask = cached
            if cached_size == (h, w) and cached_signature == signature:
                return cached_mask

        rect = source.get("ignore_rect")
        polys = source.get("ignore_polys")
        if not polys:
            legacy_poly = source.get("ignore_poly")
            if legacy_poly:
                polys = [legacy_poly]

        if rect is None and not polys:
            self._ignore_mask_cache.pop(source_name, None)
            return None

        mask = np.full((h, w), 255, dtype=np.uint8)
        if rect is not None:
            try:
                x0, y0, x1, y1 = rect
                left = int(_clamp(float(x0), 0.0, 1.0) * w)
                right = int(_clamp(float(x1), 0.0, 1.0) * w)
                top = int(_clamp(float(y0), 0.0, 1.0) * h)
                bottom = int(_clamp(float(y1), 0.0, 1.0) * h)
                if right > left and bottom > top:
                    mask[top:bottom, left:right] = 0
            except Exception:  # noqa: BLE001
                pass

        if polys:
            for poly in polys:
                if not isinstance(poly, (list, tuple)) or len(poly) < 3:
                    continue
                pts: list[list[int]] = []
                for point in poly:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        continue
                    try:
                        x = int(_clamp(float(point[0]), 0.0, 1.0) * w)
                        y = int(_clamp(float(point[1]), 0.0, 1.0) * h)
                    except Exception:  # noqa: BLE001
                        continue
                    pts.append([x, y])
                if len(pts) < 3:
                    continue
                contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [contour], 0)

        self._ignore_mask_cache[source_name] = ((h, w), signature, mask)
        return mask

    def _build_fullscreen_display_frame(self, source_name: str, frame: np.ndarray | None) -> np.ndarray | None:
        if frame is None:
            return None

        runtime = self.runtimes.get(source_name)
        source = runtime.source if runtime is not None else self._get_source_by_name(source_name)
        if source is None:
            return frame
        if not bool(source.get("ignore_polys") or source.get("ignore_poly") or source.get("ignore_rect")):
            return frame

        mask = self._get_or_build_ignore_mask(source_name, source, frame)
        if mask is None:
            return frame

        ignored = mask == 0
        if not np.any(ignored):
            return frame

        output = frame.copy()
        overlay = output.copy()
        overlay[ignored] = (70, 95, 220)
        cv2.addWeighted(overlay, 0.30, output, 0.70, 0.0, dst=output)
        return output

    def _update_video_duplicate_hint(self) -> None:
        if not hasattr(self, "video_duplicate_label"):
            return
        raw_path = self.video_path_edit.text().strip() if hasattr(self, "video_path_edit") else ""
        if not raw_path:
            self.video_duplicate_label.setVisible(False)
            self.video_duplicate_label.setText("")
            return
        video_path = resolve_path(raw_path)
        duplicate = self._normalize_source_value("video", video_path) in self._existing_source_values("video")
        if duplicate:
            self.video_duplicate_label.setText("To wideo jest juz dodane.")
            self.video_duplicate_label.setVisible(True)
        else:
            self.video_duplicate_label.setVisible(False)
            self.video_duplicate_label.setText("")

    def _open_source_mask_editor(self) -> None:
        row = self.source_table.currentRow()
        if row < 0 or row >= len(self.sources):
            QMessageBox.information(self, "Mask", "Wybierz zrodlo w tabeli.")
            return

        source = self.sources[row]
        source_name = str(source.get("name", "source"))
        runtime = self.runtimes.get(source_name)
        if runtime is None or runtime.last_input is None:
            QMessageBox.information(self, "Mask", "Brak klatki do edycji. Uruchom podglad zrodla.")
            return

        poly_norm = source.get("ignore_polys")
        if not poly_norm:
            legacy_poly = source.get("ignore_poly")
            if legacy_poly:
                poly_norm = [legacy_poly]
        rect = source.get("ignore_rect")
        if rect and not poly_norm:
            x0, y0, x1, y1 = rect
            poly_norm = [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]

        dialog = MaskEditorDialog(runtime.last_input, poly_norm, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_poly = dialog.get_rect()
        if not new_poly:
            source.pop("ignore_polys", None)
        else:
            source["ignore_polys"] = new_poly
        source.pop("ignore_poly", None)
        source.pop("ignore_rect", None)

        # Keep runtime source in sync so live inference uses the new mask immediately.
        runtime.source = dict(source)
        self._ignore_mask_cache.pop(source_name, None)
        self._rebuild_source_lookup()

        if self.live_running and runtime.last_input is not None:
            self._enqueue_inference_frame(source_name, runtime.last_input, time.perf_counter())

        self._persist_config(show_message=False)
        self._rebuild_source_table()
        self._refresh_tile(source_name)

    def _remove_selected_source(self) -> None:
        row = self.source_table.currentRow()
        if row < 0 or row >= len(self.sources):
            return

        source = self.sources.pop(row)
        source_name = str(source.get("name", ""))

        runtime = self.runtimes.pop(source_name, None)
        if runtime is not None:
            self._finalize_event_clip(source_name, runtime, time.perf_counter())
            runtime.release()
        self._clear_async_state_for_source(source_name)
        self._ignore_mask_cache.pop(source_name, None)
        self._rebuild_source_lookup()
        with self._infer_lock:
            self.trackers.pop(source_name, None)

        self.tiles.pop(source_name, None)
        self.zoom_levels.pop(source_name, None)
        self.pan_offsets.pop(source_name, None)

        if self.focused_source == source_name:
            self._close_fullscreen_source()
            self.focused_source = None

        self._rebuild_source_table()
        self._rebuild_live_layout()
        self._persist_config(show_message=False)
        self._log(f"Source removed: {source_name}")

    def _sync_runtimes_with_sources(self) -> None:
        self._rebuild_source_lookup()
        source_names = {str(source.get("name", "")) for source in self.sources}

        for name in list(self.runtimes):
            if name in source_names:
                continue
            self._finalize_event_clip(name, self.runtimes[name], time.perf_counter())
            self.runtimes[name].release()
            del self.runtimes[name]
            self._clear_async_state_for_source(name)
            self._ignore_mask_cache.pop(name, None)
            with self._infer_lock:
                self.trackers.pop(name, None)

        for source in self.sources:
            source_name = str(source.get("name", ""))
            runtime = self.runtimes.get(source_name)
            if runtime is None:
                runtime = SourceRuntime(source=dict(source))
                self.runtimes[source_name] = runtime
                self._reset_tracker_for_source(source_name, runtime)
            else:
                runtime.source = dict(source)

            if source_name not in self.zoom_levels:
                self.zoom_levels[source_name] = 1.0
            if source_name not in self.pan_offsets:
                self.pan_offsets[source_name] = (0.0, 0.0)

    def _rebuild_source_table(self) -> None:
        self._table_updating = True
        try:
            self.source_table.setRowCount(len(self.sources))
            for row, source in enumerate(self.sources):
                name = str(source.get("name", "source"))
                source_type = str(source.get("type", "video"))
                value = str(source.get("value", ""))
                enabled = bool(source.get("enabled", True))
                masks = source.get("ignore_polys") or source.get("ignore_poly") or source.get("ignore_rect")
                mask_count = 0
                if isinstance(masks, list):
                    if masks and isinstance(masks[0], (list, tuple)) and len(masks) > 0:
                        first = masks[0]
                        if first and isinstance(first, (list, tuple)) and len(first) == 2:
                            mask_count = 1
                        else:
                            mask_count = len(masks)
                elif masks:
                    mask_count = 1

                self.source_table.setItem(row, 0, QTableWidgetItem(name))
                self.source_table.setItem(row, 1, QTableWidgetItem(source_type))
                self.source_table.setItem(row, 2, QTableWidgetItem(value))
                mask_label = "tak" if mask_count > 0 else "nie"
                if mask_count > 1:
                    mask_label = f"tak ({mask_count})"
                self.source_table.setItem(row, 3, QTableWidgetItem(mask_label))

                random_enabled = bool(source.get("random_start", False)) if source_type == "video" else False
                if source_type == "video":
                    random_checkbox = QPushButton("✓" if random_enabled else "")
                    random_checkbox.setCheckable(True)
                    random_checkbox.setChecked(random_enabled)
                    random_checkbox.setFixedSize(22, 22)
                    random_checkbox.setStyleSheet(
                        "QPushButton {"
                        "background: #1a202b;"
                        "border: 1px solid #3b4657;"
                        "border-radius: 4px;"
                        "color: #ffffff;"
                        "padding: 0px;"
                        "font-size: 14px;"
                        "font-weight: 800;"
                        "}"
                        "QPushButton:checked { background: #2f81f7; border: 1px solid #2363c0; }"
                    )
                    random_checkbox.toggled.connect(lambda checked, b=random_checkbox: b.setText("✓" if checked else ""))
                    random_checkbox.toggled.connect(lambda checked, r=row: self._on_source_random_toggled(r, checked))
                    random_cell = QWidget()
                    random_layout = QHBoxLayout(random_cell)
                    random_layout.setContentsMargins(0, 0, 0, 0)
                    random_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    random_layout.addWidget(random_checkbox)
                    self.source_table.setCellWidget(row, 4, random_cell)
                else:
                    item = QTableWidgetItem("-")
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.source_table.setItem(row, 4, item)

                enabled_checkbox = QPushButton("✓" if enabled else "")
                enabled_checkbox.setCheckable(True)
                enabled_checkbox.setChecked(enabled)
                enabled_checkbox.setFixedSize(22, 22)
                enabled_checkbox.setStyleSheet(
                    "QPushButton {"
                    "background: #1a202b;"
                    "border: 1px solid #3b4657;"
                    "border-radius: 4px;"
                    "color: #ffffff;"
                    "padding: 0px;"
                    "font-size: 14px;"
                    "font-weight: 800;"
                    "}"
                    "QPushButton:checked { background: #2aa96b; border: 1px solid #1f8a55; }"
                )
                enabled_checkbox.toggled.connect(lambda checked, b=enabled_checkbox: b.setText("✓" if checked else ""))
                enabled_checkbox.toggled.connect(lambda checked, r=row: self._on_source_enabled_toggled(r, checked))
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell_layout.addWidget(enabled_checkbox)
                self.source_table.setCellWidget(row, 5, cell)
        finally:
            self._table_updating = False
            self._update_source_table_height()

    def _update_source_table_height(self) -> None:
        if not hasattr(self, "source_table") or self.source_table is None:
            return

        row_count = int(self.source_table.rowCount())
        min_rows = max(1, int(self._source_table_min_visible_rows))
        max_rows = max(min_rows, int(self._source_table_max_visible_rows))
        visible_rows = min(max(row_count, min_rows), max_rows)

        header_h = self.source_table.horizontalHeader().height()
        row_h = self.source_table.verticalHeader().defaultSectionSize()
        frame_h = self.source_table.frameWidth() * 2
        target_height = int(header_h + frame_h + (visible_rows * row_h) + 4)

        self.source_table.setFixedHeight(target_height)

    def _on_source_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_updating:
            return

        row = item.row()
        if row < 0 or row >= len(self.sources):
            return

        if item.column() != 3:
            return

        enabled = item.checkState() == Qt.CheckState.Checked
        item.setText("yes" if enabled else "no")
        self.sources[row]["enabled"] = enabled

        source_name = str(self.sources[row].get("name", ""))
        if not enabled:
            runtime = self.runtimes.get(source_name)
            if runtime is not None:
                self._finalize_event_clip(source_name, runtime, time.perf_counter())
                runtime.release()
            self._clear_async_state_for_source(source_name)
            with self._infer_lock:
                self.trackers.pop(source_name, None)
            if self.focused_source == source_name:
                self._close_fullscreen_source()
                self.focused_source = None

        self._rebuild_live_layout()
        self._persist_config(show_message=False)

    def _on_source_enabled_toggled(self, row: int, enabled: bool) -> None:
        if self._table_updating:
            return
        if row < 0 or row >= len(self.sources):
            return
        self.sources[row]["enabled"] = enabled
        self._rebuild_source_lookup()

        source_name = str(self.sources[row].get("name", ""))
        if not enabled:
            runtime = self.runtimes.get(source_name)
            if runtime is not None:
                self._finalize_event_clip(source_name, runtime, time.perf_counter())
                runtime.release()
            self._clear_async_state_for_source(source_name)
            with self._infer_lock:
                self.trackers.pop(source_name, None)
            if self.focused_source == source_name:
                self._close_fullscreen_source()
                self.focused_source = None

        self._rebuild_live_layout()
        self._persist_config(show_message=False)

    def _on_source_random_toggled(self, row: int, enabled: bool) -> None:
        if self._table_updating:
            return
        if row < 0 or row >= len(self.sources):
            return
        if str(self.sources[row].get("type", "")) != "video":
            return
        self.sources[row]["random_start"] = enabled
        self._rebuild_source_lookup()

        source_name = str(self.sources[row].get("name", ""))
        runtime = self.runtimes.get(source_name)
        if runtime is not None:
            runtime.release()
            runtime.capture = None
        self._persist_config(show_message=False)

    # ---------- live layout ----------
    def _get_enabled_sources(self) -> list[dict[str, Any]]:
        return [source for source in self.sources if bool(source.get("enabled", True))]

    def _ensure_tile(self, source_name: str) -> VideoTile:
        tile = self.tiles.get(source_name)
        if tile is not None:
            tile.set_header_visibility(self.live_tile_header_visible)
            return tile

        tile = VideoTile(source_name)
        tile.clicked.connect(self._on_tile_clicked)
        tile.right_clicked.connect(self._on_tile_right_clicked)
        tile.zoom_delta.connect(self._on_tile_zoom_delta)
        tile.pan_delta.connect(self._on_tile_pan_delta)
        tile.set_header_visibility(self.live_tile_header_visible)
        self.tiles[source_name] = tile
        return tile

    def _clear_live_grid(self) -> None:
        while self.live_grid_layout.count() > 0:
            item = self.live_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _auto_grid_dimensions(self, count: int) -> tuple[int, int]:
        if count <= 1:
            return 1, 1

        viewport = self.live_scroll.viewport().size()
        width = max(1, viewport.width())
        height = max(1, viewport.height())
        spacing_x = max(0, int(self.live_grid_layout.horizontalSpacing()))
        spacing_y = max(0, int(self.live_grid_layout.verticalSpacing()))
        target_aspect = 16.0 / 9.0

        best_cols = 1
        best_rows = count
        best_score = -1.0
        best_empty = count

        for cols in range(1, count + 1):
            rows = int(math.ceil(count / cols))
            tile_w = (width - max(0, cols - 1) * spacing_x) / float(cols)
            tile_h = (height - max(0, rows - 1) * spacing_y) / float(rows)
            if tile_w <= 1 or tile_h <= 1:
                continue

            fit_w = min(tile_w, tile_h * target_aspect)
            fit_h = min(tile_h, tile_w / target_aspect)
            visible_area = max(1.0, fit_w * fit_h)

            empty_slots = max(0, cols * rows - count)
            empty_penalty = 0.08 * empty_slots * visible_area
            score = visible_area - empty_penalty

            if (
                score > best_score + 1e-6
                or (abs(score - best_score) <= 1e-6 and empty_slots < best_empty)
                or (
                    abs(score - best_score) <= 1e-6
                    and empty_slots == best_empty
                    and rows < best_rows
                )
            ):
                best_cols = cols
                best_rows = rows
                best_score = score
                best_empty = empty_slots

        return best_cols, best_rows

    def _apply_grid_tile_sizes(self, names_to_show: list[str], columns: int, rows: int) -> None:
        viewport = self.live_scroll.viewport().size()
        margins = self.live_grid_layout.contentsMargins()
        spacing_x = max(0, int(self.live_grid_layout.horizontalSpacing()))
        spacing_y = max(0, int(self.live_grid_layout.verticalSpacing()))

        avail_w = max(1, viewport.width() - margins.left() - margins.right())
        avail_h = max(1, viewport.height() - margins.top() - margins.bottom())

        cell_w = max(1, int((avail_w - max(0, columns - 1) * spacing_x) / max(1, columns)))
        cell_h = max(1, int((avail_h - max(0, rows - 1) * spacing_y) / max(1, rows)))

        used_w = (cell_w * columns) + (max(0, columns - 1) * spacing_x)
        used_h = (cell_h * rows) + (max(0, rows - 1) * spacing_y)
        target_w = max(viewport.width(), used_w + margins.left() + margins.right())
        target_h = max(viewport.height(), used_h + margins.top() + margins.bottom())

        self.live_grid_widget.setMinimumSize(target_w, target_h)
        self.live_grid_widget.resize(target_w, target_h)

        for source_name in names_to_show:
            tile = self.tiles.get(source_name)
            if tile is None:
                continue
            tile.setMinimumSize(cell_w, cell_h)
            tile.setMaximumSize(cell_w, cell_h)

    def _rebuild_live_layout(self) -> None:
        enabled_sources = self._get_enabled_sources()
        enabled_names = [str(source.get("name", "")) for source in enabled_sources]

        if self.focused_source and self.focused_source not in enabled_names:
            self._close_fullscreen_source()
            self.focused_source = None

        self._clear_live_grid()

        if not enabled_names:
            self.live_scroll.hide()
            self.live_placeholder.show()
            self._position_overlay_controls()
            return

        self.live_placeholder.hide()
        self.live_scroll.show()

        names_to_show = enabled_names
        self.live_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.live_grid_layout.setHorizontalSpacing(self.live_tile_spacing)
        self.live_grid_layout.setVerticalSpacing(self.live_tile_spacing)
        configured_columns = self.runtime_cfg.get("grid_columns", "auto")
        if isinstance(configured_columns, int):
            columns = max(1, min(len(names_to_show), configured_columns))
            rows = max(1, int(math.ceil(len(names_to_show) / columns)))
        else:
            text = str(configured_columns).strip().lower()
            if text in {"", "auto", "0"}:
                columns, rows = self._auto_grid_dimensions(len(names_to_show))
            else:
                try:
                    columns = max(1, min(len(names_to_show), int(text)))
                    rows = max(1, int(math.ceil(len(names_to_show) / columns)))
                except ValueError:
                    columns, rows = self._auto_grid_dimensions(len(names_to_show))

        for index, source_name in enumerate(names_to_show):
            row = index // columns
            column = index % columns
            tile = self._ensure_tile(source_name)
            tile.set_focus_state(False)
            self.live_grid_layout.addWidget(tile, row, column)

        for col in range(columns):
            self.live_grid_layout.setColumnStretch(col, 1)
        for row in range(rows):
            self.live_grid_layout.setRowStretch(row, 1)
        self._apply_grid_tile_sizes(names_to_show, columns, rows)

        self._position_overlay_controls()

    def _on_tile_clicked(self, source_name: str) -> None:
        if self.focused_source == source_name and self._is_fullscreen_visible():
            self._switch_to_grid_view()
            return
        self.focused_source = source_name
        self._open_fullscreen_source(source_name)

    def _on_tile_right_clicked(self, source_name: str) -> None:
        if self.focused_source == source_name and self._is_fullscreen_visible():
            self._switch_to_grid_view()

    def _switch_to_grid_view(self) -> None:
        previous_source = self.focused_source
        self._close_fullscreen_source()
        if previous_source:
            self.zoom_levels[previous_source] = 1.0
            self.pan_offsets[previous_source] = (0.0, 0.0)
        self.focused_source = None
        self._rebuild_live_layout()

    def _ensure_fullscreen_window(self) -> FullscreenVideoWindow:
        if self.fullscreen_window is None:
            self.fullscreen_window = FullscreenVideoWindow()
            self.fullscreen_window.request_close.connect(self._switch_to_grid_view)
            self.fullscreen_window.zoom_delta.connect(
                lambda delta: self._change_focus_zoom(1 if delta > 0 else -1)
            )
            self.fullscreen_window.pan_delta.connect(self._on_fullscreen_pan_delta)
        return self.fullscreen_window

    def _is_fullscreen_visible(self) -> bool:
        return bool(self.fullscreen_window is not None and self.fullscreen_window.isVisible())

    def _open_fullscreen_source(self, source_name: str) -> None:
        window = self._ensure_fullscreen_window()
        window.set_source_name(source_name)
        runtime = self.runtimes.get(source_name)
        if runtime is not None:
            zoom = self.zoom_levels.get(source_name, 1.0)
            pan_x, pan_y = self.pan_offsets.get(source_name, (0.0, 0.0))
            fullscreen_frame = self._build_fullscreen_display_frame(source_name, runtime.last_output)
            window.set_frame(fullscreen_frame, zoom=zoom, pan_x=pan_x, pan_y=pan_y)
        window.showFullScreen()
        window.raise_()
        window.activateWindow()
        self._position_overlay_controls()

    def _close_fullscreen_source(self) -> None:
        if self.fullscreen_window is not None and self.fullscreen_window.isVisible():
            self.fullscreen_window.hide()

    def _on_fullscreen_pan_delta(self, dx: float, dy: float) -> None:
        if not self.focused_source:
            return

        zoom = self.zoom_levels.get(self.focused_source, 1.0)
        if zoom <= 1.01:
            return

        if self.fullscreen_window is not None and self.fullscreen_window.isVisible():
            width = max(1, self.fullscreen_window.canvas.width())
            height = max(1, self.fullscreen_window.canvas.height())
        else:
            tile = self.tiles.get(self.focused_source)
            if tile is None:
                return
            width = max(1, tile.canvas.width())
            height = max(1, tile.canvas.height())

        pan_x, pan_y = self.pan_offsets.get(self.focused_source, (0.0, 0.0))
        pan_x -= (dx / float(width)) * (2.0 / zoom)
        pan_y -= (dy / float(height)) * (2.0 / zoom)

        self.pan_offsets[self.focused_source] = (
            _clamp(pan_x, -1.0, 1.0),
            _clamp(pan_y, -1.0, 1.0),
        )
        self._refresh_tile(self.focused_source)

    def _update_live_header_toggle_button(self) -> None:
        if not hasattr(self, "live_header_toggle_btn") or self.live_header_toggle_btn is None:
            return
        if self.live_tile_header_visible:
            self.live_header_toggle_btn.setText("◂")
            self.live_header_toggle_btn.setToolTip("Ukryj gorny pasek informacji nad kafelkami kamer.")
        else:
            self.live_header_toggle_btn.setText("▸")
            self.live_header_toggle_btn.setToolTip("Pokaz gorny pasek informacji nad kafelkami kamer.")

    def _update_navigation_toggle_button(self) -> None:
        if not hasattr(self, "navigation_toggle_btn") or self.navigation_toggle_btn is None:
            return
        if self.navigation_tabs_visible:
            text = "▴"
            tooltip = "Ukryj gorne zakladki aplikacji oraz podzakladki Live/Nagrania."
        else:
            text = "▾"
            tooltip = "Pokaz gorne zakladki aplikacji oraz podzakladki Live/Nagrania."
        self.navigation_toggle_btn.setText(text)
        self.navigation_toggle_btn.setToolTip(tooltip)
        if hasattr(self, "navigation_overlay_btn") and self.navigation_overlay_btn is not None:
            self.navigation_overlay_btn.setText(text)
            self.navigation_overlay_btn.setToolTip(tooltip)

    def _set_navigation_tabs_visibility(self, visible: bool, *, persist: bool) -> None:
        visible = bool(visible)
        self.navigation_tabs_visible = visible
        self.main_tabs.tabBar().setVisible(visible)
        if hasattr(self, "preview_tabs") and self.preview_tabs is not None:
            self.preview_tabs.tabBar().setVisible(visible)
        if hasattr(self, "navigation_corner_widget") and self.navigation_corner_widget is not None:
            self.navigation_corner_widget.setVisible(visible)
        if hasattr(self, "navigation_overlay_btn") and self.navigation_overlay_btn is not None:
            self.navigation_overlay_btn.setVisible(not visible)
        if hasattr(self, "exit_app_btn") and self.exit_app_btn is not None:
            self.exit_app_btn.setVisible(visible)
        self._update_navigation_toggle_button()
        self._position_overlay_controls()

        if persist:
            self._persist_config(show_message=False)

    def _toggle_navigation_tabs_visibility(self) -> None:
        self._set_navigation_tabs_visibility(not self.navigation_tabs_visible, persist=True)

    def _set_live_tile_header_visibility(self, visible: bool, *, persist: bool) -> None:
        visible = bool(visible)
        if self.live_tile_header_visible == visible:
            self._update_live_header_toggle_button()
            return

        self.live_tile_header_visible = visible
        for tile in self.tiles.values():
            tile.set_header_visibility(visible)
        self._update_live_header_toggle_button()
        self._position_overlay_controls()

        if self.focused_source:
            self._refresh_tile(self.focused_source)

        if persist:
            self._persist_config(show_message=False)

    def _toggle_live_tile_header_visibility(self) -> None:
        self._set_live_tile_header_visibility(not self.live_tile_header_visible, persist=True)

    def _on_preview_subtab_changed(self, index: int) -> None:
        is_live = int(index) == 0
        if hasattr(self, "live_header_toggle_btn") and self.live_header_toggle_btn is not None:
            self.live_header_toggle_btn.setVisible(is_live)
        self._position_overlay_controls()

    def _update_live_overlay_margin(self) -> None:
        if not hasattr(self, "live_view_layout"):
            return
        self.live_view_layout.setContentsMargins(0, 0, 0, 0)

    def _position_overlay_controls(self) -> None:
        side_margin = 0
        top_margin = 0

        if hasattr(self, "exit_app_btn") and self.exit_app_btn is not None:
            parent = self.exit_app_btn.parentWidget()
            if parent is not None:
                x = max(side_margin, parent.width() - self.exit_app_btn.width() - side_margin)
                y = top_margin
                self.exit_app_btn.move(x, y)
                self.exit_app_btn.raise_()

        if (
            hasattr(self, "navigation_overlay_btn")
            and self.navigation_overlay_btn is not None
            and self.navigation_overlay_btn.isVisible()
        ):
            self.navigation_overlay_btn.move(0, 2)
            self.navigation_overlay_btn.raise_()

        if hasattr(self, "preview_tabs") and self.preview_tabs is not None:
            container = self.preview_tabs
            if container.currentIndex() != 0:
                return

            tab_bar = container.tabBar()
            controls_y = side_margin
            if tab_bar.isVisible():
                controls_y = max(side_margin, tab_bar.geometry().bottom() + 8)
            elif hasattr(self, "navigation_overlay_btn") and self.navigation_overlay_btn is not None:
                controls_y = max(
                    controls_y,
                    self.navigation_overlay_btn.y() + self.navigation_overlay_btn.height() + 8,
                )

            toggle_y = controls_y

            if hasattr(self, "live_header_toggle_btn") and self.live_header_toggle_btn is not None:
                header_toggle_x = side_margin
                header_toggle_y = toggle_y
                self.live_header_toggle_btn.move(header_toggle_x, header_toggle_y)
                self.live_header_toggle_btn.raise_()

    def _on_tile_zoom_delta(self, source_name: str, delta: int) -> None:
        if self.focused_source != source_name:
            return
        self._change_focus_zoom(1 if delta > 0 else -1)

    def _change_focus_zoom(self, direction: int) -> None:
        if not self.focused_source:
            return

        current = self.zoom_levels.get(self.focused_source, 1.0)
        scale = 1.12 if direction > 0 else (1.0 / 1.12)
        updated = _clamp(current * scale, 1.0, 8.0)
        self.zoom_levels[self.focused_source] = updated

        if updated <= 1.01:
            self.pan_offsets[self.focused_source] = (0.0, 0.0)

        self._refresh_tile(self.focused_source)

    def _reset_focus_zoom(self) -> None:
        if not self.focused_source:
            return
        self.zoom_levels[self.focused_source] = 1.0
        self.pan_offsets[self.focused_source] = (0.0, 0.0)
        self._refresh_tile(self.focused_source)

    def _on_tile_pan_delta(self, source_name: str, dx: float, dy: float) -> None:
        if self.focused_source != source_name:
            return

        zoom = self.zoom_levels.get(source_name, 1.0)
        if zoom <= 1.01:
            return

        tile = self.tiles.get(source_name)
        if tile is None:
            return

        width = max(1, tile.canvas.width())
        height = max(1, tile.canvas.height())

        pan_x, pan_y = self.pan_offsets.get(source_name, (0.0, 0.0))
        pan_x -= (dx / float(width)) * (2.0 / zoom)
        pan_y -= (dy / float(height)) * (2.0 / zoom)

        self.pan_offsets[source_name] = (
            _clamp(pan_x, -1.0, 1.0),
            _clamp(pan_y, -1.0, 1.0),
        )
        self._refresh_tile(source_name)

    def _queue_async_notice(self, message: str) -> None:
        with self._infer_lock:
            self._infer_notices.append(str(message))

    def _drain_async_notices(self) -> None:
        with self._infer_lock:
            if not self._infer_notices:
                return
            pending = list(self._infer_notices)
            self._infer_notices.clear()
        for message in pending:
            self._log(message)

    def _clear_async_state_for_source(self, source_name: str) -> None:
        with self._infer_lock:
            self._infer_pending_frames.pop(source_name, None)
            self._infer_results.pop(source_name, None)
            self._infer_last_submit_ts.pop(source_name, None)
            self._uniform_track_memory.pop(source_name, None)

    def _enqueue_inference_frame(self, source_name: str, frame: np.ndarray, submit_ts: float) -> None:
        frame_to_infer = frame
        runtime = self.runtimes.get(source_name)
        source = runtime.source if runtime is not None else self._get_source_by_name(source_name)
        mask = self._get_or_build_ignore_mask(source_name, source, frame)
        if mask is not None:
            frame_to_infer = cv2.bitwise_and(frame, frame, mask=mask)
        with self._infer_lock:
            # Keep only the newest frame for each source (drop stale work).
            self._infer_pending_frames[source_name] = frame_to_infer
            self._infer_last_submit_ts[source_name] = float(submit_ts)
            self._infer_has_work_event.set()

    def _pull_inference_batch(
        self,
        *,
        max_items: int | None = None,
        exclude_sources: set[str] | None = None,
    ) -> tuple[list[str], list[np.ndarray]]:
        with self._infer_lock:
            excluded = exclude_sources or set()
            source_names = list(self._infer_pending_frames.keys())
            if not source_names:
                self._infer_has_work_event.clear()
                return [], []

            total = len(source_names)
            start = self._infer_worker_rr_cursor % total
            ordered = [source_names[(start + idx) % total] for idx in range(total)]
            available = [source_name for source_name in ordered if source_name not in excluded]
            limit = self._effective_max_infer_per_tick() if max_items is None else max(1, int(max_items))
            chosen = available[:limit]
            if not chosen:
                if self._infer_pending_frames:
                    self._infer_has_work_event.set()
                else:
                    self._infer_has_work_event.clear()
                return [], []
            self._infer_worker_rr_cursor = (start + len(chosen)) % max(1, total)

            batch_sources: list[str] = []
            batch_frames: list[np.ndarray] = []
            for source_name in chosen:
                frame = self._infer_pending_frames.pop(source_name, None)
                if frame is None:
                    continue
                batch_sources.append(source_name)
                batch_frames.append(frame)
            if self._infer_pending_frames:
                self._infer_has_work_event.set()
            else:
                self._infer_has_work_event.clear()
        return batch_sources, batch_frames

    def _coalesce_inference_batch(
        self,
        source_batch: list[str],
        frame_batch: list[np.ndarray],
        *,
        mode: str,
    ) -> tuple[list[str], list[np.ndarray]]:
        if not source_batch:
            return source_batch, frame_batch

        coalesce_ms = self._infer_batch_coalesce_ms
        if mode == "night":
            infer_interval_ms = 1000.0 / max(1.0, self._effective_model_target_fps())
            night_floor_ms = min(12.0, max(6.0, infer_interval_ms * 0.20))
            coalesce_ms = max(coalesce_ms, night_floor_ms)
            if len(source_batch) == 1:
                coalesce_ms = min(14.0, coalesce_ms + 2.0)
        target_size = self._effective_max_infer_per_tick()
        if coalesce_ms <= 0.0 or len(source_batch) >= target_size:
            return source_batch, frame_batch

        # Short coalescing window allows neighboring sources to join one GPU pass.
        deadline = time.perf_counter() + (coalesce_ms / 1000.0)
        selected_sources = set(source_batch)
        while len(source_batch) < target_size:
            now_ts = time.perf_counter()
            if now_ts >= deadline:
                break
            wait_sec = max(0.0005, min(0.0015, deadline - now_ts))
            self._infer_has_work_event.wait(wait_sec)

            remaining = target_size - len(source_batch)
            if remaining <= 0:
                break
            extra_sources, extra_frames = self._pull_inference_batch(
                max_items=remaining,
                exclude_sources=selected_sources,
            )
            if not extra_sources:
                continue

            source_batch.extend(extra_sources)
            frame_batch.extend(extra_frames)
            selected_sources.update(extra_sources)

        return source_batch, frame_batch

    def _start_inference_worker(self) -> None:
        if self._infer_thread is not None and self._infer_thread.is_alive():
            return

        with self._infer_lock:
            self._infer_stop_event.clear()
            self._infer_has_work_event.clear()
            self._infer_worker_error = None
            self._infer_pending_frames.clear()
            self._infer_results.clear()
            self._infer_notices.clear()
            self._infer_worker_rr_cursor = 0

        self._infer_thread = threading.Thread(
            target=self._inference_worker_loop,
            name="live-inference-worker",
            daemon=True,
        )
        self._infer_thread.start()

    def _stop_inference_worker(self) -> None:
        self._infer_stop_event.set()
        self._infer_has_work_event.set()
        worker = self._infer_thread
        self._infer_thread = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

        with self._infer_lock:
            self._infer_pending_frames.clear()
            self._infer_results.clear()
            self._infer_notices.clear()
            self._infer_worker_error = None
            self._infer_worker_rr_cursor = 0
            self._infer_has_work_event.clear()

    def _stabilize_detection_boxes(
        self,
        runtime: SourceRuntime,
        boxes: list[PersonDetection],
        now_ts: float,
    ) -> list[PersonDetection]:
        hold_sec = max(0.0, float(self._detection_box_hold_sec))
        current_boxes = [replace(box) for box in boxes]
        if hold_sec <= 1e-6:
            runtime.detection_box_memory = [(replace(box), float(now_ts)) for box in current_boxes]
            return current_boxes

        previous = [
            (box, ts)
            for box, ts in runtime.detection_box_memory
            if (now_ts - float(ts)) <= hold_sec
        ]
        matched_previous: set[int] = set()
        for current in current_boxes:
            best_index = -1
            best_iou = 0.0
            for index, (old_box, _old_ts) in enumerate(previous):
                if index in matched_previous:
                    continue
                if current.track_id is not None and old_box.track_id is not None and int(current.track_id) != int(old_box.track_id):
                    continue
                score = self._bbox_iou(current, old_box)
                if score > best_iou:
                    best_iou = score
                    best_index = index
            if best_index >= 0 and best_iou >= 0.25:
                matched_previous.add(best_index)

        stabilized = list(current_boxes)
        retained_memory: list[tuple[PersonDetection, float]] = [(replace(box), float(now_ts)) for box in current_boxes]
        for index, (old_box, old_ts) in enumerate(previous):
            if index in matched_previous:
                continue
            overlaps_current = any(self._bbox_iou(old_box, current) >= 0.35 for current in current_boxes)
            if overlaps_current:
                continue
            held_box = replace(old_box)
            stabilized.append(held_box)
            retained_memory.append((replace(old_box), float(old_ts)))

        runtime.detection_box_memory = retained_memory
        return stabilized

    def _counts_from_stabilized_boxes(self, boxes: list[PersonDetection], mode: str) -> tuple[int, int]:
        tracked_ids = {box.track_id for box in boxes if box.track_id is not None}
        person_count = len(tracked_ids) if tracked_ids else len(boxes)
        if mode == "night" or not self._uniform_detection_enabled():
            return person_count, person_count
        return person_count, sum(1 for box in boxes if box.is_intruder)

    def _inference_worker_loop(self) -> None:
        while not self._infer_stop_event.is_set():
            if not self._infer_has_work_event.wait(0.05):
                continue
            mode = resolve_security_mode(self.security_cfg)
            source_batch, frame_batch = self._pull_inference_batch()
            if not source_batch:
                continue

            source_batch, frame_batch = self._coalesce_inference_batch(source_batch, frame_batch, mode=mode)
            self._profile_batch_size(len(source_batch))

            try:
                batch_results = self._predict_batch_for_mode(frame_batch, mode)
            except Exception as exc:  # noqa: BLE001
                with self._infer_lock:
                    self._infer_worker_error = (
                        f"Inference worker failed on '{source_batch[0] if source_batch else 'unknown'}': {exc}"
                    )
                self._infer_stop_event.set()
                break

            now = time.perf_counter()
            count = min(len(source_batch), len(batch_results))
            if len(batch_results) != len(source_batch):
                self._queue_async_notice(
                    "[warn] async batch mismatch: "
                    f"expected={len(source_batch)} got={len(batch_results)}"
                )

            for idx in range(count):
                source_name = source_batch[idx]
                frame = frame_batch[idx]
                result = batch_results[idx]
                runtime = self.runtimes.get(source_name)
                boxes, person_count, intruder_count = self._extract_mode_detections(source_name, runtime, result, frame, mode)
                alert_value = intruder_count if mode == "day" else person_count
                alert = should_raise_alert(alert_value, mode, self.security_cfg)
                payload = AsyncInferenceResult(
                    infer_ts=now,
                    person_count=person_count,
                    intruder_count=intruder_count,
                    mode=mode,
                    alert=alert,
                    boxes=boxes,
                )
                with self._infer_lock:
                    self._infer_results[source_name] = payload

    def _apply_async_inference_updates(self) -> bool:
        self._drain_async_notices()

        with self._infer_lock:
            worker_error = self._infer_worker_error
            self._infer_worker_error = None
            updates = self._infer_results
            self._infer_results = {}

        if worker_error:
            self.stop_live()
            QMessageBox.critical(self, "Inference error", worker_error)
            self._log(worker_error)
            return False

        for source_name, payload in updates.items():
            runtime = self.runtimes.get(source_name)
            if runtime is None:
                continue

            if runtime.last_infer_ts > 0:
                infer_delta = max(1e-6, payload.infer_ts - runtime.last_infer_ts)
                runtime.infer_fps = 1.0 / infer_delta
            runtime.last_infer_ts = payload.infer_ts
            stabilized_boxes = self._stabilize_detection_boxes(runtime, payload.boxes, payload.infer_ts)
            person_count, intruder_count = self._counts_from_stabilized_boxes(stabilized_boxes, payload.mode)
            runtime.person_count = person_count
            runtime.intruder_count = intruder_count
            runtime.mode = payload.mode
            alert_value = intruder_count if payload.mode == "day" else person_count
            runtime.alert = should_raise_alert(alert_value, payload.mode, self.security_cfg)
            runtime.last_boxes = stabilized_boxes
            runtime.status = "alert" if runtime.alert else "ok"
        return True

    def _update_load_shed_state(self, now_ts: float) -> None:
        if self._load_shed_level <= 0.0:
            return

        if now_ts < self._load_shed_decay_block_until_ts:
            return

        if (now_ts - self._load_shed_last_pressure_ts) < self._load_shed_hold_seconds:
            return

        new_level = max(0.0, self._load_shed_level - (self._load_shed_decay_per_sec * max(0.0, self.frame_interval_ms) / 1000.0))
        if abs(new_level - self._load_shed_level) < 1e-6:
            return
        self._load_shed_level = new_level

        if self._load_shed_level <= 0.0 and (now_ts - self._load_shed_last_log_ts) >= 2.0:
            self._load_shed_last_log_ts = now_ts
            self._log("[info] Obciazenie wrocilo do normy. Przywracam pelna plynnosc podgladu i inferencji.")

    def _register_event_writer_pressure(self) -> None:
        now_ts = time.perf_counter()
        self._load_shed_last_pressure_ts = now_ts
        self._load_shed_decay_block_until_ts = max(self._load_shed_decay_block_until_ts, now_ts + self._load_shed_sticky_seconds)
        if self._load_shed_level <= 0.0:
            self._load_shed_level = self._load_shed_initial_level
        else:
            self._load_shed_level = min(1.0, self._load_shed_level + 0.25)

        if (now_ts - self._load_shed_last_log_ts) >= 2.0:
            self._load_shed_last_log_ts = now_ts
            effective_view_fps = self._effective_view_target_fps()
            effective_model_fps = self._effective_model_target_fps()
            self._log(
                "[warn] Wykryto przeciazenie zapisu klipow. "
                "Writer klipow nie nadaza z enkodowaniem/zapisem na dysk. Tymczasowo ograniczam obciazenie "
                f"(view_fps~{effective_view_fps:.1f}, model_fps~{effective_model_fps:.1f}) "
                f"na co najmniej {self._load_shed_sticky_seconds:.0f}s."
            )

    def _effective_view_target_fps(self) -> float:
        configured = float(_clamp(float(self.view_target_fps), 1.0, 60.0))
        if self._load_shed_level <= 0.0:
            return configured
        reduction_factor = 1.0 - (0.60 * self._load_shed_level)
        reduced = configured * max(0.10, reduction_factor)
        return max(self._load_shed_min_view_fps, reduced)

    def _effective_model_target_fps(self) -> float:
        configured = max(1.0, float(self.model_target_fps))
        if self._load_shed_level <= 0.0:
            return configured
        reduction_factor = 1.0 - (0.65 * self._load_shed_level)
        reduced = configured * max(0.10, reduction_factor)
        return max(self._load_shed_min_model_fps, reduced)

    def _effective_max_infer_per_tick(self) -> int:
        base = max(1, int(self.max_infer_per_tick))
        if self._load_shed_level <= 0.0:
            return base
        if self._load_shed_level >= 0.50:
            return 1
        return max(1, base - 1)

    def _compute_effective_view_fps_cap(self) -> float:
        configured_cap = self._effective_view_target_fps()
        if not bool(self.runtime_cfg.get("view_cap_to_source_fps", True)):
            return configured_cap
        enabled_sources = self._get_enabled_sources()
        if not enabled_sources:
            return configured_cap

        source_fps_values: list[float] = []
        default_live_fps = max(
            1.0,
            float(
                self.runtime_cfg.get(
                    "camera_fps",
                    self.runtime_cfg.get("video_fps_fallback", 25.0),
                )
            ),
        )

        for source in enabled_sources:
            source_name = str(source.get("name", "source"))
            runtime = self.runtimes.get(source_name)
            if runtime is None:
                continue
            if runtime.source_fps > 1e-3:
                source_fps_values.append(float(runtime.source_fps))
            else:
                source_fps_values.append(default_live_fps)

        if not source_fps_values:
            return configured_cap

        source_cap = max(1.0, min(float(max(source_fps_values)), 60.0))
        return min(configured_cap, source_cap)

    def _compute_live_timer_interval_ms(self) -> int:
        effective_cap = self._compute_effective_view_fps_cap()
        target_interval_ms = max(1, int(round(1000.0 / max(1.0, effective_cap))))
        return max(1, max(int(self.frame_interval_ms), target_interval_ms))

    def _maybe_adjust_live_timer_interval(self, now_ts: float) -> None:
        if not self.live_running:
            return
        if self._live_timer_last_adjust_ts > 0.0 and (now_ts - self._live_timer_last_adjust_ts) < 0.5:
            return

        desired_ms = self._compute_live_timer_interval_ms()
        self._live_timer_last_adjust_ts = now_ts
        if desired_ms == self._live_timer_interval_ms:
            return

        self._live_timer_interval_ms = desired_ms
        self.live_timer.setInterval(desired_ms)

    # ---------- live inference ----------
    def _start_capture_reader(self, source_name: str, runtime: SourceRuntime) -> None:
        capture = runtime.capture
        if capture is None:
            return
        if runtime.capture_reader_thread is not None and runtime.capture_reader_thread.is_alive():
            return

        stop_event = threading.Event()
        runtime.capture_reader_stop_event = stop_event
        runtime.capture_latest_frame = None
        runtime.capture_latest_seq = 0
        runtime.capture_last_consumed_seq = 0

        runtime.capture_reader_thread = threading.Thread(
            target=self._capture_reader_loop,
            args=(source_name, stop_event),
            name=f"capture-reader-{source_name}",
            daemon=True,
        )
        runtime.capture_reader_thread.start()

    def _capture_reader_loop(self, source_name: str, stop_event: threading.Event) -> None:
        next_due_ts = time.perf_counter()
        last_error_log_ts = 0.0
        while not stop_event.is_set():
            runtime = self.runtimes.get(source_name)
            if runtime is None:
                break

            try:
                capture = runtime.capture
                if capture is None or not capture.isOpened():
                    stop_event.wait(0.01)
                    continue

                source_type = str(runtime.source.get("type", "video")).lower()
                if source_type == "video" and runtime.playback_interval_sec > 1e-6:
                    now = time.perf_counter()
                    if now < next_due_ts:
                        stop_event.wait(min(0.005, max(0.0, next_due_ts - now)))
                        continue

                    interval = runtime.playback_interval_sec
                    elapsed = now - next_due_ts
                    frames_to_advance = max(1, 1 + int(elapsed / interval))
                    max_advance = max(1, int(max(1.0, runtime.source_fps) * 2.0))
                    frames_to_advance = min(frames_to_advance, max_advance)

                    grabs_needed = max(0, frames_to_advance - 1)
                    for _ in range(grabs_needed):
                        if not capture.grab():
                            break

                    next_due_ts += frames_to_advance * interval
                    if next_due_ts < (now - (2.0 * interval)):
                        next_due_ts = now

                ok, frame = capture.read()
                if not ok or frame is None:
                    if source_type == "video" and self.loop_videos:
                        try:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            next_due_ts = time.perf_counter()
                        except Exception:  # noqa: BLE001
                            pass
                        continue
                    stop_event.wait(0.005)
                    continue
            except cv2.error as exc:
                now = time.perf_counter()
                if now - last_error_log_ts >= 2.0:
                    self._queue_async_notice(f"[warn] capture-reader '{source_name}' OpenCV error: {exc}")
                    last_error_log_ts = now
                try:
                    if runtime.capture is not None:
                        runtime.capture.release()
                except Exception:  # noqa: BLE001
                    pass
                runtime.capture = None
                stop_event.wait(0.05)
                continue
            except Exception as exc:  # noqa: BLE001
                now = time.perf_counter()
                if now - last_error_log_ts >= 2.0:
                    self._queue_async_notice(f"[warn] capture-reader '{source_name}' error: {exc}")
                    last_error_log_ts = now
                stop_event.wait(0.05)
                continue

            now_ts = time.perf_counter()
            if runtime.last_capture_frame_ts > 0.0:
                frame_delta = max(1e-6, now_ts - runtime.last_capture_frame_ts)
                measured_source_fps = 1.0 / frame_delta
                if runtime.source_fps <= 0.0:
                    runtime.source_fps = measured_source_fps
                else:
                    runtime.source_fps = _ema(runtime.source_fps, measured_source_fps, 0.35)
            runtime.last_capture_frame_ts = now_ts

            with self._capture_lock:
                runtime.capture_latest_frame = frame
                runtime.capture_latest_seq += 1

    def _build_tracker_args(self) -> SimpleNamespace:
        backend = str(self.tracker_cfg.get("backend", self.tracker_backend_name)).strip().lower() or self.tracker_backend_name
        high = float(_clamp(float(self.tracker_cfg.get("track_high_thresh", 0.5)), 0.0, 1.0))
        low = float(_clamp(float(self.tracker_cfg.get("track_low_thresh", 0.1)), 0.0, high))
        new_track = float(_clamp(float(self.tracker_cfg.get("new_track_thresh", 0.6)), low, 1.0))
        match = float(_clamp(float(self.tracker_cfg.get("match_thresh", 0.8)), 0.0, 1.0))
        track_buffer = max(1, int(self.tracker_cfg.get("track_buffer", 30)))
        fuse_score = bool(self.tracker_cfg.get("fuse_score", True))
        args = SimpleNamespace(
            track_high_thresh=high,
            track_low_thresh=low,
            new_track_thresh=new_track,
            track_buffer=track_buffer,
            match_thresh=match,
            fuse_score=fuse_score,
        )
        if backend == "botsort":
            args.tracker_type = "botsort"
            args.proximity_thresh = float(_clamp(float(self.tracker_cfg.get("proximity_thresh", 0.5)), 0.0, 1.0))
            args.appearance_thresh = float(_clamp(float(self.tracker_cfg.get("appearance_thresh", 0.25)), 0.0, 1.0))
            args.with_reid = bool(self.tracker_cfg.get("with_reid", False))
            args.gmc_method = str(self.tracker_cfg.get("gmc_method", "none")).strip() or "none"
            args.cmc_method = str(self.tracker_cfg.get("cmc_method", args.gmc_method)).strip() or args.gmc_method
            args.lambda_ = float(_clamp(float(self.tracker_cfg.get("lambda", 0.98)), 0.0, 1.0))
        else:
            args.tracker_type = "bytetrack"
        return args

    def _reset_tracker_for_source(self, source_name: str, runtime: SourceRuntime | None = None) -> None:
        with self._infer_lock:
            self.trackers.pop(source_name, None)
            self._uniform_track_memory.pop(source_name, None)
        if runtime is not None:
            runtime.last_tracked_boxes = None
            runtime.last_tracker_update_ts = 0.0
        if not self.tracker_enabled or self.tracker_backend_cls is None:
            return

        configured_rate = int(self.tracker_cfg.get("frame_rate", 30))
        if configured_rate > 0:
            frame_rate = configured_rate
        elif runtime is not None and runtime.source_fps > 1e-3:
            frame_rate = max(1, int(round(runtime.source_fps)))
        else:
            frame_rate = 30

        try:
            tracker = self.tracker_backend_cls(args=self._build_tracker_args(), frame_rate=frame_rate)
            with self._infer_lock:
                self.trackers[source_name] = tracker
        except Exception as exc:  # noqa: BLE001
            self._queue_async_notice(f"[warn] Tracker init failed for '{source_name}' ({self.tracker_backend_name}): {exc}")
            with self._infer_lock:
                self.trackers.pop(source_name, None)

    def _ensure_capture(self, runtime: SourceRuntime) -> bool:
        source_name = str(runtime.source.get("name", "source"))
        source_type = str(runtime.source.get("type", "video")).lower()

        if runtime.capture is not None and runtime.capture.isOpened():
            if runtime.capture_reader_thread is None or not runtime.capture_reader_thread.is_alive():
                self._start_capture_reader(source_name, runtime)
            return True

        self._finalize_event_clip(source_name, runtime, time.perf_counter())
        runtime.release()
        capture_source = runtime.source
        if source_type == "camera" and not capture_source.get("backend") and not capture_source.get("camera_backend"):
            capture_source = dict(runtime.source)
            capture_source["backend"] = self.camera_backend
        runtime.capture = open_capture(capture_source)
        ok = runtime.capture.isOpened()
        runtime.source_fps = 0.0
        runtime.last_capture_frame_ts = 0.0
        runtime.playback_interval_sec = 0.0
        runtime.last_frame_due_ts = 0.0
        runtime.last_tick_ts = 0.0
        runtime.last_infer_ts = 0.0
        runtime.fps = 0.0
        runtime.infer_fps = 0.0
        runtime.ui_fps = 0.0
        runtime.smoothed_source_fps = 0.0
        runtime.smoothed_view_fps = 0.0
        runtime.smoothed_infer_fps = 0.0
        runtime.display_source_fps = 0.0
        runtime.display_view_fps = 0.0
        runtime.display_infer_fps = 0.0
        runtime.last_meta_fps_update_ts = 0.0
        runtime.last_render_ts = 0.0
        runtime.person_count = 0
        runtime.intruder_count = 0
        runtime.last_boxes = None
        runtime.detection_box_memory.clear()
        runtime.last_decorated_capture_seq = 0
        runtime.last_decorated_infer_ts = 0.0
        runtime.no_frame_refresh_needed = True
        runtime.event_prebuffer_frames.clear()
        runtime.event_prebuffer_last_store_ts = 0.0
        runtime.person_visible_since_ts = 0.0
        runtime.person_visible_duration_sec = 0.0
        runtime.last_event_capture_ts = 0.0
        runtime.event_saved_in_streak = False
        runtime.event_last_seen_ts = 0.0
        runtime.event_max_person_count = 0
        runtime.event_max_intruder_count = 0
        self._clear_async_state_for_source(source_name)
        with self._infer_lock:
            self.trackers.pop(source_name, None)
        with self._capture_lock:
            runtime.capture_latest_frame = None
            runtime.capture_latest_seq = 0
            runtime.capture_last_consumed_seq = 0

        if ok and source_type == "video":
            raw_fps = float(runtime.capture.get(cv2.CAP_PROP_FPS))
            fallback_fps = max(1.0, float(self.runtime_cfg.get("video_fps_fallback", 25.0)))
            runtime.source_fps = raw_fps if raw_fps > 1e-3 else fallback_fps
            runtime.playback_interval_sec = 1.0 / max(1.0, runtime.source_fps)
        if ok and source_type in {"camera", "stream", "video"}:
            capture_buffer_size = int(self.runtime_cfg.get("capture_buffer_size", 1))
            if capture_buffer_size > 0 and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try:
                    runtime.capture.set(cv2.CAP_PROP_BUFFERSIZE, capture_buffer_size)
                except Exception:  # noqa: BLE001
                    pass

            if source_type == "camera":
                camera_width = int(self.runtime_cfg.get("camera_width", 0))
                camera_height = int(self.runtime_cfg.get("camera_height", 0))
                camera_fps = float(self.runtime_cfg.get("camera_fps", 0))
                apply_requested_camera_settings = not runtime.camera_settings_rejected
                if apply_requested_camera_settings and camera_width > 0:
                    try:
                        runtime.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera_width))
                    except Exception:  # noqa: BLE001
                        pass
                if apply_requested_camera_settings and camera_height > 0:
                    try:
                        runtime.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera_height))
                    except Exception:  # noqa: BLE001
                        pass
                if apply_requested_camera_settings and camera_fps > 0:
                    try:
                        runtime.capture.set(cv2.CAP_PROP_FPS, float(camera_fps))
                    except Exception:  # noqa: BLE001
                        pass

                if apply_requested_camera_settings and not camera_capture_can_read(runtime.capture, attempts=3):
                    runtime.camera_settings_rejected = True
                    self._queue_async_notice(
                        f"[warn] Camera '{source_name}' rejected requested "
                        "resolution/FPS; reopening with default camera settings."
                    )
                    try:
                        runtime.capture.release()
                    except Exception:  # noqa: BLE001
                        pass
                    runtime.capture = open_capture(capture_source)
                    ok = runtime.capture.isOpened() and camera_capture_can_read(runtime.capture, attempts=3)
                    if not ok:
                        self._queue_async_notice(
                            f"[warn] Camera '{source_name}' cannot deliver frames with default settings."
                        )

                requested_fps = float(self.runtime_cfg.get("camera_fps", 30))
                try:
                    actual_fps = float(runtime.capture.get(cv2.CAP_PROP_FPS)) if runtime.capture is not None else 0.0
                except Exception:  # noqa: BLE001
                    actual_fps = 0.0
                runtime.source_fps = actual_fps if actual_fps > 1e-3 else max(1.0, requested_fps)

            if ok:
                self._start_capture_reader(source_name, runtime)
        if ok:
            self._reset_tracker_for_source(source_name, runtime)

        runtime.status = "open" if ok else "failed"
        return ok

    def _read_frame(self, runtime: SourceRuntime) -> tuple[np.ndarray | None, bool]:
        read_started_ts = time.perf_counter()
        if not self._ensure_capture(runtime):
            self._profile_add("capture", time.perf_counter() - read_started_ts)
            return None, False

        latest_frame: np.ndarray | None = None
        latest_seq = 0
        consumed_seq = 0
        with self._capture_lock:
            if runtime.capture_latest_frame is not None:
                latest_frame = runtime.capture_latest_frame
            latest_seq = int(runtime.capture_latest_seq)
            consumed_seq = int(runtime.capture_last_consumed_seq)

        if latest_frame is not None:
            fresh = latest_seq != consumed_seq
            if fresh:
                with self._capture_lock:
                    runtime.capture_last_consumed_seq = latest_seq
            runtime.last_input = latest_frame
            self._profile_add("capture", time.perf_counter() - read_started_ts)
            return latest_frame, fresh

        runtime.status = "no-frame"
        if runtime.last_input is not None:
            self._profile_add("capture", time.perf_counter() - read_started_ts)
            return runtime.last_input, False
        self._profile_add("capture", time.perf_counter() - read_started_ts)
        return None, False

    def _predict_batch_with_fallback(self, frames: list[np.ndarray]) -> list[Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not frames:
            return []

        with self._model_lock:
            try:
                results = self.model.predict(frames, **self.predict_kwargs)
                return self._results_to_list(results)
            except Exception as exc:  # noqa: BLE001
                if not self.compile_enabled:
                    raise

                if not self._is_compile_runtime_error(exc):
                    raise

                if not self.compile_fallback_applied:
                    self.compile_fallback_applied = True
                    self._queue_async_notice(
                        "compile failed for inference, fallback to compile=False "
                        f"({exc.__class__.__name__}: {exc})"
                    )

                self.compile_enabled = False
                self.inference_cfg["compile"] = False
                self.predict_kwargs.pop("compile", None)
                try:
                    self.model.predictor = None
                except Exception:  # noqa: BLE001
                    pass

                results = self.model.predict(frames, **self.predict_kwargs)
                return self._results_to_list(results)

    @staticmethod
    def _tensor_like_to_torch(value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        try:
            if isinstance(value, torch.Tensor):
                return value.detach()
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            return torch.as_tensor(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _tensor_like_to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            return np.asarray(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _results_to_list(results: Any) -> list[Any]:
        if isinstance(results, list):
            return results
        return list(results)

    def _extract_person_box_arrays(
        self,
        result: Any,
    ) -> tuple[list[PersonDetection], np.ndarray | None, np.ndarray | None]:
        box_extract_started_ts = time.perf_counter()
        raw_boxes = getattr(result, "boxes", None)
        if raw_boxes is None:
            self._profile_add("box_extract", time.perf_counter() - box_extract_started_ts)
            return [], None, None

        try:
            xyxy_tensor = self._tensor_like_to_torch(raw_boxes.xyxy)
            conf_tensor = self._tensor_like_to_torch(raw_boxes.conf)
            cls_tensor = self._tensor_like_to_torch(raw_boxes.cls)
            raw_data_tensor = self._tensor_like_to_torch(raw_boxes.data if hasattr(raw_boxes, "data") else None)
        except Exception:  # noqa: BLE001
            self._profile_add("box_extract", time.perf_counter() - box_extract_started_ts)
            return [], None, None

        if xyxy_tensor is None or xyxy_tensor.ndim != 2 or int(xyxy_tensor.shape[1]) < 4:
            self._profile_add("box_extract", time.perf_counter() - box_extract_started_ts)
            return [], None, None

        row_count = int(xyxy_tensor.shape[0])
        if row_count <= 0:
            self._profile_add("box_extract", time.perf_counter() - box_extract_started_ts)
            return [], None, None

        if conf_tensor is None or int(conf_tensor.numel()) != row_count:
            conf_tensor = torch.ones((row_count,), dtype=torch.float32, device=xyxy_tensor.device)
        else:
            conf_tensor = conf_tensor.reshape(-1).to(dtype=torch.float32)

        if cls_tensor is None or int(cls_tensor.numel()) != row_count:
            cls_tensor = torch.zeros((row_count,), dtype=torch.float32, device=xyxy_tensor.device)
        else:
            cls_tensor = cls_tensor.reshape(-1)

        class_mask = cls_tensor.to(dtype=torch.int64) == 0
        geometry_mask = (xyxy_tensor[:, 2] > xyxy_tensor[:, 0]) & (xyxy_tensor[:, 3] > xyxy_tensor[:, 1])
        keep_mask = class_mask & geometry_mask

        keep_indices_tensor = torch.nonzero(keep_mask, as_tuple=False).flatten()
        if int(keep_indices_tensor.numel()) <= 0:
            self._profile_add("box_extract", time.perf_counter() - box_extract_started_ts)
            return [], None, None

        selected_xyxy = xyxy_tensor.index_select(0, keep_indices_tensor).to(dtype=torch.float32)
        selected_conf = conf_tensor.index_select(0, keep_indices_tensor)

        xyxy_array = self._tensor_like_to_numpy(selected_xyxy)
        conf_array = self._tensor_like_to_numpy(selected_conf)
        keep_index_array = self._tensor_like_to_numpy(keep_indices_tensor.to(dtype=torch.int32))

        if xyxy_array is None or conf_array is None or keep_index_array is None:
            self._profile_add("box_extract", time.perf_counter() - box_extract_started_ts)
            return [], None, None

        raw_box_array: np.ndarray | None = None
        if raw_data_tensor is not None and raw_data_tensor.ndim == 2 and int(raw_data_tensor.shape[0]) == row_count:
            selected_raw = raw_data_tensor.index_select(0, keep_indices_tensor)
            raw_box_array = self._tensor_like_to_numpy(selected_raw)
        if raw_box_array is None:
            selected_cls = cls_tensor.index_select(0, keep_indices_tensor).to(dtype=torch.float32)
            raw_box_tensor = torch.cat(
                (
                    selected_xyxy,
                    selected_conf.unsqueeze(1),
                    selected_cls.unsqueeze(1),
                ),
                dim=1,
            )
            raw_box_array = self._tensor_like_to_numpy(raw_box_tensor)

        detections: list[PersonDetection] = []
        selected_count = int(xyxy_array.shape[0])
        for index in range(selected_count):
            coords = xyxy_array[index]
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])
            detections.append(
                PersonDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(conf_array[index]),
                    track_id=None,
                )
            )

        self._profile_add("box_extract", time.perf_counter() - box_extract_started_ts)
        return detections, keep_index_array, raw_box_array

    def _predict_batch_for_mode(self, frames: list[np.ndarray], mode: str) -> list[Any]:
        predict_started_ts = time.perf_counter()
        if mode == "day" and self._uniform_detection_enabled() and self.day_seg_model is not None:
            with self._model_lock:
                results = self._results_to_list(self.day_seg_model.predict(frames, **self.day_seg_predict_kwargs))
            self._profile_add("predict", time.perf_counter() - predict_started_ts, units=len(frames))
            return results
        results = self._predict_batch_with_fallback(frames)
        self._profile_add("predict", time.perf_counter() - predict_started_ts, units=len(frames))
        return results

    def _extract_person_boxes(self, result: Any) -> list[PersonDetection]:
        detections, _keep_indices, _raw_box_array = self._extract_person_box_arrays(result)
        return detections

    def _extract_person_segmentation_masks(
        self,
        result: Any,
        pre_extracted: tuple[list[PersonDetection], np.ndarray | None] | None = None,
    ) -> tuple[list[PersonDetection], np.ndarray | None]:
        if pre_extracted is None:
            detections, keep_indices, _raw_box_array = self._extract_person_box_arrays(result)
        else:
            detections, keep_indices = pre_extracted
        if not detections:
            return [], None

        raw_masks = getattr(result, "masks", None)
        if raw_masks is None:
            return detections, None

        try:
            mask_tensor = self._tensor_like_to_torch(raw_masks.data)
            if mask_tensor is None or mask_tensor.ndim != 3:
                return detections, None
            if keep_indices is None or keep_indices.size <= 0:
                return detections, None

            index_tensor = torch.as_tensor(keep_indices, device=mask_tensor.device, dtype=torch.int64)
            index_tensor = index_tensor[index_tensor < int(mask_tensor.shape[0])]
            if int(index_tensor.numel()) <= 0:
                return detections, None

            selected_masks_tensor = mask_tensor.index_select(0, index_tensor)
            selected_masks = self._tensor_like_to_numpy(selected_masks_tensor)
            if selected_masks is None or selected_masks.ndim != 3:
                return detections, None
            return detections, np.ascontiguousarray(selected_masks)
        except Exception:  # noqa: BLE001
            return detections, None

    def _analyze_day_uniform_detections(
        self,
        source_name: str,
        frame: np.ndarray,
        result: Any,
        tracked_boxes: list[PersonDetection],
        pre_extracted: tuple[list[PersonDetection], np.ndarray | None] | None = None,
    ) -> list[PersonDetection]:
        uniform_started_ts = time.perf_counter()
        detections_only, selected_masks = self._extract_person_segmentation_masks(result, pre_extracted=pre_extracted)
        if not detections_only:
            self._profile_add("uniform", time.perf_counter() - uniform_started_ts)
            return []

        self._attach_track_ids_to_day_detections(detections_only, tracked_boxes)

        tolerance = _uniform_color_tolerance(self.uniform_cfg.get("color_tolerance", UNIFORM_COLOR_TOLERANCE_DEFAULT))
        min_pixels = max(20, int(self.uniform_cfg.get("min_mask_pixels", UNIFORM_MIN_MASK_PIXELS_DEFAULT)))
        center_band_fraction = _clamp(float(self.uniform_cfg.get("center_band_fraction", 0.6)), 0.3, 1.0)
        target_upper_hex = _normalize_hex_color(self.uniform_cfg.get("top_color", UNIFORM_TOP_DEFAULT), UNIFORM_TOP_DEFAULT)
        target_lower_hex = _normalize_hex_color(self.uniform_cfg.get("bottom_color", UNIFORM_BOTTOM_DEFAULT), UNIFORM_BOTTOM_DEFAULT)
        target_upper_bgr = _hex_to_bgr(target_upper_hex)
        target_lower_bgr = _hex_to_bgr(target_lower_hex)
        min_section_pixels = max(24, min_pixels // 3)
        now_ts = time.perf_counter()

        with self._infer_lock:
            source_memory_snapshot = dict(self._uniform_track_memory.get(source_name, {}))

        detections: list[PersonDetection] = []
        fresh_analysis_used = 0
        for index, detection in enumerate(detections_only):
            cache_entry = None
            if detection.track_id is not None:
                cache_entry = source_memory_snapshot.get(int(detection.track_id))
            if cache_entry and self._apply_worker_hold_decision(detection, cache_entry, now_ts):
                detections.append(detection)
                continue
            if (
                cache_entry
                and self._uniform_recheck_interval_sec > 0.0
                and (now_ts - float(cache_entry.get("analysis_ts", 0.0))) <= self._uniform_recheck_interval_sec
                and self._apply_cached_uniform_decision(detection, cache_entry)
            ):
                detections.append(detection)
                continue

            if cache_entry is not None and fresh_analysis_used >= self._uniform_max_fresh_per_cycle:
                if self._apply_cached_uniform_decision(detection, cache_entry):
                    detections.append(detection)
                    continue

            fresh_analysis_used += 1

            seg_mask = None
            if selected_masks is not None and index < int(selected_masks.shape[0]):
                seg_mask = selected_masks[index]
            mask_roi_info = _extract_mask_roi_for_detection(seg_mask, detection, frame.shape)
            if mask_roi_info is None:
                detection.has_segmentation = False
                detection.upper_match = None
                detection.lower_match = None
                detection.uniform_match = None
                detection.visible_section_count = 0
                detection.is_intruder = True
                detection.label = "intruz"
                self._store_uniform_analysis_cache(source_name, detection, now_ts)
                detections.append(detection)
                continue

            working_mask, clipped_bbox = mask_roi_info
            x1, y1, x2, y2 = clipped_bbox
            frame_roi = frame[y1:y2, x1:x2]
            detection.has_segmentation = True

            bounds = _mask_row_percentile_bounds(working_mask)
            if bounds is None:
                detection.has_segmentation = False
                detection.upper_match = None
                detection.lower_match = None
                detection.uniform_match = None
                detection.visible_section_count = 0
                detection.is_intruder = True
                detection.label = "intruz"
                self._store_uniform_analysis_cache(source_name, detection, now_ts)
                detections.append(detection)
                continue

            upper_start, upper_end, lower_start, lower_end = bounds

            upper_sample = _compute_region_color(
                frame_roi,
                working_mask,
                max(0, upper_start),
                min(working_mask.shape[0], upper_end),
                center_band_fraction=center_band_fraction,
            )
            lower_sample = _compute_region_color(
                frame_roi,
                working_mask,
                max(0, lower_start),
                min(working_mask.shape[0], lower_end),
                center_band_fraction=center_band_fraction,
            )

            upper_match: bool | None = None
            lower_match: bool | None = None
            visible_section_count = 0
            if upper_sample is not None:
                detection.upper_color_hex = upper_sample[0]
                if upper_sample[1] >= min_section_pixels:
                    visible_section_count += 1
                if upper_sample[1] >= min_pixels:
                    upper_match = _uniform_color_matches(_hex_to_bgr(upper_sample[0]), target_upper_bgr, tolerance)
            if lower_sample is not None:
                detection.lower_color_hex = lower_sample[0]
                if lower_sample[1] >= min_section_pixels:
                    visible_section_count += 1
                if lower_sample[1] >= min_pixels:
                    lower_match = _uniform_color_matches(_hex_to_bgr(lower_sample[0]), target_lower_bgr, tolerance)

            detection.upper_match = upper_match
            detection.lower_match = lower_match
            detection.visible_section_count = visible_section_count

            false_count = int(upper_match is False) + int(lower_match is False)
            true_count = int(upper_match is True) + int(lower_match is True)
            if false_count > 0:
                detection.uniform_match = False
                detection.is_intruder = True
            elif true_count > 0:
                detection.uniform_match = True
                detection.is_intruder = False
            else:
                detection.uniform_match = None
                detection.is_intruder = True
            detection.label = "intruz" if detection.is_intruder else "pracownik"
            self._store_uniform_analysis_cache(source_name, detection, now_ts)
            detections.append(detection)

        self._profile_add("uniform", time.perf_counter() - uniform_started_ts)
        return detections

    def _extract_tracked_person_boxes(
        self,
        source_name: str,
        runtime: SourceRuntime | None,
        result: Any,
        frame: np.ndarray,
        mode: str = "day",
        pre_extracted: tuple[list[PersonDetection], np.ndarray | None, np.ndarray | None] | None = None,
    ) -> list[PersonDetection]:
        if not self.tracker_enabled:
            if pre_extracted is not None:
                return pre_extracted[0]
            return self._extract_person_boxes(result)

        default_update_fps = min(float(self.model_target_fps), 12.0)
        tracker_update_fps = max(1.0, float(self.tracker_cfg.get("max_update_fps", default_update_fps)))
        if pre_extracted is None:
            raw_detections, _keep_indices, raw_box_array = self._extract_person_box_arrays(result)
        else:
            raw_detections, _keep_indices, raw_box_array = pre_extracted
        if runtime is not None and runtime.last_tracked_boxes is not None and runtime.last_tracker_update_ts > 0.0:
            now_ts = time.perf_counter()
            if (now_ts - runtime.last_tracker_update_ts) < (1.0 / tracker_update_fps):
                tracker_started_ts = time.perf_counter()
                reused = self._reuse_recent_track_ids(raw_detections, runtime.last_tracked_boxes)
                self._profile_add("tracker", time.perf_counter() - tracker_started_ts)
                return reused

        if raw_box_array is None or raw_box_array.ndim != 2 or raw_box_array.shape[0] == 0:
            return []

        with self._infer_lock:
            tracker = self.trackers.get(source_name)
        if tracker is None:
            self._reset_tracker_for_source(source_name, runtime)
            with self._infer_lock:
                tracker = self.trackers.get(source_name)
            if tracker is None:
                return raw_detections

        try:
            tracker_started_ts = time.perf_counter()
            raw_boxes = getattr(result, "boxes", None)
            if raw_boxes is not None and hasattr(raw_boxes, "cpu"):
                raw_boxes = raw_boxes.cpu()
            tracks = tracker.update(raw_boxes, img=frame)
            self._profile_add("tracker", time.perf_counter() - tracker_started_ts)
        except Exception as exc:  # noqa: BLE001
            now_ts = time.perf_counter()
            last_warn_ts = float(self._tracker_warn_last_ts.get(source_name, 0.0))
            if (now_ts - last_warn_ts) >= 2.0:
                self._tracker_warn_last_ts[source_name] = now_ts
                self._queue_async_notice(
                    f"[warn] Tracker failed on '{source_name}' ({self.tracker_backend_name}), fallback to raw detections: {exc}"
                )
            self._reset_tracker_for_source(source_name, runtime)
            return raw_detections

        boxes: list[PersonDetection] = []
        if tracks is None or len(tracks) == 0:
            if runtime is not None:
                runtime.last_tracked_boxes = []
                runtime.last_tracker_update_ts = time.perf_counter()
            return boxes

        track_rows = self._tensor_like_to_numpy(tracks)
        if track_rows is None or track_rows.ndim != 2:
            return boxes

        for row in track_rows:
            if len(row) < 7:
                continue
            x1, y1, x2, y2, track_id, score, cls_id = row[:7]
            if int(cls_id) != 0:
                continue

            ix1 = int(round(float(x1)))
            iy1 = int(round(float(y1)))
            ix2 = int(round(float(x2)))
            iy2 = int(round(float(y2)))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            try:
                parsed_track_id: int | None = int(track_id)
            except Exception:  # noqa: BLE001
                parsed_track_id = None
            boxes.append(
                PersonDetection(
                    x1=ix1,
                    y1=iy1,
                    x2=ix2,
                    y2=iy2,
                    conf=float(score),
                    track_id=parsed_track_id,
                )
            )
        if runtime is not None:
            runtime.last_tracked_boxes = boxes
            runtime.last_tracker_update_ts = time.perf_counter()
        return boxes

    def _bbox_iou(self, first: PersonDetection, second: PersonDetection) -> float:
        return self._bbox_iou_coords(first.x1, first.y1, first.x2, first.y2, second.x1, second.y1, second.x2, second.y2)

    @staticmethod
    def _bbox_iou_coords(
        ax1: int,
        ay1: int,
        ax2: int,
        ay2: int,
        bx1: int,
        by1: int,
        bx2: int,
        by2: int,
    ) -> float:
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        first_area = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        second_area = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        union = first_area + second_area - inter_area
        if union <= 1e-6:
            return 0.0
        return inter_area / union

    def _attach_track_ids_to_day_detections(
        self,
        detections: list[PersonDetection],
        tracked_boxes: list[PersonDetection],
    ) -> None:
        match_started_ts = time.perf_counter()
        if not detections or not tracked_boxes:
            self._profile_add("match_ids", time.perf_counter() - match_started_ts)
            return

        used_tracked: set[int] = set()
        for detection in sorted(detections, key=lambda item: float(item.conf), reverse=True):
            best_index = -1
            best_iou = 0.0
            for index, tracked in enumerate(tracked_boxes):
                if index in used_tracked:
                    continue
                score = self._bbox_iou(detection, tracked)
                if score > best_iou:
                    best_iou = score
                    best_index = index

            if best_index < 0 or best_iou < 0.15:
                continue

            tracked = tracked_boxes[best_index]
            detection.track_id = tracked.track_id
            detection.conf = max(float(detection.conf), float(tracked.conf))
            used_tracked.add(best_index)
        self._profile_add("match_ids", time.perf_counter() - match_started_ts)

    def _reuse_recent_track_ids(
        self,
        detections: list[PersonDetection],
        previous_tracked_boxes: list[PersonDetection] | None,
    ) -> list[PersonDetection]:
        if not detections or not previous_tracked_boxes:
            return detections
        self._attach_track_ids_to_day_detections(detections, previous_tracked_boxes)
        return detections

    def _apply_cached_uniform_decision(
        self,
        detection: PersonDetection,
        entry: dict[str, Any],
    ) -> bool:
        analysis_ts = float(entry.get("analysis_ts", 0.0))
        cached_bbox = entry.get("analysis_bbox")
        if analysis_ts <= 0.0 or not isinstance(cached_bbox, (list, tuple)) or len(cached_bbox) != 4:
            return False

        try:
            cx1 = int(cached_bbox[0])
            cy1 = int(cached_bbox[1])
            cx2 = int(cached_bbox[2])
            cy2 = int(cached_bbox[3])
        except Exception:  # noqa: BLE001
            return False

        if self._bbox_iou_coords(detection.x1, detection.y1, detection.x2, detection.y2, cx1, cy1, cx2, cy2) < self._uniform_recheck_iou:
            return False

        detection.has_segmentation = bool(entry.get("has_segmentation", False))
        detection.upper_match = entry.get("upper_match")
        detection.lower_match = entry.get("lower_match")
        detection.upper_color_hex = str(entry.get("upper_color_hex", "") or "")
        detection.lower_color_hex = str(entry.get("lower_color_hex", "") or "")
        detection.uniform_match = entry.get("uniform_match")
        detection.visible_section_count = int(entry.get("visible_section_count", 0) or 0)
        detection.is_intruder = bool(entry.get("is_intruder", True))
        detection.label = "intruz" if detection.is_intruder else "pracownik"
        detection.uniform_cached_decision = True
        return True

    def _apply_worker_hold_decision(
        self,
        detection: PersonDetection,
        entry: dict[str, Any],
        now_ts: float,
    ) -> bool:
        hold_until = float(entry.get("worker_hold_until_ts", 0.0))
        if hold_until <= now_ts or bool(entry.get("is_intruder", True)):
            return False

        detection.has_segmentation = bool(entry.get("has_segmentation", False))
        detection.upper_match = entry.get("upper_match")
        detection.lower_match = entry.get("lower_match")
        detection.upper_color_hex = str(entry.get("upper_color_hex", "") or "")
        detection.lower_color_hex = str(entry.get("lower_color_hex", "") or "")
        detection.uniform_match = True
        detection.visible_section_count = int(entry.get("visible_section_count", 0) or 0)
        detection.is_intruder = False
        detection.label = "pracownik"
        detection.uniform_cached_decision = True
        return True

    def _store_uniform_analysis_cache(
        self,
        source_name: str,
        detection: PersonDetection,
        now_ts: float,
    ) -> None:
        if detection.track_id is None:
            return

        track_id = int(detection.track_id)
        with self._infer_lock:
            source_memory = self._uniform_track_memory.setdefault(source_name, {})
            entry = source_memory.get(track_id)
            if entry is None:
                entry = {}
                source_memory[track_id] = entry
            previous_hold_until = float(entry.get("worker_hold_until_ts", 0.0))
            if detection.is_intruder or detection.uniform_cached_decision:
                worker_hold_until = previous_hold_until
            else:
                worker_hold_until = max(previous_hold_until, float(now_ts) + self._uniform_worker_hold_sec)
            entry.update(
                {
                    "analysis_ts": float(now_ts),
                    "analysis_bbox": [int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)],
                    "has_segmentation": bool(detection.has_segmentation),
                    "upper_match": detection.upper_match,
                    "lower_match": detection.lower_match,
                    "upper_color_hex": detection.upper_color_hex,
                    "lower_color_hex": detection.lower_color_hex,
                    "uniform_match": detection.uniform_match,
                    "visible_section_count": int(detection.visible_section_count),
                    "is_intruder": bool(detection.is_intruder),
                    "worker_hold_until_ts": float(worker_hold_until),
                    "last_seen_ts": float(now_ts),
                }
            )

    def _apply_uniform_temporal_memory(self, source_name: str, detections: list[PersonDetection]) -> None:
        now_ts = time.perf_counter()
        with self._infer_lock:
            source_memory = self._uniform_track_memory.setdefault(source_name, {})

            for track_id in list(source_memory.keys()):
                last_seen = float(source_memory[track_id].get("last_seen_ts", 0.0))
                if (now_ts - last_seen) > self._uniform_memory_ttl_sec:
                    del source_memory[track_id]

            for detection in detections:
                if detection.track_id is None:
                    continue

                track_id = int(detection.track_id)
                entry = source_memory.get(track_id, {})
                previous_score = float(entry.get("score", 0.0))
                bad_streak = int(entry.get("bad_streak", 0))
                previous_worker_streak = int(entry.get("worker_streak", 0))
                worker_streak = previous_worker_streak
                previous_is_intruder = bool(entry.get("is_intruder", True))
                previous_last_seen = float(entry.get("last_seen_ts", 0.0))

                false_count = int(detection.upper_match is False) + int(detection.lower_match is False)
                true_count = int(detection.upper_match is True) + int(detection.lower_match is True)
                unknown_count = 2 - false_count - true_count

                if false_count > 0:
                    vote = -1.0
                elif true_count == 2:
                    vote = 1.0
                elif true_count == 1 and unknown_count == 1:
                    vote = 0.45
                elif true_count == 0 and unknown_count == 2:
                    vote = previous_score * self._uniform_memory_decay
                else:
                    vote = 0.10

                decayed = previous_score * self._uniform_memory_decay
                score = ((1.0 - self._uniform_memory_alpha) * decayed) + (self._uniform_memory_alpha * vote)
                score = float(_clamp(score, -1.0, 1.0))

                current_intruder = bool(detection.is_intruder)
                bad_streak = bad_streak + 1 if current_intruder else 0
                worker_streak = worker_streak + 1 if not current_intruder else 0

                strong_worker_history = score >= self._uniform_memory_min_worker_score
                recent_confirmed_worker = (
                    not previous_is_intruder
                    and previous_last_seen > 0.0
                    and (now_ts - previous_last_seen) <= min(self._uniform_memory_ttl_sec, 2.5)
                    and (previous_worker_streak >= 2 or previous_score >= self._uniform_memory_min_worker_score)
                )
                if current_intruder and strong_worker_history:
                    keep_as_worker = False
                    if false_count <= 1 and bad_streak <= self._uniform_memory_max_bad_streak:
                        keep_as_worker = True
                    elif false_count == 2 and bad_streak <= 1 and float(detection.conf) < 0.90:
                        keep_as_worker = True
                    elif false_count == 0 and unknown_count >= 1:
                        keep_as_worker = True
                    elif recent_confirmed_worker and false_count == 0 and unknown_count >= 1:
                        keep_as_worker = True
                    elif (
                        recent_confirmed_worker
                        and false_count <= 1
                        and bad_streak <= (self._uniform_memory_max_bad_streak + 1)
                        and int(detection.visible_section_count) <= 1
                    ):
                        keep_as_worker = True
                    elif (
                        recent_confirmed_worker
                        and false_count == 2
                        and bad_streak <= 1
                        and float(detection.conf) < 0.75
                        and int(detection.visible_section_count) <= 1
                    ):
                        keep_as_worker = True

                    if keep_as_worker:
                        detection.is_intruder = False
                        detection.uniform_match = True
                        detection.label = "pracownik"
                        bad_streak = max(0, bad_streak - 1)
                        worker_streak = max(worker_streak, previous_worker_streak + 1)

                if detection.is_intruder:
                    detection.label = "intruz"
                else:
                    detection.label = "pracownik"

                previous_hold_until = float(entry.get("worker_hold_until_ts", 0.0))
                worker_hold_until = previous_hold_until
                if not detection.is_intruder and not detection.uniform_cached_decision:
                    worker_hold_until = max(previous_hold_until, now_ts + self._uniform_worker_hold_sec)

                source_memory[track_id] = {
                    "score": score,
                    "bad_streak": bad_streak,
                    "worker_streak": worker_streak,
                    "last_seen_ts": now_ts,
                    "analysis_ts": float(entry.get("analysis_ts", now_ts)),
                    "analysis_bbox": entry.get("analysis_bbox", [int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)]),
                    "has_segmentation": bool(detection.has_segmentation),
                    "upper_match": detection.upper_match,
                    "lower_match": detection.lower_match,
                    "upper_color_hex": detection.upper_color_hex,
                    "lower_color_hex": detection.lower_color_hex,
                    "uniform_match": detection.uniform_match,
                    "visible_section_count": int(detection.visible_section_count),
                    "is_intruder": bool(detection.is_intruder),
                    "worker_hold_until_ts": float(worker_hold_until),
                }

    def _extract_mode_detections(
        self,
        source_name: str,
        runtime: SourceRuntime | None,
        result: Any,
        frame: np.ndarray,
        mode: str,
    ) -> tuple[list[PersonDetection], int, int]:
        postprocess_started_ts = time.perf_counter()
        if mode == "day" and self._uniform_detection_enabled():
            pre_detections, pre_keep_indices, pre_raw_box_array = self._extract_person_box_arrays(result)
            tracked_boxes = self._extract_tracked_person_boxes(
                source_name,
                runtime,
                result,
                frame,
                mode,
                pre_extracted=(pre_detections, pre_keep_indices, pre_raw_box_array),
            )
            detections = self._analyze_day_uniform_detections(
                source_name,
                frame,
                result,
                tracked_boxes,
                pre_extracted=(pre_detections, pre_keep_indices),
            )
            self._apply_uniform_temporal_memory(source_name, detections)

            counts_started_ts = time.perf_counter()
            tracked_ids = {detection.track_id for detection in detections if detection.track_id is not None}
            person_count = len(tracked_ids) if tracked_ids else len(detections)
            intruder_count = sum(1 for detection in detections if detection.is_intruder)
            self._profile_add("counts_labels", time.perf_counter() - counts_started_ts)
            self._profile_add("postprocess", time.perf_counter() - postprocess_started_ts)
            return detections, person_count, intruder_count

        detections = self._extract_tracked_person_boxes(source_name, runtime, result, frame, mode)
        counts_started_ts = time.perf_counter()
        tracked_ids = {detection.track_id for detection in detections if detection.track_id is not None}
        person_count = len(tracked_ids) if tracked_ids else len(detections)
        intruder_count = person_count if mode == "night" or not self._uniform_detection_enabled() else 0
        for detection in detections:
            detection.is_intruder = mode == "night" or not self._uniform_detection_enabled()
            detection.uniform_match = None
            detection.label = "intruz" if detection.is_intruder else "pracownik"
        self._profile_add("counts_labels", time.perf_counter() - counts_started_ts)
        self._profile_add("postprocess", time.perf_counter() - postprocess_started_ts)
        return detections, person_count, intruder_count

    def _draw_person_boxes(
        self,
        frame: np.ndarray,
        boxes: list[PersonDetection] | None,
    ) -> np.ndarray:
        output = frame.copy()
        if not boxes:
            return output

        frame_h, frame_w = output.shape[:2]
        style = _scaled_annotation_style(
            output,
            reference_height=self.annotation_reference_height,
            label_font_base=self.annotation_label_font_scale,
            status_font_base=self.annotation_status_font_scale,
            min_scale=self.annotation_min_resolution_scale,
            max_scale=self.annotation_max_resolution_scale,
        )
        label_font_scale = float(style["label_font_scale"])
        text_thickness = int(style["text_thickness"])
        box_thickness = int(style["box_thickness"])
        pad_x = int(style["pad_x"])
        pad_y = int(style["pad_y"])
        gap = int(style["gap"])

        for detection in boxes:
            x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
            box_color = (0, 0, 255) if detection.is_intruder else (42, 169, 107)
            if detection.uniform_match is None and not detection.is_intruder:
                box_color = (255, 176, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, box_thickness)
            if detection.track_id is None:
                base_label = detection.label or "person"
            else:
                base_label = f"{detection.label or 'person'}#{detection.track_id}"
            label = f"{base_label} {detection.conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                text_thickness,
            )
            label_w = tw + (pad_x * 2)
            text_x = int(_clamp(float(x1), 0.0, float(max(0, frame_w - label_w))))
            text_y = y1 - gap
            top = text_y - th - baseline - pad_y
            if top < 0:
                text_y = min(frame_h - baseline - pad_y, y1 + th + baseline + (pad_y * 2) + gap)
                top = max(0, text_y - th - baseline - pad_y)
            bottom = min(frame_h, text_y + pad_y)
            cv2.rectangle(
                output,
                (text_x, top),
                (min(frame_w, text_x + label_w), bottom),
                box_color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (text_x + pad_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                (20, 20, 20),
                text_thickness,
                cv2.LINE_AA,
            )
        return output

    def _decorate_live_frame(
        self,
        frame: np.ndarray,
        *,
        source_name: str,
        mode: str,
        person_count: int,
        intruder_count: int,
        alert: bool,
        boxes: list[PersonDetection] | None = None,
    ) -> np.ndarray:
        draw_started_ts = time.perf_counter()
        color = (0, 0, 255) if alert else (0, 185, 0)
        status = "ALERT" if alert else "OK"
        text = f"{source_name} | mode:{mode} | person:{person_count} | intruder:{intruder_count} | {status}"

        output = self._draw_person_boxes(frame, boxes)
        style = _scaled_annotation_style(
            output,
            reference_height=self.annotation_reference_height,
            label_font_base=self.annotation_label_font_scale,
            status_font_base=self.annotation_status_font_scale,
            min_scale=self.annotation_min_resolution_scale,
            max_scale=self.annotation_max_resolution_scale,
        )
        cv2.putText(
            output,
            text,
            (int(style["status_x"]), int(style["status_y"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(style["status_font_scale"]),
            color,
            int(style["text_thickness"]),
            cv2.LINE_AA,
        )
        self._profile_add("draw", time.perf_counter() - draw_started_ts)
        return output

    def _refresh_tile(self, source_name: str) -> None:
        runtime = self.runtimes.get(source_name)
        if runtime is None:
            return

        now = time.perf_counter()
        if runtime.last_render_ts > 0:
            render_delta = max(1e-6, now - runtime.last_render_ts)
            runtime.ui_fps = 1.0 / render_delta
        runtime.last_render_ts = now

        zoom = self.zoom_levels.get(source_name, 1.0)
        pan_x, pan_y = self.pan_offsets.get(source_name, (0.0, 0.0))
        source_fps = runtime.source_fps if runtime.source_fps > 0.0 else runtime.fps
        view_fps = runtime.ui_fps if runtime.ui_fps > 0.0 else runtime.fps
        if source_fps > 0.0:
            view_fps = min(view_fps, source_fps)
        ai_fps = runtime.infer_fps

        smooth_alpha = float(_clamp(float(self.runtime_cfg.get("fps_runtime_smoothing", 0.35)), 0.0, 1.0))
        if source_fps > 0.0:
            runtime.smoothed_source_fps = _ema(runtime.smoothed_source_fps, source_fps, smooth_alpha)
        if view_fps > 0.0:
            runtime.smoothed_view_fps = _ema(runtime.smoothed_view_fps, view_fps, smooth_alpha)
        if ai_fps > 0.0:
            runtime.smoothed_infer_fps = _ema(runtime.smoothed_infer_fps, ai_fps, smooth_alpha)

        source_fps = runtime.smoothed_source_fps or source_fps
        view_fps = runtime.smoothed_view_fps or view_fps
        ai_fps = runtime.smoothed_infer_fps or ai_fps
        display_update_sec = max(0.05, float(self.runtime_cfg.get("fps_display_update_sec", 0.15)))
        display_alpha = float(_clamp(float(self.runtime_cfg.get("fps_display_smoothing", 0.35)), 0.01, 1.0))
        display_quant_step = max(0.1, float(self.runtime_cfg.get("fps_display_quant_step", 0.1)))
        view_display_bias = float(_clamp(float(self.runtime_cfg.get("view_fps_display_bias", 1.0)), 0.8, 1.25))
        ai_display_bias = float(_clamp(float(self.runtime_cfg.get("ai_fps_display_bias", 1.0)), 0.8, 1.35))

        if runtime.last_meta_fps_update_ts <= 0.0 or (now - runtime.last_meta_fps_update_ts) >= display_update_sec:
            runtime.display_source_fps = _ema(runtime.display_source_fps, source_fps, display_alpha)
            runtime.display_view_fps = _ema(runtime.display_view_fps, view_fps * view_display_bias, display_alpha)
            runtime.display_infer_fps = _ema(runtime.display_infer_fps, ai_fps * ai_display_bias, display_alpha)
            runtime.last_meta_fps_update_ts = now

        shown_source_fps = _quantize_fps(runtime.display_source_fps or source_fps, display_quant_step)
        shown_view_fps = _quantize_fps(runtime.display_view_fps or view_fps, display_quant_step)
        shown_ai_fps = _quantize_fps(runtime.display_infer_fps or ai_fps, display_quant_step)
        mask_on = False
        source = runtime.source
        if source is not None:
            mask_on = bool(source.get("ignore_polys") or source.get("ignore_poly") or source.get("ignore_rect"))
        mask_text = " | mask:on" if mask_on else ""
        meta = (
            f"{runtime.status} | mode:{runtime.mode} | person:{runtime.person_count} | intruder:{runtime.intruder_count} "
            f"| src:{shown_source_fps:.1f} | view:{shown_view_fps:.1f} | ai:{shown_ai_fps:.1f}{mask_text}"
        )

        tile = self.tiles.get(source_name)
        fullscreen_active = self.focused_source == source_name and self._is_fullscreen_visible()
        if tile is not None and not fullscreen_active:
            tile_frame = runtime.last_output
            tile.set_alert_state(runtime.alert)
            tile.update_view(
                tile_frame,
                meta_text=meta,
                zoom=zoom,
                pan_x=pan_x,
                pan_y=pan_y,
            )

        if self.focused_source == source_name and self.fullscreen_window is not None and self.fullscreen_window.isVisible():
            self.fullscreen_window.set_source_name(source_name)
            fullscreen_frame = self._build_fullscreen_display_frame(source_name, runtime.last_output)
            self.fullscreen_window.set_frame(fullscreen_frame, zoom=zoom, pan_x=pan_x, pan_y=pan_y)

    def _tick_live(self) -> None:
        now_ts = time.perf_counter()
        self._update_load_shed_state(now_ts)
        self._maybe_adjust_live_timer_interval(now_ts)
        self._maybe_flush_debug_profile()

        if not self._apply_async_inference_updates():
            return

        enabled_sources = self._get_enabled_sources()
        if not enabled_sources:
            return

        infer_interval_sec = 1.0 / max(1.0, self._effective_model_target_fps())
        latest_frames: dict[str, np.ndarray] = {}
        source_names: list[str] = []

        for source in enabled_sources:
            source_name = str(source.get("name", "source"))
            runtime = self.runtimes.get(source_name)
            if runtime is None:
                continue
            source_names.append(source_name)

            frame, fresh_frame = self._read_frame(runtime)
            if frame is None:
                self._finalize_event_clip(source_name, runtime, time.perf_counter())
                runtime.last_output = None
                runtime.status = "no-frame"
                runtime.fps = 0.0
                runtime.person_count = 0
                runtime.intruder_count = 0
                runtime.detection_box_memory.clear()
                runtime.person_visible_since_ts = 0.0
                runtime.person_visible_duration_sec = 0.0
                runtime.event_saved_in_streak = False
                if runtime.no_frame_refresh_needed:
                    self._refresh_tile(source_name)
                    runtime.no_frame_refresh_needed = False
                continue
            runtime.no_frame_refresh_needed = True

            now = time.perf_counter()
            if fresh_frame and runtime.last_tick_ts > 0:
                delta = max(1e-6, now - runtime.last_tick_ts)
                runtime.fps = 1.0 / delta
            if fresh_frame:
                runtime.last_tick_ts = now

            latest_frames[source_name] = frame
            if fresh_frame:
                with self._infer_lock:
                    last_submit_ts = self._infer_last_submit_ts.get(source_name, 0.0)
                infer_due = last_submit_ts <= 0.0 or (now - last_submit_ts) >= infer_interval_sec
                if infer_due:
                    self._enqueue_inference_frame(source_name, frame, now)

        for source_name in source_names:
            runtime = self.runtimes.get(source_name)
            if runtime is None:
                continue

            frame = latest_frames.get(source_name)
            if frame is None:
                continue

            current_capture_seq = int(runtime.capture_last_consumed_seq)
            ai_changed = runtime.last_infer_ts > runtime.last_decorated_infer_ts
            frame_changed = current_capture_seq != runtime.last_decorated_capture_seq
            must_refresh = (
                runtime.last_output is None
                or frame_changed
                or ai_changed
            )
            if not must_refresh:
                continue

            boxes = runtime.last_boxes or []
            if runtime.last_infer_ts > 0:
                runtime.status = "alert" if runtime.alert else "ok"
            else:
                runtime.status = "live"
            event_now = time.perf_counter()
            self._update_event_visibility_state(source_name, runtime, event_now)
            runtime.last_output = self._decorate_live_frame(
                frame,
                source_name=source_name,
                mode=runtime.mode,
                person_count=runtime.person_count,
                intruder_count=runtime.intruder_count,
                alert=runtime.alert,
                boxes=boxes,
            )
            self._maybe_capture_event_snapshot(
                source_name=source_name,
                runtime=runtime,
                raw_frame=frame,
                decorated_frame=runtime.last_output,
            )
            self._store_event_prebuffer_frame(
                runtime,
                raw_frame=frame,
                decorated_frame=runtime.last_output,
            )
            runtime.last_decorated_capture_seq = current_capture_seq
            runtime.last_decorated_infer_ts = runtime.last_infer_ts

            self._refresh_tile(source_name)

    def start_live(self) -> None:
        if self.live_running:
            return

        self._flush_pending_settings()
        self._apply_controls_to_runtime_state()
        ensure_windows_compile_env(self.inference_cfg, compile_value=self.inference_cfg.get("compile", False))
        self._rebuild_predict_kwargs()
        with self._infer_lock:
            self._infer_last_submit_ts.clear()
        self._start_inference_worker()
        self._live_timer_interval_ms = self._compute_live_timer_interval_ms()
        self._live_timer_last_adjust_ts = time.perf_counter()
        self.live_timer.start(self._live_timer_interval_ms)
        self.live_running = True
        self._log(
            "Live inference started "
            f"(view_target_fps={self.view_target_fps:.1f}, timer={self._live_timer_interval_ms}ms)."
        )

    def stop_live(self) -> None:
        if not self.live_running:
            return

        self.live_timer.stop()
        self._stop_inference_worker()
        self.live_running = False
        self._live_timer_last_adjust_ts = 0.0
        stop_ts = time.perf_counter()
        for source_name, runtime in self.runtimes.items():
            self._finalize_event_clip(source_name, runtime, stop_ts)
            runtime.release()
        with self._infer_lock:
            self.trackers.clear()
            self._infer_last_submit_ts.clear()
            self._uniform_track_memory.clear()

        self._log("Live inference stopped.")

    # ---------- recordings ----------
    def _noop_click(self, _source_name: str) -> None:
        return

    def _browse_recording_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select recording",
            str(resolve_path("data/videos")),
            "Video files (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;All files (*.*)",
        )
        if file_path:
            self.recording_path_edit.setText(file_path)

    def _release_recording_capture(self) -> None:
        if self.recording_capture is not None:
            self.recording_capture.release()
            self.recording_capture = None

    def _load_recording_from_input(self) -> None:
        raw_path = self.recording_path_edit.text().strip()
        if not raw_path:
            QMessageBox.warning(self, "Recording", "Provide recording path first.")
            return

        path = resolve_path(raw_path)
        if not path.exists():
            QMessageBox.warning(self, "Recording", f"File does not exist:\n{path}")
            return

        self._open_recording(path)

    def _open_recording(self, path: Path) -> None:
        self._recording_pause()
        self._release_recording_capture()

        capture = open_video_file_capture(path)
        if not capture.isOpened():
            QMessageBox.warning(self, "Recording", f"Unable to open recording:\n{path}")
            return

        self.recording_capture = capture
        self.recording_frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.recording_fps = fps if fps > 1e-3 else 25.0
        self.recording_duration_sec = (
            float(self.recording_frame_count) / self.recording_fps if self.recording_frame_count > 0 else 0.0
        )
        self.recording_current_frame = 0
        self.recording_zoom = 1.0
        self.recording_pan_x = 0.0
        self.recording_pan_y = 0.0

        self.recording_slider_internal = True
        self.recording_slider.setRange(0, max(0, self.recording_frame_count - 1))
        self.recording_slider.setValue(0)
        self.recording_slider_internal = False

        self._seek_recording(0)

        self.runtime_cfg["last_recording_path"] = _to_relative_or_abs(path)
        self._persist_config(show_message=False)
        self._log(f"Recording loaded: {path}")

    def _seek_recording(self, frame_index: int) -> None:
        if self.recording_capture is None:
            return

        frame_index = int(_clamp(float(frame_index), 0.0, float(max(0, self.recording_frame_count - 1))))
        self.recording_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.recording_capture.read()
        if not ok or frame is None:
            return

        self.recording_current_frame = frame_index
        self.recording_canvas.set_frame(
            frame,
            zoom=self.recording_zoom,
            pan_x=self.recording_pan_x,
            pan_y=self.recording_pan_y,
        )

        self.recording_slider_internal = True
        self.recording_slider.setValue(frame_index)
        self.recording_slider_internal = False

        current_sec = float(frame_index) / self.recording_fps
        self.recording_time_label.setText(
            f"{_format_seconds(current_sec)} / {_format_seconds(self.recording_duration_sec)}"
        )

    def _recording_play(self) -> None:
        if self.recording_capture is None:
            self._load_recording_from_input()
            if self.recording_capture is None:
                return

        interval_ms = max(10, int(1000.0 / max(1.0, self.recording_fps)))
        self.recording_timer.start(interval_ms)
        self.recording_playing = True
        self._log("Recording playback started.")

    def _recording_pause(self) -> None:
        if not self.recording_playing:
            return
        self.recording_timer.stop()
        self.recording_playing = False
        self._log("Recording playback paused.")

    def _recording_stop(self) -> None:
        was_playing = self.recording_playing
        self.recording_timer.stop()
        self.recording_playing = False
        if self.recording_capture is not None:
            self._seek_recording(0)
        if was_playing:
            self._log("Recording playback stopped.")

    def _tick_recording(self) -> None:
        if self.recording_capture is None:
            self.recording_timer.stop()
            self.recording_playing = False
            return

        ok, frame = self.recording_capture.read()
        if not ok or frame is None:
            self._recording_stop()
            return

        current_frame = int(self.recording_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if current_frame < 0:
            current_frame = 0
        self.recording_current_frame = current_frame

        self.recording_canvas.set_frame(
            frame,
            zoom=self.recording_zoom,
            pan_x=self.recording_pan_x,
            pan_y=self.recording_pan_y,
        )

        self.recording_slider_internal = True
        self.recording_slider.setValue(current_frame)
        self.recording_slider_internal = False

        current_sec = float(current_frame) / self.recording_fps
        self.recording_time_label.setText(
            f"{_format_seconds(current_sec)} / {_format_seconds(self.recording_duration_sec)}"
        )

    def _on_recording_slider_pressed(self) -> None:
        self.recording_slider_user_drag = True

    def _on_recording_slider_released(self) -> None:
        self.recording_slider_user_drag = False
        self._seek_recording(self.recording_slider.value())

    def _on_recording_slider_changed(self, value: int) -> None:
        if self.recording_slider_internal:
            return
        if self.recording_capture is None:
            return
        if not self.recording_slider_user_drag:
            self._seek_recording(value)
        else:
            current_sec = float(value) / max(1.0, self.recording_fps)
            self.recording_time_label.setText(
                f"{_format_seconds(current_sec)} / {_format_seconds(self.recording_duration_sec)}"
            )

    def _change_recording_zoom(self, direction: int) -> None:
        scale = 1.12 if direction > 0 else (1.0 / 1.12)
        self.recording_zoom = _clamp(self.recording_zoom * scale, 1.0, 8.0)
        if self.recording_zoom <= 1.01:
            self.recording_pan_x = 0.0
            self.recording_pan_y = 0.0
        self._seek_recording(self.recording_current_frame)

    def _reset_recording_zoom(self) -> None:
        self.recording_zoom = 1.0
        self.recording_pan_x = 0.0
        self.recording_pan_y = 0.0
        self._seek_recording(self.recording_current_frame)

    def _on_recording_zoom_delta(self, _source_name: str, delta: int) -> None:
        self._change_recording_zoom(1 if delta > 0 else -1)

    def _on_recording_pan_delta(self, _source_name: str, dx: float, dy: float) -> None:
        if self.recording_zoom <= 1.01:
            return

        width = max(1, self.recording_canvas.width())
        height = max(1, self.recording_canvas.height())

        self.recording_pan_x = _clamp(self.recording_pan_x - (dx / float(width)) * (2.0 / self.recording_zoom), -1.0, 1.0)
        self.recording_pan_y = _clamp(self.recording_pan_y - (dy / float(height)) * (2.0 / self.recording_zoom), -1.0, 1.0)
        self._seek_recording(self.recording_current_frame)

    # ---------- events ----------
    def _load_event_entries(self) -> None:
        self.events_output_dir.mkdir(parents=True, exist_ok=True)
        self.event_entries = []

        if not self.events_index_path.exists():
            media_files: list[Path] = []
            for suffix in sorted(EVENT_IMAGE_SUFFIXES | EVENT_VIDEO_SUFFIXES):
                media_files.extend(self.events_output_dir.glob(f"*{suffix}"))
            for file_path in sorted(media_files, key=lambda p: p.stat().st_mtime):
                if not file_path.is_file():
                    continue
                self.event_entries.append(
                    {
                        "timestamp": float(file_path.stat().st_mtime),
                        "source": "source",
                        "mode": "day",
                        "persons": 0,
                        "intruders": 0,
                        "visible_sec": 0.0,
                        "alert": False,
                        "file": _to_relative_or_abs(file_path),
                    }
                )
            if self._enforce_event_retention_limit():
                self._save_event_entries_index()
            if hasattr(self, "events_period_combo"):
                self._refresh_events_filter_controls()
            if hasattr(self, "events_table"):
                self._refresh_events_table()
            return

        try:
            payload = json.loads(self.events_index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = {}

        raw_entries = []
        if isinstance(payload, dict):
            raw_entries = payload.get("events", [])
        elif isinstance(payload, list):
            raw_entries = payload

        loaded_entries: list[dict[str, Any]] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            file_value = str(raw.get("file", "")).strip()
            if not file_value:
                continue
            file_path = resolve_path(file_value)
            if not file_path.exists():
                continue
            timestamp = float(raw.get("timestamp", 0.0) or 0.0)
            if timestamp <= 0.0:
                timestamp = file_path.stat().st_mtime
            loaded_entries.append(
                {
                    "timestamp": timestamp,
                    "source": str(raw.get("source", "source")),
                    "mode": str(raw.get("mode", "day")),
                    "persons": int(raw.get("persons", 0) or 0),
                    "intruders": int(raw.get("intruders", raw.get("persons", 0)) or 0),
                    "visible_sec": float(raw.get("visible_sec", 0.0) or 0.0),
                    "alert": bool(raw.get("alert", False)),
                    "file": _to_relative_or_abs(file_path),
                }
            )

        loaded_entries.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        self.event_entries = loaded_entries
        if self._enforce_event_retention_limit():
            self._save_event_entries_index()
        if hasattr(self, "events_period_combo"):
            self._refresh_events_filter_controls()
        if hasattr(self, "events_table"):
            self._refresh_events_table()

    def _save_event_entries_index(self) -> None:
        self.events_output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "events": self.event_entries}
        self.events_index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _event_datetime(self, entry: dict[str, Any]) -> datetime | None:
        timestamp = float(entry.get("timestamp", 0.0) or 0.0)
        if timestamp <= 0.0:
            return None
        try:
            return datetime.fromtimestamp(timestamp)
        except Exception:  # noqa: BLE001
            return None

    def _event_period_options(self) -> list[tuple[str, dict[str, str]]]:
        now_date = datetime.now().date()
        seen_days: set[str] = set()
        seen_weeks: set[str] = set()
        seen_months: set[str] = set()
        seen_years: set[str] = set()
        day_items: list[tuple[str, dict[str, str]]] = []
        week_items: list[tuple[str, dict[str, str]]] = []
        month_items: list[tuple[str, dict[str, str]]] = []
        year_items: list[tuple[str, dict[str, str]]] = []

        ordered_entries = sorted(self.event_entries, key=lambda item: float(item.get("timestamp", 0.0) or 0.0), reverse=True)
        for entry in ordered_entries:
            dt = self._event_datetime(entry)
            if dt is None:
                continue
            event_date = dt.date()
            diff_days = (now_date - event_date).days
            iso_date = event_date.isoformat()
            if 0 <= diff_days <= 6:
                if iso_date in seen_days:
                    continue
                seen_days.add(iso_date)
                if diff_days == 0:
                    label = f"Dzisiaj ({iso_date})"
                elif diff_days == 1:
                    label = f"Wczoraj ({iso_date})"
                else:
                    label = f"{iso_date} ({diff_days} dni temu)"
                day_items.append((label, {"kind": "day", "value": iso_date}))
                continue

            iso_year, iso_week, _iso_weekday = dt.isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"
            if week_key not in seen_weeks:
                seen_weeks.add(week_key)
                week_items.append((f"Tydzien {week_key}", {"kind": "week", "value": week_key}))

            month_key = dt.strftime("%Y-%m")
            if month_key not in seen_months:
                seen_months.add(month_key)
                month_items.append((f"Miesiac {month_key}", {"kind": "month", "value": month_key}))

            year_key = dt.strftime("%Y")
            if year_key not in seen_years:
                seen_years.add(year_key)
                year_items.append((f"Rok {year_key}", {"kind": "year", "value": year_key}))

        items: list[tuple[str, dict[str, str]]] = [("Wszystkie zdarzenia", {"kind": "all", "value": "all"})]
        items.extend(day_items)
        items.extend(week_items)
        items.extend(month_items)
        items.extend(year_items)
        return items

    def _refresh_events_filter_controls(self) -> None:
        if not hasattr(self, "events_period_combo") or self.events_period_combo is None:
            return

        current_period = self.events_period_combo.currentData()
        current_camera = self.events_camera_combo.currentData() if hasattr(self, "events_camera_combo") else None

        self.events_period_combo.blockSignals(True)
        self.events_camera_combo.blockSignals(True)
        try:
            self.events_period_combo.clear()
            for label, payload in self._event_period_options():
                self.events_period_combo.addItem(label, payload)

            self.events_camera_combo.clear()
            self.events_camera_combo.addItem("Wszystkie kamery", "all")
            camera_names = {
                str(entry.get("source", "")).strip()
                for entry in self.event_entries
                if str(entry.get("source", "")).strip()
            }
            camera_names.update(
                str(source.get("name", "")).strip() for source in self.sources if str(source.get("name", "")).strip()
            )
            for camera_name in sorted(camera_names):
                self.events_camera_combo.addItem(camera_name, camera_name)

            period_index = self.events_period_combo.findData(current_period)
            self.events_period_combo.setCurrentIndex(0 if period_index < 0 else period_index)
            camera_index = self.events_camera_combo.findData(current_camera)
            self.events_camera_combo.setCurrentIndex(0 if camera_index < 0 else camera_index)
        finally:
            self.events_period_combo.blockSignals(False)
            self.events_camera_combo.blockSignals(False)

    def _event_matches_period(self, entry_dt: datetime, period_payload: dict[str, str] | None) -> bool:
        if not isinstance(period_payload, dict):
            return True
        kind = str(period_payload.get("kind", "all"))
        value = str(period_payload.get("value", "all"))
        if kind == "all":
            return True
        if kind == "day":
            return entry_dt.strftime("%Y-%m-%d") == value
        if kind == "week":
            iso_year, iso_week, _weekday = entry_dt.isocalendar()
            return f"{iso_year}-W{iso_week:02d}" == value
        if kind == "month":
            return entry_dt.strftime("%Y-%m") == value
        if kind == "year":
            return entry_dt.strftime("%Y") == value
        return True

    def _event_matches_hour_filter(self, entry_dt: datetime) -> bool:
        if not hasattr(self, "events_hour_filter_checkbox") or not self.events_hour_filter_checkbox.isChecked():
            return True
        from_time = self.events_hour_from_edit.time().toPyTime()
        to_time = self.events_hour_to_edit.time().toPyTime()
        entry_time = entry_dt.time()
        if from_time <= to_time:
            return from_time <= entry_time <= to_time
        return entry_time >= from_time or entry_time <= to_time

    def _event_group_header_label(self, entry_dt: datetime) -> str:
        weekday_names = {
            0: "Poniedzialek",
            1: "Wtorek",
            2: "Sroda",
            3: "Czwartek",
            4: "Piatek",
            5: "Sobota",
            6: "Niedziela",
        }
        month_names = {
            1: "Styczen",
            2: "Luty",
            3: "Marzec",
            4: "Kwiecien",
            5: "Maj",
            6: "Czerwiec",
            7: "Lipiec",
            8: "Sierpien",
            9: "Wrzesien",
            10: "Pazdziernik",
            11: "Listopad",
            12: "Grudzien",
        }

        now_date = datetime.now().date()
        event_date = entry_dt.date()
        diff_days = (now_date - event_date).days
        if diff_days == 0:
            return f"Dzisiaj - {event_date.strftime('%Y-%m-%d')}"
        if diff_days == 1:
            return f"Wczoraj - {event_date.strftime('%Y-%m-%d')}"
        if 0 <= diff_days <= 6:
            return f"{weekday_names.get(entry_dt.weekday(), event_date.strftime('%A'))} - {event_date.strftime('%Y-%m-%d')}"
        return month_names.get(entry_dt.month, entry_dt.strftime("%Y-%m"))

    def _event_group_key(self, entry_dt: datetime) -> str:
        now_date = datetime.now().date()
        event_date = entry_dt.date()
        diff_days = (now_date - event_date).days
        if 0 <= diff_days <= 6:
            return f"day:{event_date.isoformat()}"
        return f"month:{entry_dt.strftime('%Y-%m')}"

    def _filtered_event_entries(self) -> list[tuple[int, dict[str, Any]]]:
        selected_camera = str(self.events_camera_combo.currentData() or "all") if hasattr(self, "events_camera_combo") else "all"
        selected_mode = str(self.events_mode_combo.currentData() or "all") if hasattr(self, "events_mode_combo") else "all"
        period_payload = self.events_period_combo.currentData() if hasattr(self, "events_period_combo") else {"kind": "all", "value": "all"}
        exact_day_enabled = hasattr(self, "events_exact_day_checkbox") and self.events_exact_day_checkbox.isChecked()
        exact_day_value = self.events_exact_day_edit.date().toString("yyyy-MM-dd") if exact_day_enabled else ""

        filtered: list[tuple[int, dict[str, Any]]] = []
        ordered = list(enumerate(self.event_entries))
        ordered.reverse()
        for entry_index, entry in ordered:
            source_text = str(entry.get("source", "")).strip()
            if selected_camera not in {"", "all"} and source_text != selected_camera:
                continue

            mode_text = str(entry.get("mode", "day")).strip().lower()
            if selected_mode not in {"", "all"} and mode_text != selected_mode:
                continue

            dt = self._event_datetime(entry)
            if dt is None:
                continue

            if exact_day_enabled and dt.strftime("%Y-%m-%d") != exact_day_value:
                continue
            if not self._event_matches_period(dt, period_payload):
                continue
            if not self._event_matches_hour_filter(dt):
                continue
            filtered.append((entry_index, entry))
        return filtered

    def _schedule_events_table_refresh(self, *_args: Any) -> None:
        if not hasattr(self, "_events_filter_timer") or self._events_filter_timer is None:
            self._refresh_events_table()
            return
        self._events_filter_timer.start(120)

    def _enforce_event_retention_limit(self) -> bool:
        max_saved = int(self.events_max_saved)
        if max_saved <= 0:
            return False
        excess = len(self.event_entries) - max_saved
        if excess <= 0:
            return False

        removed = self.event_entries[:excess]
        self.event_entries = self.event_entries[excess:]
        for entry in removed:
            file_value = str(entry.get("file", "")).strip()
            if not file_value:
                continue
            file_path = resolve_path(file_value)
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:  # noqa: BLE001
                pass
        return True

    def _refresh_events_table(self, *_args: Any, select_newest: bool = False) -> None:
        if not hasattr(self, "events_table"):
            return
        if hasattr(self, "_events_filter_timer") and self._events_filter_timer.isActive():
            self._events_filter_timer.stop()

        self.events_table.blockSignals(True)
        self._event_table_updating = True
        try:
            ordered = self._filtered_event_entries()
            self.events_table.clearSpans()
            self.events_table.clearSelection()
            self.events_table.setCurrentCell(-1, -1)
            rows_to_render: list[tuple[str, Any]] = []
            previous_group_key = ""
            for entry_index, entry in ordered:
                dt = self._event_datetime(entry)
                if dt is None:
                    continue
                group_key = self._event_group_key(dt)
                if group_key != previous_group_key:
                    rows_to_render.append(("group", self._event_group_header_label(dt)))
                    previous_group_key = group_key
                rows_to_render.append(("event", (entry_index, entry)))

            self.events_table.setRowCount(len(rows_to_render))
            for row, (row_kind, payload) in enumerate(rows_to_render):
                if row_kind == "group":
                    header_item = QTableWidgetItem(str(payload))
                    header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    header_item.setForeground(QColor("#d9e7ff"))
                    header_item.setBackground(QColor("#243041"))
                    header_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    header_font = header_item.font()
                    header_font.setPointSize(max(12, header_font.pointSize() + 2))
                    header_font.setBold(True)
                    header_item.setFont(header_font)
                    self.events_table.setSpan(row, 0, 1, self.events_table.columnCount())
                    self.events_table.setItem(row, 0, header_item)
                    for col in range(1, self.events_table.columnCount()):
                        filler_item = QTableWidgetItem("")
                        filler_item.setFlags(Qt.ItemFlag.NoItemFlags)
                        filler_item.setBackground(QColor("#243041"))
                        self.events_table.setItem(row, col, filler_item)
                    continue

                entry_index, entry = payload
                dt = self._event_datetime(entry)
                if dt is None:
                    date_text = "-"
                    hour_text = "-"
                else:
                    date_text = dt.strftime("%Y-%m-%d")
                    hour_text = dt.strftime("%H:%M:%S")
                source_text = str(entry.get("source", "source"))
                mode_raw = str(entry.get("mode", "day")).strip().lower()
                mode_text = "Dzien" if mode_raw == "day" else "Noc" if mode_raw == "night" else mode_raw
                persons_text = str(int(entry.get("persons", 0) or 0))
                visible_text = f"{float(entry.get('visible_sec', 0.0) or 0.0):.1f}"
                file_text = str(entry.get("file", ""))

                values = [date_text, hour_text, source_text, mode_text, persons_text, visible_text, file_text]
                for col, text in enumerate(values):
                    display_text = f"   {text}" if col < 6 else text
                    item = QTableWidgetItem(display_text)
                    item.setData(Qt.ItemDataRole.UserRole, int(entry_index))
                    if row % 2 == 0:
                        item.setBackground(QColor("#141b25"))
                    else:
                        item.setBackground(QColor("#101722"))
                    item.setForeground(QColor("#edf2fb"))
                    self.events_table.setItem(row, col, item)

            if hasattr(self, "events_status_label"):
                self.events_status_label.setText(
                    f"Zdarzenia: {len(ordered)} / {len(self.event_entries)}"
                )
        finally:
            self._event_table_updating = False
            self.events_table.blockSignals(False)

        if select_newest and self.events_table.rowCount() > 0:
            for row in range(self.events_table.rowCount()):
                item = self.events_table.item(row, 0)
                if item is not None and item.data(Qt.ItemDataRole.UserRole) is not None:
                    self.events_table.setCurrentCell(row, 0)
                    break
        elif self.events_table.rowCount() > 0:
            self.events_preview.set_frame(None)
            self.events_preview.setText("Wybierz zdarzenie z listy, aby zaladowac podglad.")
        elif self.events_table.rowCount() <= 0:
            self.events_preview.set_frame(None)
            self.events_preview.setText("Brak zapisanych zdarzen.")

    def _get_selected_event_entry(self) -> dict[str, Any] | None:
        if not hasattr(self, "events_table"):
            return None
        row = self.events_table.currentRow()
        if row < 0:
            return None
        item = self.events_table.item(row, 0)
        if item is None:
            return None
        index_value = item.data(Qt.ItemDataRole.UserRole)
        if index_value is None:
            return None
        try:
            entry_index = int(index_value)
        except Exception:  # noqa: BLE001
            return None
        if entry_index < 0 or entry_index >= len(self.event_entries):
            return None
        return self.event_entries[entry_index]

    def _on_event_table_selection_changed(self) -> None:
        if self._event_table_updating:
            return
        entry = self._get_selected_event_entry()
        if entry is None:
            self.events_preview.set_frame(None)
            self.events_preview.setText("Brak wybranego zdarzenia.")
            return

        file_path = resolve_path(str(entry.get("file", "")))
        if not file_path.exists():
            self.events_preview.set_frame(None)
            self.events_preview.setText("Plik zdarzenia nie istnieje.")
            return

        dt = self._event_datetime(entry)
        if dt is not None:
            mode_raw = str(entry.get("mode", "day")).strip().lower()
            mode_text = "dzien" if mode_raw == "day" else "noc" if mode_raw == "night" else mode_raw
            self.events_preview.setText(
                f"Kamera: {entry.get('source', 'source')}\n"
                f"Tryb: {mode_text}\n"
                f"Czas: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        frame: np.ndarray | None = None
        suffix = file_path.suffix.lower()
        if suffix in EVENT_IMAGE_SUFFIXES:
            try:
                raw_data = np.fromfile(str(file_path), dtype=np.uint8)
                frame = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001
                frame = None
        else:
            capture = open_video_file_capture(file_path)
            try:
                if capture.isOpened():
                    ok, first_frame = capture.read()
                    if ok and first_frame is not None:
                        frame = first_frame
            except Exception:  # noqa: BLE001
                frame = None
            finally:
                try:
                    capture.release()
                except Exception:  # noqa: BLE001
                    pass
        if frame is None:
            self.events_preview.set_frame(None)
            self.events_preview.setText("Nie mozna wczytac pliku zdarzenia.")
            return
        self.events_preview.set_frame(frame)

    def _open_selected_event_file(self) -> None:
        entry = self._get_selected_event_entry()
        if entry is None:
            QMessageBox.information(self, "Events", "Select event first.")
            return

        file_path = resolve_path(str(entry.get("file", "")))
        if not file_path.exists():
            QMessageBox.warning(self, "Events", f"File does not exist:\n{file_path}")
            return

        try:
            if hasattr(os, "startfile"):
                os.startfile(str(file_path))  # type: ignore[attr-defined]
            else:
                QMessageBox.information(self, "Events", f"File path:\n{file_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Events", f"Unable to open file:\n{exc}")

    def _clear_all_events(self) -> None:
        if not self.event_entries and not self.events_output_dir.exists():
            QMessageBox.information(self, "Events", "Brak zapisanych zdarzen.")
            return

        reply = QMessageBox.question(
            self,
            "Events",
            "Usunac wszystkie zapisane zdarzenia?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for entry in list(self.event_entries):
            file_value = str(entry.get("file", "")).strip()
            if not file_value:
                continue
            file_path = resolve_path(file_value)
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:  # noqa: BLE001
                pass

        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.m4v", ".tmp_*"):
            for file_path in self.events_output_dir.glob(pattern):
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception:  # noqa: BLE001
                    pass

        self.event_entries.clear()
        try:
            if self.events_index_path.exists():
                self.events_index_path.unlink()
        except Exception:  # noqa: BLE001
            pass
        if hasattr(self, "events_period_combo"):
            self._refresh_events_filter_controls()
        self._refresh_events_table()
        self._log("All saved events were removed.")

    def _start_event_writer_thread(self) -> None:
        if self._event_writer_thread is not None and self._event_writer_thread.is_alive():
            return
        with self._event_writer_cond:
            self._event_writer_stop = False
            self._event_writer_queue.clear()
            self._event_writer_latest_frames.clear()
            self._event_writer_prebuffer_frames.clear()
            self._event_writer_queued_keys.clear()
            self._event_writer_inflight_keys.clear()
        self._event_writer_thread = threading.Thread(
            target=self._event_writer_loop,
            name="event-clip-writer",
            daemon=True,
        )
        self._event_writer_thread.start()

    def _event_writer_schedule_key_locked(self, key: tuple[str, int]) -> None:
        if key in self._event_writer_queued_keys:
            return
        self._event_writer_queue.append(key)
        self._event_writer_queued_keys.add(key)

    def _stop_event_writer_thread(self) -> None:
        with self._event_writer_cond:
            self._event_writer_stop = True
            self._event_writer_cond.notify_all()
        worker = self._event_writer_thread
        self._event_writer_thread = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.5)
        with self._event_writer_cond:
            self._event_writer_queue.clear()
            self._event_writer_latest_frames.clear()
            self._event_writer_prebuffer_frames.clear()
            self._event_writer_queued_keys.clear()
            self._event_writer_inflight_keys.clear()

    def _event_writer_loop(self) -> None:
        while True:
            with self._event_writer_cond:
                while not self._event_writer_stop and not self._event_writer_queue:
                    self._event_writer_cond.wait(timeout=0.2)
                if self._event_writer_stop and not self._event_writer_queue:
                    return
                source_name, generation = self._event_writer_queue.popleft()
                key = (source_name, generation)
                self._event_writer_queued_keys.discard(key)
                prebuffer = self._event_writer_prebuffer_frames.get(key)
                frame = None
                if prebuffer:
                    frame = prebuffer.popleft()
                    if not prebuffer:
                        self._event_writer_prebuffer_frames.pop(key, None)
                if frame is None:
                    frame = self._event_writer_latest_frames.pop(key, None)
                if frame is None:
                    continue
                self._event_writer_inflight_keys.add(key)

            runtime = self.runtimes.get(source_name)
            if runtime is not None and runtime.event_clip_generation == generation and runtime.event_clip_writer is not None:
                target_size = runtime.event_clip_frame_size or (frame.shape[1], frame.shape[0])
                frame_to_write = frame
                if frame_to_write.shape[1] != target_size[0] or frame_to_write.shape[0] != target_size[1]:
                    frame_to_write = cv2.resize(frame_to_write, target_size, interpolation=cv2.INTER_LINEAR)
                try:
                    runtime.event_clip_writer.write(frame_to_write)
                    runtime.event_clip_frames_written += 1
                except Exception:  # noqa: BLE001
                    runtime.event_clip_failed = True
                    self._queue_async_notice(
                        f"[warn] Blad zapisu klatki klipu zdarzenia dla '{source_name}'. "
                        "Sprobuj zmniejszyc liczbe klatek (model_target_fps/view_target_fps) "
                        "lub liczbe aktywnych zrodel."
                    )

            with self._event_writer_cond:
                self._event_writer_inflight_keys.discard(key)
                if key in self._event_writer_latest_frames or key in self._event_writer_prebuffer_frames:
                    self._event_writer_schedule_key_locked(key)
                self._event_writer_cond.notify_all()

    def _enqueue_event_clip_frame(self, source_name: str, runtime: SourceRuntime, frame: np.ndarray) -> None:
        generation = int(runtime.event_clip_generation)
        key = (source_name, generation)
        with self._event_writer_cond:
            # Keep only the newest pending frame per clip so the writer never falls
            # behind by encoding stale history when the UI/inference outruns disk/codec.
            self._event_writer_latest_frames[key] = frame.copy()
            self._event_writer_schedule_key_locked(key)
            self._event_writer_cond.notify()

    def _wait_for_event_clip_flush(self, source_name: str, generation: int, timeout_sec: float = 0.35) -> None:
        key = (source_name, int(generation))
        deadline = time.perf_counter() + max(0.01, timeout_sec)
        with self._event_writer_cond:
            while (
                key in self._event_writer_inflight_keys
                or key in self._event_writer_latest_frames
                or key in self._event_writer_prebuffer_frames
                or key in self._event_writer_queued_keys
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self._event_writer_cond.wait(timeout=min(0.05, remaining))

    def _trim_event_prebuffer_locked(self, runtime: SourceRuntime, now_ts: float) -> None:
        keep_sec = max(0.0, float(self.events_prebuffer_seconds))
        if keep_sec <= 0.0:
            runtime.event_prebuffer_frames.clear()
            return
        cutoff_ts = now_ts - keep_sec
        while runtime.event_prebuffer_frames and float(runtime.event_prebuffer_frames[0][0]) < cutoff_ts:
            runtime.event_prebuffer_frames.popleft()

    def _store_event_prebuffer_frame(
        self,
        runtime: SourceRuntime,
        raw_frame: np.ndarray,
        decorated_frame: np.ndarray | None,
    ) -> None:
        keep_sec = max(0.0, float(self.events_prebuffer_seconds))
        if keep_sec <= 0.0:
            runtime.event_prebuffer_frames.clear()
            runtime.event_prebuffer_last_store_ts = 0.0
            return

        now_ts = time.perf_counter()
        sample_interval_sec = 1.0 / max(1.0, float(self.events_clip_fps))
        if runtime.event_prebuffer_last_store_ts > 0.0 and (now_ts - runtime.event_prebuffer_last_store_ts) < sample_interval_sec:
            self._trim_event_prebuffer_locked(runtime, now_ts)
            return

        frame_to_store = decorated_frame if self.events_save_annotated and decorated_frame is not None else raw_frame
        runtime.event_prebuffer_frames.append((now_ts, frame_to_store.copy()))
        runtime.event_prebuffer_last_store_ts = now_ts
        self._trim_event_prebuffer_locked(runtime, now_ts)

    def _prime_event_prebuffer_for_clip(self, source_name: str, runtime: SourceRuntime) -> None:
        keep_sec = max(0.0, float(self.events_prebuffer_seconds))
        if keep_sec <= 0.0 or not runtime.event_prebuffer_frames:
            return

        key = (source_name, int(runtime.event_clip_generation))
        with self._event_writer_cond:
            if key in self._event_writer_prebuffer_frames:
                return
            buffered_frames = deque(frame.copy() for _ts, frame in runtime.event_prebuffer_frames)
            if not buffered_frames:
                return
            self._event_writer_prebuffer_frames[key] = buffered_frames
            self._event_writer_schedule_key_locked(key)
            self._event_writer_cond.notify()

    def _clear_event_clip_state(self, runtime: SourceRuntime, *, delete_temp_file: bool) -> Path | None:
        if runtime.event_clip_writer is not None:
            try:
                runtime.event_clip_writer.release()
            except Exception:  # noqa: BLE001
                pass
        runtime.event_clip_writer = None

        temp_path = runtime.event_clip_temp_path
        runtime.event_clip_temp_path = None
        if delete_temp_file and temp_path is not None:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:  # noqa: BLE001
                pass

        runtime.event_clip_frame_size = None
        runtime.event_clip_frames_written = 0
        runtime.event_clip_last_enqueue_ts = 0.0
        runtime.event_clip_started_wall_ts = 0.0
        runtime.event_clip_generation += 1
        runtime.event_clip_failed = False
        return temp_path

    def _finalize_event_clip(self, source_name: str, runtime: SourceRuntime, now_ts: float) -> None:
        has_pending_clip = (
            runtime.event_clip_writer is not None
            or runtime.event_clip_temp_path is not None
            or runtime.event_clip_frames_written > 0
        )
        if not has_pending_clip:
            return

        current_generation = int(runtime.event_clip_generation)
        self._wait_for_event_clip_flush(source_name, current_generation)

        frames_written = int(runtime.event_clip_frames_written)
        wall_ts = runtime.event_clip_started_wall_ts if runtime.event_clip_started_wall_ts > 0.0 else time.time()
        visible_sec = float(runtime.person_visible_duration_sec)
        max_person_count = int(runtime.event_max_person_count)
        max_intruder_count = int(runtime.event_max_intruder_count)
        failed = bool(runtime.event_clip_failed)
        temp_path = self._clear_event_clip_state(runtime, delete_temp_file=False)
        if temp_path is None:
            return

        cooldown_ok = (
            runtime.last_event_capture_ts <= 0.0
            or self.events_cooldown_seconds <= 0.0
            or (now_ts - runtime.last_event_capture_ts) >= self.events_cooldown_seconds
        )
        should_save = (
            self.events_enabled
            and not failed
            and frames_written > 0
            and temp_path.exists()
            and visible_sec >= self.events_min_visible_seconds
            and cooldown_ok
            and (not self.events_once_per_streak or not runtime.event_saved_in_streak)
        )

        if not should_save:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:  # noqa: BLE001
                pass
            return

        safe_source = _safe_file_part(source_name, fallback="source")
        timestamp_text = time.strftime("%Y%m%d_%H%M%S", time.localtime(wall_ts))
        millis = int((wall_ts - int(wall_ts)) * 1000.0)
        suffix = temp_path.suffix.lower() if temp_path.suffix else ".mp4"
        output_path = self.events_output_dir / f"{timestamp_text}_{millis:03d}_{safe_source}{suffix}"
        collision_index = 2
        while output_path.exists():
            output_path = self.events_output_dir / f"{timestamp_text}_{millis:03d}_{safe_source}_{collision_index}{suffix}"
            collision_index += 1

        moved = False
        try:
            temp_path.replace(output_path)
            moved = True
        except Exception:  # noqa: BLE001
            moved = False

        if not moved:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:  # noqa: BLE001
                pass
            self._queue_async_notice(
                f"[warn] Nie udalo sie domknac i zapisac klipu zdarzenia dla '{source_name}'."
            )
            return

        entry = {
            "timestamp": float(wall_ts),
            "source": source_name,
            "mode": runtime.mode,
            "persons": max_person_count,
            "intruders": max_intruder_count,
            "visible_sec": visible_sec,
            "alert": bool(runtime.alert),
            "file": _to_relative_or_abs(output_path),
        }
        self.event_entries.append(entry)
        self._enforce_event_retention_limit()
        self._save_event_entries_index()

        runtime.last_event_capture_ts = now_ts
        runtime.event_saved_in_streak = True
        runtime.event_last_seen_ts = 0.0
        runtime.event_max_person_count = 0
        runtime.event_max_intruder_count = 0

        show_latest = bool(
            hasattr(self, "main_tabs")
            and hasattr(self, "events_tab_page")
            and self.main_tabs.currentWidget() is self.events_tab_page
        )
        self._refresh_events_table(select_newest=show_latest)
        self._log(
            f"Event saved: {source_name}, visible={visible_sec:.1f}s, "
            f"persons={max_person_count}, intruders={max_intruder_count}, file={output_path}"
        )

    def _ensure_event_clip_writer(self, source_name: str, runtime: SourceRuntime, frame: np.ndarray) -> bool:
        if runtime.event_clip_writer is not None and runtime.event_clip_temp_path is not None:
            return True

        self.events_output_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        frame_size = (int(width), int(height))
        clip_fps = max(1.0, float(self.events_clip_fps))

        wall_ts = runtime.event_clip_started_wall_ts if runtime.event_clip_started_wall_ts > 0.0 else time.time()
        runtime.event_clip_started_wall_ts = wall_ts
        safe_source = _safe_file_part(source_name, fallback="source")
        timestamp_text = time.strftime("%Y%m%d_%H%M%S", time.localtime(wall_ts))
        millis = int((wall_ts - int(wall_ts)) * 1000.0)
        temp_base = self.events_output_dir / (
            f".tmp_{timestamp_text}_{millis:03d}_{safe_source}_{os.getpid()}_{id(runtime)}"
        )

        writer, temp_path = _open_event_video_writer(temp_base, fps=clip_fps, frame_size=frame_size)
        if writer is None or temp_path is None:
            self._queue_async_notice(
                f"[warn] Nie udalo sie uruchomic zapisu klipu zdarzenia dla '{source_name}'. "
                "Sprawdz uprawnienia do folderu zapisu i obciazenie dysku."
            )
            self._clear_event_clip_state(runtime, delete_temp_file=True)
            return False

        runtime.event_clip_writer = writer
        runtime.event_clip_temp_path = temp_path
        runtime.event_clip_frame_size = frame_size
        runtime.event_clip_frames_written = 0
        runtime.event_clip_last_enqueue_ts = 0.0
        runtime.event_clip_generation += 1
        runtime.event_clip_failed = False
        self._prime_event_prebuffer_for_clip(source_name, runtime)
        return True

    def _update_event_visibility_state(self, source_name: str, runtime: SourceRuntime, now_ts: float) -> None:
        visible_count = runtime.intruder_count if runtime.mode == "day" else runtime.person_count
        suspicious_present = visible_count >= self.events_min_person_count and runtime.last_infer_ts > 0
        if suspicious_present:
            if runtime.person_visible_since_ts <= 0.0:
                runtime.person_visible_since_ts = now_ts
                runtime.person_visible_duration_sec = 0.0
                runtime.event_saved_in_streak = False
                runtime.event_clip_started_wall_ts = time.time()
                runtime.event_last_seen_ts = now_ts
                runtime.event_max_person_count = int(visible_count)
                runtime.event_max_intruder_count = int(runtime.intruder_count)
            else:
                runtime.person_visible_duration_sec = max(0.0, now_ts - runtime.person_visible_since_ts)
                runtime.event_last_seen_ts = now_ts
                runtime.event_max_person_count = max(runtime.event_max_person_count, int(visible_count))
                runtime.event_max_intruder_count = max(runtime.event_max_intruder_count, int(runtime.intruder_count))
            return

        if runtime.person_visible_since_ts > 0.0:
            runtime.person_visible_duration_sec = max(
                0.0,
                (runtime.event_last_seen_ts or now_ts) - runtime.person_visible_since_ts,
            )
            linger_ok = bool(self.events_linger_seconds > 0.0) and (
                now_ts - (runtime.event_last_seen_ts or now_ts) <= self.events_linger_seconds
            )
            if linger_ok:
                return
            self._finalize_event_clip(source_name, runtime, now_ts)
        else:
            self._clear_event_clip_state(runtime, delete_temp_file=True)

        runtime.person_visible_since_ts = 0.0
        runtime.person_visible_duration_sec = 0.0
        runtime.event_saved_in_streak = False
        runtime.event_last_seen_ts = 0.0
        runtime.event_max_person_count = 0
        runtime.event_max_intruder_count = 0

    def _maybe_capture_event_snapshot(
        self,
        source_name: str,
        runtime: SourceRuntime,
        raw_frame: np.ndarray,
        decorated_frame: np.ndarray | None,
    ) -> None:
        if not self.events_enabled:
            self._clear_event_clip_state(runtime, delete_temp_file=True)
            return
        if runtime.person_visible_since_ts <= 0.0:
            return

        clip_frame = decorated_frame if self.events_save_annotated and decorated_frame is not None else raw_frame
        if clip_frame is None:
            return

        now_ts = time.perf_counter()
        clip_interval_sec = 1.0 / max(1.0, float(self.events_clip_fps))
        if runtime.event_clip_last_enqueue_ts > 0.0 and (now_ts - runtime.event_clip_last_enqueue_ts) < clip_interval_sec:
            return

        if not self._ensure_event_clip_writer(source_name, runtime, clip_frame):
            return

        if runtime.event_clip_writer is None:
            return
        runtime.event_clip_last_enqueue_ts = now_ts
        self._enqueue_event_clip_frame(source_name, runtime, clip_frame)

    # ---------- logs ----------
    def _clear_logs(self) -> None:
        self._log_entries.clear()
        self._pending_log_lines.clear()
        if self._log_flush_timer.isActive():
            self._log_flush_timer.stop()
        self.logs_text.clear()
        self._log("Logs cleared.")

    def _export_logs(self) -> None:
        if not self._log_entries:
            QMessageBox.information(self, "Logs", "No logs to export.")
            return

        logs_dir = resolve_path("logs/app/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        output_path = logs_dir / f"app_{time.strftime('%Y%m%d_%H%M%S')}.log"

        try:
            output_path.write_text("\n".join(self._log_entries) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Logs", f"Failed to export logs:\n{exc}")
            return

        QMessageBox.information(self, "Logs", f"Logs exported:\n{output_path}")

    # ---------- config persistence ----------
    def _apply_controls_to_runtime_state(self) -> None:
        self.security_cfg["mode"] = self.security_mode_combo.currentText().strip().lower()
        self.security_cfg["night_start_hour"] = int(self.night_start_spin.value())
        self.security_cfg["night_end_hour"] = int(self.night_end_spin.value())
        self.security_cfg["day_person_threshold"] = int(self.day_threshold_spin.value())
        self.security_cfg["night_person_threshold"] = int(self.night_threshold_spin.value())

        selected_profile_key = str(self.yolo_profile_combo.currentData() or self._infer_yolo_profile_key()).strip().lower()
        if selected_profile_key in YOLO_PROFILE_PRESETS or selected_profile_key == YOLO_PROFILE_CUSTOM:
            self.runtime_cfg["yolo_profile"] = selected_profile_key

        self.inference_cfg["conf"] = float(self.conf_spin.value())
        self.inference_cfg["iou"] = float(self.iou_spin.value())
        self.inference_cfg["imgsz"] = int(self._current_imgsz_value())
        self.inference_cfg["max_det"] = int(self.max_det_spin.value())
        current_model_selection = self._current_yolo_model_selection()
        selected_model_name = current_model_selection["model_name"]
        selected_model_path = current_model_selection["selected_model_path"]
        if selected_model_name:
            self.model_cfg["name"] = selected_model_name
        self.model_cfg["selected_model_path"] = selected_model_path

        day_seg_selection = self._current_day_seg_model_selection()
        day_seg_cfg = self._day_segmentation_model_cfg()
        if day_seg_selection["model_name"]:
            day_seg_cfg["name"] = day_seg_selection["model_name"]
        day_seg_cfg["selected_model_path"] = day_seg_selection["selected_model_path"]
        day_seg_cfg["enabled"] = bool(self.uniform_enabled_checkbox.isChecked())
        self.model_cfg["day_segmentation"] = dict(day_seg_cfg)

        raw_device = self.device_edit.text().strip()
        if raw_device.lower() in {"", "auto", "none"}:
            self.inference_cfg["device"] = "auto"
        else:
            self.inference_cfg["device"] = raw_device

        self.inference_cfg["half"] = bool(self.half_checkbox.isChecked())
        self.inference_cfg["compile"] = bool(self.compile_checkbox.isChecked())
        self.compile_enabled = bool(self.compile_checkbox.isChecked())

        self.runtime_cfg["start_maximized"] = bool(self.start_maximized_checkbox.isChecked())
        self.runtime_cfg["loop_videos"] = bool(self.loop_videos)
        self.runtime_cfg["frame_interval_ms"] = int(self.frame_interval_ms)
        self.runtime_cfg["view_target_fps"] = float(self.view_target_fps)
        self.model_target_fps = int(self.model_target_fps_spin.value())
        self.runtime_cfg["model_target_fps"] = int(self.model_target_fps)
        self.runtime_cfg["max_infer_per_tick"] = int(self.max_infer_per_tick)
        self.runtime_cfg["live_tile_spacing"] = int(self.live_tile_spacing)
        self.runtime_cfg["show_live_tile_headers"] = bool(self.live_tile_header_visible)
        self.runtime_cfg["show_navigation_tabs"] = bool(self.navigation_tabs_visible)
        self.runtime_cfg["console_logs"] = bool(self.console_logs_enabled)
        self.runtime_cfg["suppress_opencv_warnings"] = bool(self.suppress_opencv_warnings)
        self.runtime_cfg["camera_backend"] = str(getattr(self, "camera_backend", self.runtime_cfg.get("camera_backend", "msmf")))
        self.runtime_cfg["detection_box_hold_sec"] = float(self._detection_box_hold_sec)
        self.runtime_cfg["auto_scan_cameras_on_startup"] = bool(self.auto_scan_cameras_on_startup)
        self.runtime_cfg["auto_start_live"] = bool(self.auto_start_live)

        self.uniform_cfg["enabled"] = bool(self.uniform_enabled_checkbox.isChecked())
        self.uniform_cfg["top_color"] = _normalize_hex_color(
            getattr(self, "_selected_uniform_top_color", self.uniform_cfg.get("top_color", UNIFORM_TOP_DEFAULT)),
            UNIFORM_TOP_DEFAULT,
        )
        self.uniform_cfg["bottom_color"] = _normalize_hex_color(
            getattr(self, "_selected_uniform_bottom_color", self.uniform_cfg.get("bottom_color", UNIFORM_BOTTOM_DEFAULT)),
            UNIFORM_BOTTOM_DEFAULT,
        )
        self.uniform_cfg["color_tolerance"] = _uniform_color_tolerance(self.uniform_tolerance_spin.value())
        self.uniform_cfg["min_mask_pixels"] = int(self.uniform_min_pixels_spin.value())

        previous_output_dir = self.events_output_dir
        previous_output_raw = self.events_output_dir_raw

        self.events_cfg["enabled"] = bool(self.events_enabled_checkbox.isChecked())
        self.events_cfg["min_visible_seconds"] = float(self.events_min_visible_spin.value())
        self.events_cfg["cooldown_seconds"] = float(self.events_cooldown_spin.value())
        self.events_cfg["linger_seconds"] = float(self.events_linger_spin.value())
        self.events_cfg["min_person_count"] = int(self.events_min_person_spin.value())
        self.events_cfg["clip_fps"] = int(self.events_clip_fps_spin.value())
        self.events_cfg["prebuffer_seconds"] = float(self.events_prebuffer_spin.value())
        self.events_cfg["max_saved_events"] = int(self.events_max_saved_spin.value())
        self.events_cfg["save_annotated_frame"] = bool(self.events_save_annotated_checkbox.isChecked())
        self.events_cfg["once_per_streak"] = bool(self.events_once_per_streak_checkbox.isChecked())
        output_dir_raw = self.events_output_dir_edit.text().strip() or "logs/app/events"
        self.events_cfg["output_dir"] = output_dir_raw

        self.events_enabled = bool(self.events_cfg["enabled"])
        self.events_min_visible_seconds = max(0.1, float(self.events_cfg["min_visible_seconds"]))
        self.events_cooldown_seconds = max(0.0, float(self.events_cfg["cooldown_seconds"]))
        self.events_linger_seconds = max(0.0, float(self.events_cfg["linger_seconds"]))
        self.events_min_person_count = max(1, int(self.events_cfg["min_person_count"]))
        self.events_clip_fps = max(1.0, float(self.events_cfg["clip_fps"]))
        self.events_prebuffer_seconds = max(0.0, float(self.events_cfg["prebuffer_seconds"]))
        self.events_max_saved = max(0, int(self.events_cfg["max_saved_events"]))
        self.events_save_annotated = bool(self.events_cfg["save_annotated_frame"])
        self.events_once_per_streak = bool(self.events_cfg["once_per_streak"])
        self.events_output_dir_raw = output_dir_raw
        self.events_output_dir = resolve_path(output_dir_raw)
        self.events_index_path = self.events_output_dir / "events_index.json"

        output_changed = (
            previous_output_raw != self.events_output_dir_raw
            or previous_output_dir.resolve() != self.events_output_dir.resolve()
        )
        if output_changed:
            self._load_event_entries()
        else:
            if self._enforce_event_retention_limit():
                self._save_event_entries_index()
            if hasattr(self, "events_table"):
                self._refresh_events_table()

    def _persist_config(
        self,
        *,
        show_message: bool,
        previous_settings: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if previous_settings is None:
            previous_settings = self._snapshot_settings_state()
            self._apply_controls_to_runtime_state()
        current_settings = self._snapshot_settings_state()
        setting_changes = self._collect_setting_changes(previous_settings, current_settings)

        self.config["model"] = dict(self.model_cfg)
        self.config["inference"] = dict(self.inference_cfg)
        self.config["tracker"] = dict(self.tracker_cfg)
        self.config["security"] = dict(self.security_cfg)
        self.config["uniform"] = dict(self.uniform_cfg)
        self.config["events"] = dict(self.events_cfg)
        self.config["runtime"] = dict(self.runtime_cfg)
        self.config["debug"] = dict(self.debug_cfg)
        self.config.pop("sources", None)

        config_signature = json.dumps(self.config, sort_keys=True, ensure_ascii=True, default=str)
        config_changed = config_signature != self._last_saved_config_signature

        self.app_settings_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(self.app_config_path, self.config)
        try:
            self._save_sources_config()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[warn] Unable to save sources config: {exc}")

        self._last_saved_config_signature = config_signature
        if config_changed:
            self._log(
                "Config changed and saved: "
                f"{self.app_config_path} (sources: {self.sources_settings_path})"
            )
            if setting_changes:
                self._log("Settings changes:")
                for change in setting_changes:
                    self._log(f"  - {change}")

        if show_message:
            QMessageBox.information(
                self,
                "Config",
                f"Settings saved:\n{self.app_config_path}\nSources saved:\n{self.sources_settings_path}",
            )

    # ---------- Qt hooks ----------
    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: ANN401
        if event.type() == QEvent.Type.Resize:
            if watched is self.live_scroll.viewport():
                self._rebuild_live_layout()
            elif watched is self.main_tabs or watched is self.main_tabs.tabBar():
                self._position_overlay_controls()
            elif (
                watched is self.live_view_container
                or watched is self.preview_tabs
                or watched is self.preview_tabs.tabBar()
            ):
                self._position_overlay_controls()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: Any) -> None:  # noqa: ANN401
        super().resizeEvent(event)
        self._position_overlay_controls()

    def keyPressEvent(self, event: Any) -> None:  # noqa: ANN401
        if event.key() == Qt.Key.Key_Escape and self._is_fullscreen_visible():
            self._switch_to_grid_view()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: Any) -> None:  # noqa: ANN401
        self._flush_pending_settings()
        self._close_fullscreen_source()
        self.stop_live()
        self._stop_inference_worker()
        self._stop_event_writer_thread()
        self._recording_pause()
        self._release_recording_capture()

        for runtime in self.runtimes.values():
            runtime.release()

        self._persist_config(show_message=False)
        self._log("Application closed.")
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQt inference app for intrusion detection.")
    parser.add_argument(
        "--config",
        default="config/inference.yaml",
        help="Path to inference config YAML.",
    )
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="Scan and print available camera indexes, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    runtime_cfg = config.get("runtime", {}) or {}
    _configure_opencv_logging(silent=bool(runtime_cfg.get("suppress_opencv_warnings", True)))

    if args.scan_cameras:
        max_index = int(runtime_cfg.get("scan_max_index", 8))
        camera_backend = str(runtime_cfg.get("camera_backend", "msmf")).strip().lower() or "msmf"
        cameras = scan_available_cameras(max_index=max_index, preferred_backend=camera_backend)
        if cameras:
            print("[camera] Available indexes:", ", ".join(str(index) for index in cameras))
        else:
            print("[camera] No available camera found.")
        return

    app = QApplication(sys.argv)
    window = InferenceWindow(args.config)
    if bool(runtime_cfg.get("start_fullscreen", True)):
        window.showFullScreen()
    elif bool(runtime_cfg.get("start_maximized", True)):
        window.showMaximized()
    else:
        window.show()
    app.exec()


if __name__ == "__main__":
    main()
