from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
from ultralytics import YOLO

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from utils.config_utils import load_yaml, resolve_path, save_yaml
from utils.inference_utils import (
    open_capture,
    open_video_file_capture,
    resolve_security_mode,
    scan_available_cameras,
    should_raise_alert,
)
from utils.model_utils import load_yolo_model
from utils.runtime_env_utils import ensure_windows_compile_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TrackedBox = tuple[int, int, int, int, float, int | None]
EVENT_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EVENT_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"}


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


def _to_relative_or_abs(path_value: Path) -> str:
    try:
        return str(path_value.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:  # noqa: BLE001
        return str(path_value.resolve())


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


def _resolve_bytetrack_backend() -> tuple[type[Any] | None, str | None]:
    if importlib.util.find_spec("lap") is None:
        return None, "missing dependency 'lap' in active environment"

    try:
        from ultralytics.trackers.byte_tracker import BYTETracker as byte_tracker_cls
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    return byte_tracker_cls, None


def _extract_run_name_from_weight_filename(filename: str) -> str | None:
    for suffix in ("_best.pt", "_last.pt"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def _infer_model_family(name: str) -> str:
    text = str(name or "").lower()
    for family in ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"):
        if family in text:
            return family
    return "-"


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


def _apply_zoom_pan(frame: np.ndarray, zoom: float, pan_x: float, pan_y: float) -> np.ndarray:
    factor = max(1.0, float(zoom))
    if factor <= 1.01:
        return frame

    height, width = frame.shape[:2]
    crop_w = max(2, int(width / factor))
    crop_h = max(2, int(height / factor))

    max_shift_x = max(0, (width - crop_w) // 2)
    max_shift_y = max(0, (height - crop_h) // 2)

    center_x = (width // 2) + int(_clamp(pan_x, -1.0, 1.0) * max_shift_x)
    center_y = (height // 2) + int(_clamp(pan_y, -1.0, 1.0) * max_shift_y)

    x0 = int(_clamp(center_x - (crop_w // 2), 0, max(0, width - crop_w)))
    y0 = int(_clamp(center_y - (crop_h // 2), 0, max(0, height - crop_h)))

    cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _write_image(path: Path, frame: np.ndarray) -> bool:
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
        suffix = ".jpg"
        path = path.with_suffix(".jpg")
    try:
        ok, encoded = cv2.imencode(suffix, frame)
    except Exception:  # noqa: BLE001
        return False
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(str(path))
        return True
    except Exception:  # noqa: BLE001
        return False


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
    playback_interval_sec: float = 0.0
    last_frame_due_ts: float = 0.0
    ui_fps: float = 0.0
    last_render_ts: float = 0.0
    person_count: int = 0
    alert: bool = False
    mode: str = "day"
    last_boxes: list[TrackedBox] | None = None
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
    event_clip_temp_path: Path | None = None
    event_clip_writer: cv2.VideoWriter | None = None
    event_clip_frame_size: tuple[int, int] | None = None
    event_clip_frames_written: int = 0

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
        self.event_clip_started_wall_ts = 0.0


@dataclass
class AsyncInferenceResult:
    infer_ts: float
    person_count: int
    mode: str
    alert: bool
    boxes: list[TrackedBox]


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

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(64, 36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #0b0d11; color: #888; border: 0px;")
        self.setText(f"{source_name}\\nBrak klatki")

    def set_expand_mode(self, enabled: bool) -> None:
        self._expand_mode = bool(enabled)
        self._refresh_pixmap()

    def set_frame(
        self,
        frame: np.ndarray | None,
        *,
        zoom: float = 1.0,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
    ) -> None:
        self._frame = frame
        self._zoom = _clamp(float(zoom), 1.0, 8.0)
        self._pan_x = _clamp(float(pan_x), -1.0, 1.0)
        self._pan_y = _clamp(float(pan_y), -1.0, 1.0)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._frame is None:
            self.setPixmap(QPixmap())
            self.setText(f"{self.source_name}\\nBrak klatki")
            return

        frame = _apply_zoom_pan(self._frame, self._zoom, self._pan_x, self._pan_y)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return

        self.setText("")
        if self._expand_mode:
            self.setScaledContents(False)
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            # In grid mode prioritize throughput over visual smoothing.
            self.setScaledContents(True)
            self.setPixmap(pixmap)

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
        self._border_width = 5
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

    def set_focus_state(self, focused: bool) -> None:
        self._is_focused = bool(focused)
        self.canvas.set_expand_mode(self._is_focused)
        self.header_widget.setVisible(not self._is_focused)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)
        self._refresh_style()

    def set_alert_state(self, alert: bool) -> None:
        self._is_alert = bool(alert)
        self._refresh_style()

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
        self.meta_label.setText(meta_text)
        self.canvas.set_frame(frame, zoom=zoom, pan_x=pan_x, pan_y=pan_y)


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

        self.model_cfg = dict(self.config.get("model", {}) or {})
        self.inference_cfg = dict(self.config.get("inference", {}) or {})
        self.security_cfg = dict(self.config.get("security", {}) or {})
        self.runtime_cfg = dict(self.config.get("runtime", {}) or {})
        self.tracker_cfg = dict(self.config.get("tracker", {}) or {})
        self.events_cfg = dict(self.config.get("events", {}) or {})
        self.console_logs_enabled = bool(self.runtime_cfg.get("console_logs", False))
        self.suppress_opencv_warnings = bool(self.runtime_cfg.get("suppress_opencv_warnings", True))
        self.auto_scan_cameras_on_startup = bool(self.runtime_cfg.get("auto_scan_cameras_on_startup", False))
        self.auto_start_live = bool(self.runtime_cfg.get("auto_start_live", True))
        _configure_opencv_logging(silent=self.suppress_opencv_warnings)
        self.tracker_enabled = bool(self.tracker_cfg.get("enabled", True))
        self.byte_tracker_cls: type[Any] | None = None
        self._tracker_disabled_reason: str | None = None
        if self.tracker_enabled:
            self.byte_tracker_cls, tracker_error = _resolve_bytetrack_backend()
            if self.byte_tracker_cls is None:
                self.tracker_enabled = False
                self._tracker_disabled_reason = f"ByteTrack disabled: {tracker_error or 'backend unavailable'}"

        self.model: YOLO | None = None
        self.model_reference = ""
        self.current_model_path: Path | None = None
        self.predict_kwargs: dict[str, Any] = {}
        self.compile_enabled = False
        self.compile_fallback_applied = False

        self.app_root_dir = resolve_path("logs/app")
        self.app_settings_dir = self.app_root_dir / "settings"
        self.sources_settings_path = self.app_settings_dir / "sources.yaml"
        self.sources = self._load_sources_config()

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
        self.model_target_fps = max(0.1, float(self.runtime_cfg.get("model_target_fps", 6.0)))
        self.max_infer_per_tick = max(1, int(self.runtime_cfg.get("max_infer_per_tick", 2)))
        self._infer_rr_cursor = 0
        self.loop_videos = bool(self.runtime_cfg.get("loop_videos", True))
        self.live_tile_spacing = max(0, int(self.runtime_cfg.get("live_tile_spacing", 4)))

        self._model_lock = threading.Lock()
        self._capture_lock = threading.RLock()
        self._infer_lock = threading.RLock()
        self._infer_stop_event = threading.Event()
        self._infer_thread: threading.Thread | None = None
        self._infer_pending_frames: dict[str, np.ndarray] = {}
        self._infer_results: dict[str, AsyncInferenceResult] = {}
        self._infer_last_submit_ts: dict[str, float] = {}
        self._infer_worker_error: str | None = None
        self._infer_notices: list[str] = []
        self._infer_worker_rr_cursor = 0
        self._live_timer_interval_ms = int(max(1, self.frame_interval_ms))
        self._live_timer_last_adjust_ts = 0.0
        self.events_enabled = bool(self.events_cfg.get("enabled", True))
        self.events_min_visible_seconds = max(0.1, float(self.events_cfg.get("min_visible_seconds", 3.0)))
        self.events_cooldown_seconds = max(0.0, float(self.events_cfg.get("cooldown_seconds", 10.0)))
        self.events_min_person_count = max(1, int(self.events_cfg.get("min_person_count", 1)))
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
        if self._tracker_disabled_reason:
            self._log(self._tracker_disabled_reason)
        elif self.tracker_enabled:
            self._log("ByteTrack enabled.")

        self.model_catalog: list[dict[str, Any]] = []

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

        self._load_model()
        self._build_ui()
        self._sync_runtimes_with_sources()
        self._rebuild_source_table()
        self._rebuild_live_layout()
        self._refresh_model_catalog()

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
            self.logs_text.append(line)

    def _auto_start_live_if_possible(self) -> None:
        if self.live_running:
            return
        if not self._get_enabled_sources():
            self._log("Auto-start skipped: no enabled sources.")
            return
        self.start_live()

    # ---------- model ----------
    def _resolve_canonical_trained_model_path(self) -> Path | None:
        model_name = str(self.model_cfg.get("name", "")).strip()
        if not model_name:
            return None
        if not model_name.lower().endswith(".pt"):
            model_name = f"{model_name}.pt"
        candidate = resolve_path(Path("models/weights") / model_name)
        if candidate.exists():
            return candidate.resolve()
        return None

    def _resolve_trained_weights_path(self) -> Path | None:
        trained_cfg = self.model_cfg.get("trained_weights", {}) or {}
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

    def _resolve_model_reference(self) -> tuple[YOLO, str, Path | None]:
        selected_model_path = str(self.model_cfg.get("selected_model_path", "")).strip()
        if selected_model_path:
            selected_path = resolve_path(selected_model_path)
            if selected_path.exists():
                return YOLO(str(selected_path)), str(selected_path), selected_path

        trained_cfg = self.model_cfg.get("trained_weights", {}) or {}
        prefer_canonical = bool(trained_cfg.get("prefer_canonical", True))
        if prefer_canonical:
            canonical_trained_path = self._resolve_canonical_trained_model_path()
            if canonical_trained_path is not None:
                return YOLO(str(canonical_trained_path)), str(canonical_trained_path), canonical_trained_path

        trained_path = self._resolve_trained_weights_path()
        fallback_to_base = bool(trained_cfg.get("fallback_to_base", True))

        if trained_path is not None:
            return YOLO(str(trained_path)), str(trained_path), trained_path

        if fallback_to_base:
            model, reference = load_yolo_model(self.model_cfg)
            reference_path = Path(reference).resolve() if Path(reference).exists() else None
            return model, reference, reference_path

        raise FileNotFoundError(
            "No model available. Set model.selected_model_path or keep trained_weights fallback enabled."
        )

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

        if self.compile_enabled:
            self.predict_kwargs["compile"] = True
        else:
            self.predict_kwargs.pop("compile", None)

    def _load_model(self) -> None:
        ensure_windows_compile_env(self.inference_cfg, compile_value=self.inference_cfg.get("compile", False))
        self.compile_enabled = _is_compile_enabled(self.inference_cfg.get("compile", False))

        model, reference, reference_path = self._resolve_model_reference()
        self.model = model
        self.model_reference = reference
        self.current_model_path = reference_path
        self.compile_fallback_applied = False

        self._rebuild_predict_kwargs()
        self._log(f"Model loaded: {reference}")

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
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tabs.addTab(self._build_settings_tab(), "Ustawienia")
        self.main_tabs.addTab(self._build_camera_config_tab(), "Konfiguracja kamer")
        self.main_tabs.addTab(self._build_preview_tab(), "Podglad kamer")
        self.main_tabs.addTab(self._build_events_tab(), "Wykryty ruch")
        self.main_tabs.addTab(self._build_logs_tab(), "Logi")
        layout.addWidget(self.main_tabs)

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
        self.main_tabs.setCurrentIndex(2)
        self.preview_tabs.setCurrentIndex(0)
        self._apply_theme()
        self._position_overlay_controls()

    def _apply_theme(self) -> None:
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
            "QTableWidget::item:selected { background: #2f81f7; color: #ffffff; }"
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
            "QCheckBox::indicator:checked { background: #2f81f7; border: 1px solid #2f81f7; }"
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

        security_box = QGroupBox("Reguly alarmu (dzien / noc)")
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
        security_grid.addWidget(QLabel("Prog alarmu w dzien (osoby):"), 3, 0)
        security_grid.addWidget(self.day_threshold_spin, 3, 1)
        security_grid.addWidget(QLabel("Prog alarmu w nocy (osoby):"), 4, 0)
        security_grid.addWidget(self.night_threshold_spin, 4, 1)

        layout.addWidget(security_box)

        inference_box = QGroupBox("Parametry detekcji YOLO")
        inference_grid = QGridLayout(inference_box)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.99)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1920)
        self.imgsz_spin.setSingleStep(32)

        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)

        self.device_edit = QLineEdit()
        self.half_checkbox = QCheckBox("FP16 (half precision)")
        self.compile_checkbox = QCheckBox("torch.compile (jesli stabilne)")
        self.start_maximized_checkbox = QCheckBox("Start aplikacji w trybie zmaksymalizowanym")

        inference_grid.addWidget(QLabel("Prog pewnosci (conf):"), 0, 0)
        inference_grid.addWidget(self.conf_spin, 0, 1)
        inference_grid.addWidget(QLabel("Prog IOU (NMS):"), 1, 0)
        inference_grid.addWidget(self.iou_spin, 1, 1)
        inference_grid.addWidget(QLabel("Rozmiar wejscia (imgsz):"), 2, 0)
        inference_grid.addWidget(self.imgsz_spin, 2, 1)
        inference_grid.addWidget(QLabel("Maks. liczba detekcji:"), 3, 0)
        inference_grid.addWidget(self.max_det_spin, 3, 1)
        inference_grid.addWidget(QLabel("Urzadzenie (np. 0/cpu):"), 4, 0)
        inference_grid.addWidget(self.device_edit, 4, 1)
        inference_grid.addWidget(self.half_checkbox, 5, 0, 1, 2)
        inference_grid.addWidget(self.compile_checkbox, 6, 0, 1, 2)
        inference_grid.addWidget(self.start_maximized_checkbox, 7, 0, 1, 2)

        layout.addWidget(inference_box)

        events_box = QGroupBox("Archiwizacja zdarzen")
        events_grid = QGridLayout(events_box)

        self.events_enabled_checkbox = QCheckBox("Zapisz klip wideo, gdy osoba jest widoczna dluzej niz prog")
        self.events_min_visible_spin = QDoubleSpinBox()
        self.events_min_visible_spin.setRange(0.3, 120.0)
        self.events_min_visible_spin.setSingleStep(0.2)
        self.events_min_visible_spin.setDecimals(1)

        self.events_cooldown_spin = QDoubleSpinBox()
        self.events_cooldown_spin.setRange(0.0, 3600.0)
        self.events_cooldown_spin.setSingleStep(0.5)
        self.events_cooldown_spin.setDecimals(1)

        self.events_min_person_spin = QSpinBox()
        self.events_min_person_spin.setRange(1, 20)

        self.events_max_saved_spin = QSpinBox()
        self.events_max_saved_spin.setRange(0, 20000)
        self.events_max_saved_spin.setSpecialValueText("0 (bez limitu)")

        self.events_save_annotated_checkbox = QCheckBox("Zapisuj klip z boxami i opisem")
        self.events_once_per_streak_checkbox = QCheckBox("Tylko jeden zapis na ciagla sekwencje wykrycia")

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

        events_grid.addWidget(self.events_enabled_checkbox, 0, 0, 1, 2)
        events_grid.addWidget(QLabel("Minimalny czas widocznosci (s):"), 1, 0)
        events_grid.addWidget(self.events_min_visible_spin, 1, 1)
        events_grid.addWidget(QLabel("Cooldown miedzy zapisami (s):"), 2, 0)
        events_grid.addWidget(self.events_cooldown_spin, 2, 1)
        events_grid.addWidget(QLabel("Min. liczba osob:"), 3, 0)
        events_grid.addWidget(self.events_min_person_spin, 3, 1)
        events_grid.addWidget(QLabel("Maks. liczba zapisanych zdarzen:"), 4, 0)
        events_grid.addWidget(self.events_max_saved_spin, 4, 1)
        events_grid.addWidget(self.events_save_annotated_checkbox, 5, 0, 1, 2)
        events_grid.addWidget(self.events_once_per_streak_checkbox, 6, 0, 1, 2)
        events_grid.addWidget(QLabel("Folder zapisu zdarzen:"), 7, 0)
        events_grid.addWidget(output_row_widget, 7, 1)

        layout.addWidget(events_box)

        model_box = QGroupBox("Wybor modelu")
        model_layout = QVBoxLayout(model_box)

        self.current_model_label = QLabel("Aktualny model: -")
        self.current_model_label.setWordWrap(True)
        self.current_model_label.setStyleSheet("color: #d8d8d8;")
        model_layout.addWidget(self.current_model_label)

        self.model_help_label = QLabel(
            "Base = surowe wagi (np. yolo26n.pt), "
            "trained/latest = ostatni best/last z treningu, "
            "trained/final = najlepszy model utrwalony per architektura."
        )
        self.model_help_label.setWordWrap(True)
        self.model_help_label.setStyleSheet("color: #9fb0c9;")
        model_layout.addWidget(self.model_help_label)

        self.model_table = QTableWidget(0, 7)
        self.model_table.setHorizontalHeaderLabels(["Model", "Zrodlo", "Arch", "mAP50", "mAP50-95", "Run", "Path"])
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_table.verticalHeader().setVisible(False)
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
        apply_model_btn = QPushButton("Zaladuj wybrany model")
        apply_model_btn.clicked.connect(self._apply_selected_model)
        model_buttons.addWidget(refresh_models_btn)
        model_buttons.addWidget(apply_model_btn)
        model_layout.addLayout(model_buttons)

        layout.addWidget(model_box)

        save_settings_btn = QPushButton("Zapisz ustawienia")
        save_settings_btn.clicked.connect(lambda: self._persist_config(show_message=True))
        layout.addWidget(save_settings_btn)

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

        self.source_table = QTableWidget(0, 4)
        self.source_table.setHorizontalHeaderLabels(["Name", "Type", "Value", "Enabled"])
        self.source_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.source_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.source_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.source_table.verticalHeader().setVisible(False)

        header = self.source_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.source_table.itemChanged.connect(self._on_source_item_changed)
        table_layout.addWidget(self.source_table)

        source_buttons = QHBoxLayout()
        remove_source_btn = QPushButton("Remove selected source")
        remove_source_btn.clicked.connect(self._remove_selected_source)
        save_source_btn = QPushButton("Save camera config")
        save_source_btn.clicked.connect(lambda: self._persist_config(show_message=True))
        source_buttons.addWidget(remove_source_btn)
        source_buttons.addWidget(save_source_btn)
        table_layout.addLayout(source_buttons)

        layout.addWidget(table_box)

        self.source_toolbox = QToolBox()
        self.source_toolbox.addItem(self._build_add_camera_page(), "Add camera")
        self.source_toolbox.addItem(self._build_add_video_page(), "Add video source")
        self.source_toolbox.addItem(self._build_add_stream_page(), "Add stream source")
        layout.addWidget(self.source_toolbox)

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
        layout.addWidget(add_btn)

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
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_video_file)
        row.addWidget(self.video_path_edit, stretch=1)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        self.video_name_edit = QLineEdit()
        self.video_name_edit.setPlaceholderText("Name (optional)")
        layout.addWidget(self.video_name_edit)

        add_btn = QPushButton("Add video source")
        add_btn.clicked.connect(self._add_video_source)
        layout.addWidget(add_btn)

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
        layout.addWidget(add_btn)

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

        self.start_live_btn = QPushButton("Start")
        self.stop_live_btn = QPushButton("Stop")
        self.grid_view_btn = QPushButton("Grid view")
        self.zoom_out_btn = QPushButton("-")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_reset_btn = QPushButton("Reset zoom")
        self.zoom_label = QLabel("Zoom: 1.00x")

        self.start_live_btn.clicked.connect(self.start_live)
        self.stop_live_btn.clicked.connect(self.stop_live)
        self.grid_view_btn.clicked.connect(self._switch_to_grid_view)
        self.zoom_out_btn.clicked.connect(lambda: self._change_focus_zoom(-1))
        self.zoom_in_btn.clicked.connect(lambda: self._change_focus_zoom(1))
        self.zoom_reset_btn.clicked.connect(self._reset_focus_zoom)

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

        self.live_controls_toggle_btn = QPushButton("⚙", self.preview_tabs)
        self.live_controls_toggle_btn.setFixedSize(40, 30)
        self.live_controls_toggle_btn.clicked.connect(self._toggle_live_controls_panel)
        self.live_controls_toggle_btn.setToolTip(
            "Pokaz/ukryj kontrolki. "
            "LPM na kaflu: pelny ekran/powrot, scroll: zoom, LPM+drag: pan, Esc: wyjscie z pelnego ekranu."
        )
        self.live_controls_toggle_btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgba(20, 24, 31, 225);"
            "color: #e7edf8;"
            "border: 1px solid #4a5568;"
            "border-radius: 6px;"
            "font-size: 15px;"
            "font-weight: 700;"
            "}"
            "QPushButton:hover { background-color: rgba(32, 38, 48, 235); }"
        )

        self.live_controls_panel = QFrame(self.preview_tabs)
        self.live_controls_panel.setFrameShape(QFrame.Shape.NoFrame)
        panel_layout = QGridLayout(self.live_controls_panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setHorizontalSpacing(6)
        panel_layout.setVerticalSpacing(6)
        panel_layout.addWidget(self.start_live_btn, 0, 0)
        panel_layout.addWidget(self.stop_live_btn, 0, 1)
        panel_layout.addWidget(self.grid_view_btn, 0, 2)
        panel_layout.addWidget(self.zoom_out_btn, 1, 0)
        panel_layout.addWidget(self.zoom_in_btn, 1, 1)
        panel_layout.addWidget(self.zoom_reset_btn, 1, 2)
        panel_layout.addWidget(self.zoom_label, 2, 0, 1, 3)

        self.start_live_btn.setMinimumWidth(86)
        self.stop_live_btn.setMinimumWidth(86)
        self.grid_view_btn.setMinimumWidth(110)
        self.zoom_out_btn.setMinimumWidth(52)
        self.zoom_in_btn.setMinimumWidth(52)
        self.zoom_reset_btn.setMinimumWidth(110)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet("color: #e6edf8; font-weight: 600;")

        self.live_controls_panel.setStyleSheet(
            "QFrame {"
            "background-color: rgba(15, 20, 29, 230);"
            "border: 1px solid #39485d;"
            "border-radius: 8px;"
            "}"
        )
        self.live_controls_panel.hide()
        self.live_controls_toggle_btn.setText("⚙")
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

        self.events_table = QTableWidget(0, 6)
        self.events_table.setHorizontalHeaderLabels(["Time", "Source", "Mode", "Persons", "Visible[s]", "File"])
        self.events_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.events_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.events_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.itemSelectionChanged.connect(self._on_event_table_selection_changed)
        events_header = self.events_table.horizontalHeader()
        events_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        events_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        events_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        events_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        events_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        events_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

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

        layout.addWidget(splitter, stretch=1)
        layout.addLayout(controls)
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

        for line in self._log_entries:
            self.logs_text.append(line)

        return page

    def _set_controls_from_config(self) -> None:
        self.security_mode_combo.setCurrentText(str(self.security_cfg.get("mode", "auto")))
        self.night_start_spin.setValue(int(self.security_cfg.get("night_start_hour", 22)))
        self.night_end_spin.setValue(int(self.security_cfg.get("night_end_hour", 6)))
        self.day_threshold_spin.setValue(int(self.security_cfg.get("day_person_threshold", 1)))
        self.night_threshold_spin.setValue(int(self.security_cfg.get("night_person_threshold", 1)))

        self.conf_spin.setValue(float(self.inference_cfg.get("conf", 0.35)))
        self.iou_spin.setValue(float(self.inference_cfg.get("iou", 0.45)))
        self.imgsz_spin.setValue(int(self.inference_cfg.get("imgsz", 960)))
        self.max_det_spin.setValue(int(self.inference_cfg.get("max_det", 100)))
        self.device_edit.setText(str(self.inference_cfg.get("device", "0")))
        self.half_checkbox.setChecked(bool(self.inference_cfg.get("half", True)))
        self.compile_checkbox.setChecked(_is_compile_enabled(self.inference_cfg.get("compile", False)))
        self.start_maximized_checkbox.setChecked(bool(self.runtime_cfg.get("start_maximized", True)))
        self.events_enabled_checkbox.setChecked(bool(self.events_cfg.get("enabled", True)))
        self.events_min_visible_spin.setValue(float(self.events_cfg.get("min_visible_seconds", 3.0)))
        self.events_cooldown_spin.setValue(float(self.events_cfg.get("cooldown_seconds", 10.0)))
        self.events_min_person_spin.setValue(int(self.events_cfg.get("min_person_count", 1)))
        self.events_max_saved_spin.setValue(int(self.events_cfg.get("max_saved_events", 300)))
        self.events_save_annotated_checkbox.setChecked(bool(self.events_cfg.get("save_annotated_frame", True)))
        self.events_once_per_streak_checkbox.setChecked(bool(self.events_cfg.get("once_per_streak", True)))
        self.events_output_dir_edit.setText(str(self.events_cfg.get("output_dir", "logs/app/events")))

        self.current_model_label.setText(f"Aktualny model: {self.model_reference}")

        last_recording = str(self.runtime_cfg.get("last_recording_path", "")).strip()
        if last_recording:
            self.recording_path_edit.setText(last_recording)

        self.stop_live_btn.setEnabled(False)
        self._suppress_setting_autosave = False

    def _bind_setting_autosave(self) -> None:
        self.security_mode_combo.currentTextChanged.connect(self._on_setting_changed)
        self.night_start_spin.valueChanged.connect(self._on_setting_changed)
        self.night_end_spin.valueChanged.connect(self._on_setting_changed)
        self.day_threshold_spin.valueChanged.connect(self._on_setting_changed)
        self.night_threshold_spin.valueChanged.connect(self._on_setting_changed)
        self.conf_spin.valueChanged.connect(self._on_setting_changed)
        self.iou_spin.valueChanged.connect(self._on_setting_changed)
        self.imgsz_spin.valueChanged.connect(self._on_setting_changed)
        self.max_det_spin.valueChanged.connect(self._on_setting_changed)
        self.device_edit.textChanged.connect(self._on_setting_changed)
        self.half_checkbox.toggled.connect(self._on_setting_changed)
        self.compile_checkbox.toggled.connect(self._on_setting_changed)
        self.start_maximized_checkbox.toggled.connect(self._on_setting_changed)
        self.events_enabled_checkbox.toggled.connect(self._on_setting_changed)
        self.events_min_visible_spin.valueChanged.connect(self._on_setting_changed)
        self.events_cooldown_spin.valueChanged.connect(self._on_setting_changed)
        self.events_min_person_spin.valueChanged.connect(self._on_setting_changed)
        self.events_max_saved_spin.valueChanged.connect(self._on_setting_changed)
        self.events_save_annotated_checkbox.toggled.connect(self._on_setting_changed)
        self.events_once_per_streak_checkbox.toggled.connect(self._on_setting_changed)
        self.events_output_dir_edit.editingFinished.connect(self._on_setting_changed)

    def _on_setting_changed(self, *_args: Any) -> None:
        if self._suppress_setting_autosave:
            return

        self._apply_controls_to_runtime_state()
        ensure_windows_compile_env(self.inference_cfg, compile_value=self.inference_cfg.get("compile", False))
        self._rebuild_predict_kwargs()
        self._persist_config(show_message=False)

    # ---------- model list ----------
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
                }
            )

        self.model_catalog = entries
        self._populate_model_table()

    def _update_current_model_label(self) -> None:
        current_path = self.current_model_path.resolve() if self.current_model_path and self.current_model_path.exists() else None
        if current_path is None:
            self.current_model_label.setText(f"Aktualny model: {self.model_reference}")
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

    def _populate_model_table(self) -> None:
        self.model_table.setRowCount(len(self.model_catalog))

        selected_row = -1
        current_path_str = str(self.current_model_path.resolve()) if self.current_model_path and self.current_model_path.exists() else ""

        for row, entry in enumerate(self.model_catalog):
            map50 = entry["map50"]
            map5095 = entry["map5095"]

            self.model_table.setItem(row, 0, QTableWidgetItem(str(entry["display_name"])))
            self.model_table.setItem(row, 1, QTableWidgetItem(str(entry["kind"])))
            self.model_table.setItem(row, 2, QTableWidgetItem(str(entry["family"])))
            self.model_table.setItem(row, 3, QTableWidgetItem("-" if map50 is None else f"{map50:.4f}"))
            self.model_table.setItem(row, 4, QTableWidgetItem("-" if map5095 is None else f"{map5095:.4f}"))
            self.model_table.setItem(row, 5, QTableWidgetItem(str(entry["run"])))
            self.model_table.setItem(row, 6, QTableWidgetItem(str(entry["path_display"])))

            if current_path_str and str(entry["path"]) == current_path_str:
                selected_row = row

        if selected_row >= 0:
            self.model_table.selectRow(selected_row)

        self._update_current_model_label()

    def _apply_selected_model(self) -> None:
        row = self.model_table.currentRow()
        if row < 0 or row >= len(self.model_catalog):
            QMessageBox.information(self, "Model", "Najpierw wybierz wiersz modelu.")
            return

        entry = self.model_catalog[row]
        model_path = Path(entry["path"]).resolve()
        if not model_path.exists():
            QMessageBox.warning(self, "Model", f"Sciezka modelu nie istnieje:\n{model_path}")
            return

        was_running = self.live_running
        if was_running:
            self.stop_live()

        try:
            self.model = YOLO(str(model_path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model", f"Nie mozna zaladowac modelu:\n{exc}")
            if was_running:
                self.start_live()
            return

        self.current_model_path = model_path
        self.model_reference = str(model_path)
        self.model_cfg["selected_model_path"] = _to_relative_or_abs(model_path)
        selected_model_name = str(entry.get("model_name", "")).strip()
        if selected_model_name.lower().endswith(".pt"):
            self.model_cfg["name"] = selected_model_name
        self.compile_fallback_applied = False

        self._apply_controls_to_runtime_state()
        self._rebuild_predict_kwargs()
        self._update_current_model_label()
        self._persist_config(show_message=False)
        self._log(f"Model switched to: {model_path}")

        if was_running:
            self.start_live()

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

    def _refresh_camera_list(self) -> None:
        max_index = int(self.runtime_cfg.get("scan_max_index", 8))
        self.camera_combo.clear()
        for camera_index in scan_available_cameras(max_index=max_index):
            self.camera_combo.addItem(f"Camera {camera_index}", camera_index)

    def _browse_video_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(resolve_path("data/videos")),
            "Video files (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;All files (*.*)",
        )
        if file_path:
            self.video_path_edit.setText(file_path)

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

    def _add_source(self, source_type: str, value: Any, name_hint: str) -> None:
        source_name = _ensure_unique_name(self._existing_source_names(), _safe_name(name_hint, f"{source_type}_source"))
        source = {
            "name": source_name,
            "type": source_type,
            "value": value,
            "enabled": True,
        }
        self.sources.append(source)
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

        base_name = self.video_name_edit.text().strip() or video_path.stem
        self._add_source("video", str(video_path), base_name)
        self.video_name_edit.clear()

    def _add_stream_source(self) -> None:
        stream_url = self.stream_url_edit.text().strip()
        if not stream_url:
            QMessageBox.warning(self, "Stream", "Provide stream URL first.")
            return

        base_name = self.stream_name_edit.text().strip() or "stream_source"
        self._add_source("stream", stream_url, base_name)
        self.stream_name_edit.clear()

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
        source_names = {str(source.get("name", "")) for source in self.sources}

        for name in list(self.runtimes):
            if name in source_names:
                continue
            self._finalize_event_clip(name, self.runtimes[name], time.perf_counter())
            self.runtimes[name].release()
            del self.runtimes[name]
            self._clear_async_state_for_source(name)
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

                self.source_table.setItem(row, 0, QTableWidgetItem(name))
                self.source_table.setItem(row, 1, QTableWidgetItem(source_type))
                self.source_table.setItem(row, 2, QTableWidgetItem(value))

                enabled_item = QTableWidgetItem("yes" if enabled else "no")
                enabled_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                enabled_item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
                self.source_table.setItem(row, 3, enabled_item)
        finally:
            self._table_updating = False

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

    # ---------- live layout ----------
    def _get_enabled_sources(self) -> list[dict[str, Any]]:
        return [source for source in self.sources if bool(source.get("enabled", True))]

    def _ensure_tile(self, source_name: str) -> VideoTile:
        tile = self.tiles.get(source_name)
        if tile is not None:
            return tile

        tile = VideoTile(source_name)
        tile.clicked.connect(self._on_tile_clicked)
        tile.right_clicked.connect(self._on_tile_right_clicked)
        tile.zoom_delta.connect(self._on_tile_zoom_delta)
        tile.pan_delta.connect(self._on_tile_pan_delta)
        self.tiles[source_name] = tile
        return tile

    def _clear_live_grid(self) -> None:
        while self.live_grid_layout.count() > 0:
            item = self.live_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _auto_grid_columns(self, count: int) -> int:
        columns, _rows = self._auto_grid_dimensions(count)
        return columns

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
            self.zoom_label.setText("Zoom: 1.00x")
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

        if self.focused_source:
            zoom = self.zoom_levels.get(self.focused_source, 1.0)
            self.zoom_label.setText(f"Zoom: {zoom:.2f}x")
        else:
            self.zoom_label.setText("Zoom: 1.00x")

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
        self._close_fullscreen_source()
        self.focused_source = None
        self.zoom_label.setText("Zoom: 1.00x")
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
            window.set_frame(runtime.last_output, zoom=zoom, pan_x=pan_x, pan_y=pan_y)
            self.zoom_label.setText(f"Zoom: {zoom:.2f}x")
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

    def _toggle_live_controls_panel(self) -> None:
        if hasattr(self, "preview_tabs") and self.preview_tabs.currentIndex() != 0:
            return
        visible = not self.live_controls_panel.isVisible()
        self.live_controls_panel.setVisible(visible)
        self.live_controls_toggle_btn.setText("✕" if visible else "⚙")
        self._update_live_overlay_margin()
        self._position_overlay_controls()

    def _on_preview_subtab_changed(self, index: int) -> None:
        if not hasattr(self, "live_controls_toggle_btn") or not hasattr(self, "live_controls_panel"):
            return
        is_live = int(index) == 0
        self.live_controls_toggle_btn.setVisible(is_live)
        if not is_live:
            self.live_controls_panel.hide()
            self.live_controls_toggle_btn.setText("⚙")
        self._position_overlay_controls()

    def _update_live_overlay_margin(self) -> None:
        if not hasattr(self, "live_view_layout"):
            return
        self.live_view_layout.setContentsMargins(0, 0, 0, 0)

    def _position_overlay_controls(self) -> None:
        side_margin = 10
        top_margin = 0

        if hasattr(self, "exit_app_btn") and self.exit_app_btn is not None:
            parent = self.exit_app_btn.parentWidget()
            if parent is not None:
                x = max(side_margin, parent.width() - self.exit_app_btn.width() - side_margin)
                y = top_margin
                self.exit_app_btn.move(x, y)
                self.exit_app_btn.raise_()

        if (
            hasattr(self, "preview_tabs")
            and self.preview_tabs is not None
            and self.live_controls_toggle_btn is not None
        ):
            container = self.preview_tabs
            if container.currentIndex() != 0:
                return

            tab_bar = container.tabBar()
            toggle_h = self.live_controls_toggle_btn.height()
            toggle_x = max(side_margin, container.width() - self.live_controls_toggle_btn.width() - side_margin)
            toggle_y = max(1, tab_bar.geometry().top() + 2)
            self.live_controls_toggle_btn.move(toggle_x, toggle_y)
            self.live_controls_toggle_btn.raise_()

            if self.live_controls_panel.isVisible():
                self.live_controls_panel.adjustSize()
                panel_max_width = max(320, int(container.width() * 0.72))
                self.live_controls_panel.setMaximumWidth(panel_max_width)
                panel_x = max(side_margin, container.width() - self.live_controls_panel.width() - side_margin)
                panel_y = toggle_y + toggle_h + 8
                self.live_controls_panel.move(panel_x, panel_y)
                self.live_controls_panel.raise_()

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

        self.zoom_label.setText(f"Zoom: {updated:.2f}x")
        self._refresh_tile(self.focused_source)

    def _reset_focus_zoom(self) -> None:
        if not self.focused_source:
            return
        self.zoom_levels[self.focused_source] = 1.0
        self.pan_offsets[self.focused_source] = (0.0, 0.0)
        self.zoom_label.setText("Zoom: 1.00x")
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

    def _enqueue_inference_frame(self, source_name: str, frame: np.ndarray, submit_ts: float) -> None:
        with self._infer_lock:
            # Keep only the newest frame for each source (drop stale work).
            self._infer_pending_frames[source_name] = frame
            self._infer_last_submit_ts[source_name] = float(submit_ts)

    def _pull_inference_batch(self) -> tuple[list[str], list[np.ndarray]]:
        with self._infer_lock:
            source_names = list(self._infer_pending_frames.keys())
            if not source_names:
                return [], []

            total = len(source_names)
            start = self._infer_worker_rr_cursor % total
            ordered = [source_names[(start + idx) % total] for idx in range(total)]
            chosen = ordered[: max(1, int(self.max_infer_per_tick))]
            self._infer_worker_rr_cursor = (start + len(chosen)) % max(1, total)

            batch_sources: list[str] = []
            batch_frames: list[np.ndarray] = []
            for source_name in chosen:
                frame = self._infer_pending_frames.pop(source_name, None)
                if frame is None:
                    continue
                batch_sources.append(source_name)
                batch_frames.append(frame)
        return batch_sources, batch_frames

    def _start_inference_worker(self) -> None:
        if self._infer_thread is not None and self._infer_thread.is_alive():
            return

        with self._infer_lock:
            self._infer_stop_event.clear()
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

    def _inference_worker_loop(self) -> None:
        while not self._infer_stop_event.is_set():
            source_batch, frame_batch = self._pull_inference_batch()
            if not source_batch:
                self._infer_stop_event.wait(0.002)
                continue

            try:
                batch_results = self._predict_batch_with_fallback(frame_batch)
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
                boxes = self._extract_tracked_person_boxes(source_name, None, result, frame)
                tracked_ids = {box[5] for box in boxes if len(box) >= 6 and box[5] is not None}
                person_count = len(tracked_ids) if tracked_ids else len(boxes)
                mode = resolve_security_mode(self.security_cfg)
                alert = should_raise_alert(person_count, mode, self.security_cfg)
                payload = AsyncInferenceResult(
                    infer_ts=now,
                    person_count=person_count,
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
            updates = dict(self._infer_results)
            self._infer_results.clear()

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
            runtime.person_count = payload.person_count
            runtime.mode = payload.mode
            runtime.alert = payload.alert
            runtime.last_boxes = payload.boxes
            runtime.status = "alert" if payload.alert else "ok"
        return True

    def _compute_effective_view_fps_cap(self) -> float:
        configured_cap = float(_clamp(float(self.view_target_fps), 1.0, 60.0))
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

            with self._capture_lock:
                runtime.capture_latest_frame = frame
                runtime.capture_latest_seq += 1

    def _build_tracker_args(self) -> SimpleNamespace:
        high = float(_clamp(float(self.tracker_cfg.get("track_high_thresh", 0.5)), 0.0, 1.0))
        low = float(_clamp(float(self.tracker_cfg.get("track_low_thresh", 0.1)), 0.0, high))
        new_track = float(_clamp(float(self.tracker_cfg.get("new_track_thresh", 0.6)), low, 1.0))
        match = float(_clamp(float(self.tracker_cfg.get("match_thresh", 0.8)), 0.0, 1.0))
        track_buffer = max(1, int(self.tracker_cfg.get("track_buffer", 30)))
        fuse_score = bool(self.tracker_cfg.get("fuse_score", True))
        return SimpleNamespace(
            track_high_thresh=high,
            track_low_thresh=low,
            new_track_thresh=new_track,
            track_buffer=track_buffer,
            match_thresh=match,
            fuse_score=fuse_score,
        )

    def _reset_tracker_for_source(self, source_name: str, runtime: SourceRuntime | None = None) -> None:
        with self._infer_lock:
            self.trackers.pop(source_name, None)
        if not self.tracker_enabled or self.byte_tracker_cls is None:
            return

        configured_rate = int(self.tracker_cfg.get("frame_rate", 30))
        if configured_rate > 0:
            frame_rate = configured_rate
        elif runtime is not None and runtime.source_fps > 1e-3:
            frame_rate = max(1, int(round(runtime.source_fps)))
        else:
            frame_rate = 30

        try:
            tracker = self.byte_tracker_cls(args=self._build_tracker_args(), frame_rate=frame_rate)
            with self._infer_lock:
                self.trackers[source_name] = tracker
        except Exception as exc:  # noqa: BLE001
            self._queue_async_notice(f"[warn] ByteTrack init failed for '{source_name}': {exc}")
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
        runtime.capture = open_capture(runtime.source)
        ok = runtime.capture.isOpened()
        runtime.source_fps = 0.0
        runtime.playback_interval_sec = 0.0
        runtime.last_frame_due_ts = 0.0
        runtime.last_tick_ts = 0.0
        runtime.last_infer_ts = 0.0
        runtime.fps = 0.0
        runtime.infer_fps = 0.0
        runtime.ui_fps = 0.0
        runtime.last_render_ts = 0.0
        runtime.last_boxes = None
        runtime.last_decorated_capture_seq = 0
        runtime.last_decorated_infer_ts = 0.0
        runtime.no_frame_refresh_needed = True
        runtime.person_visible_since_ts = 0.0
        runtime.person_visible_duration_sec = 0.0
        runtime.last_event_capture_ts = 0.0
        runtime.event_saved_in_streak = False
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
                if camera_width > 0:
                    try:
                        runtime.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera_width))
                    except Exception:  # noqa: BLE001
                        pass
                if camera_height > 0:
                    try:
                        runtime.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera_height))
                    except Exception:  # noqa: BLE001
                        pass
                if camera_fps > 0:
                    try:
                        runtime.capture.set(cv2.CAP_PROP_FPS, float(camera_fps))
                    except Exception:  # noqa: BLE001
                        pass

                requested_fps = float(self.runtime_cfg.get("camera_fps", 30))
                runtime.source_fps = max(1.0, requested_fps)

            self._start_capture_reader(source_name, runtime)
        if ok:
            self._reset_tracker_for_source(source_name, runtime)

        runtime.status = "open" if ok else "failed"
        return ok

    def _read_frame(self, runtime: SourceRuntime) -> tuple[np.ndarray | None, bool]:
        if not self._ensure_capture(runtime):
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
            return latest_frame, fresh

        runtime.status = "no-frame"
        if runtime.last_input is not None:
            return runtime.last_input, False
        return None, False

    def _predict_batch_with_fallback(self, frames: list[np.ndarray]) -> list[Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not frames:
            return []

        with self._model_lock:
            try:
                results = self.model.predict(frames, **self.predict_kwargs)
                return list(results)
            except Exception as exc:  # noqa: BLE001
                if not self.compile_enabled:
                    raise

                message = str(exc)
                compile_related = (
                    "AutoBackend does not support len()" in message
                    or "torch._inductor" in message
                    or "triton" in message.lower()
                )
                if not compile_related:
                    raise

                if not self.compile_fallback_applied:
                    self.compile_fallback_applied = True
                    self._queue_async_notice(
                        "compile failed for inference, fallback to compile=False "
                        f"({exc.__class__.__name__}: {message})"
                    )

                self.compile_enabled = False
                self.inference_cfg["compile"] = False
                self.predict_kwargs.pop("compile", None)
                try:
                    self.model.predictor = None
                except Exception:  # noqa: BLE001
                    pass

                results = self.model.predict(frames, **self.predict_kwargs)
                return list(results)

    def _extract_person_boxes(self, result: Any) -> list[TrackedBox]:
        boxes: list[TrackedBox] = []
        raw_boxes = getattr(result, "boxes", None)
        if raw_boxes is None:
            return boxes

        try:
            xyxy_values = raw_boxes.xyxy
            conf_values = raw_boxes.conf
            cls_values = raw_boxes.cls
        except Exception:  # noqa: BLE001
            return boxes

        try:
            xyxy_list = xyxy_values.tolist()
            conf_list = conf_values.tolist() if conf_values is not None else [1.0] * len(xyxy_list)
            cls_list = cls_values.tolist() if cls_values is not None else [0.0] * len(xyxy_list)
        except Exception:  # noqa: BLE001
            return boxes

        for coords, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
            if int(cls_id) != 0:
                continue
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2, float(conf), None))

        return boxes

    def _extract_tracked_person_boxes(
        self,
        source_name: str,
        runtime: SourceRuntime | None,
        result: Any,
        frame: np.ndarray,
    ) -> list[TrackedBox]:
        if not self.tracker_enabled:
            return self._extract_person_boxes(result)

        raw_boxes = getattr(result, "boxes", None)
        if raw_boxes is None:
            return []

        with self._infer_lock:
            tracker = self.trackers.get(source_name)
        if tracker is None:
            self._reset_tracker_for_source(source_name, runtime)
            with self._infer_lock:
                tracker = self.trackers.get(source_name)
            if tracker is None:
                return self._extract_person_boxes(result)

        try:
            tracks = tracker.update(raw_boxes.cpu().numpy(), img=frame)
        except Exception as exc:  # noqa: BLE001
            self._queue_async_notice(f"[warn] ByteTrack failed on '{source_name}', fallback to raw detections: {exc}")
            self._reset_tracker_for_source(source_name, runtime)
            return self._extract_person_boxes(result)

        boxes: list[TrackedBox] = []
        if tracks is None or len(tracks) == 0:
            return boxes

        for row in tracks.tolist():
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
            boxes.append((ix1, iy1, ix2, iy2, float(score), parsed_track_id))
        return boxes

    def _draw_person_boxes(
        self,
        frame: np.ndarray,
        boxes: list[TrackedBox] | None,
    ) -> np.ndarray:
        output = frame.copy()
        if not boxes:
            return output

        for box in boxes:
            if len(box) >= 6:
                x1, y1, x2, y2, conf, track_id = box[:6]
            else:
                x1, y1, x2, y2, conf = box[:5]
                track_id = None
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 80, 40), 2)
            if track_id is None:
                label = f"person {conf:.2f}"
            else:
                label = f"person#{track_id} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_y = max(18, y1 - 6)
            top = max(0, text_y - th - baseline - 4)
            cv2.rectangle(
                output,
                (x1, top),
                (x1 + tw + 8, text_y + 2),
                (255, 80, 40),
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 4, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
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
        alert: bool,
        boxes: list[TrackedBox] | None = None,
    ) -> np.ndarray:
        color = (0, 0, 255) if alert else (0, 185, 0)
        status = "ALERT" if alert else "OK"
        text = f"{source_name} | mode:{mode} | person:{person_count} | {status}"

        output = self._draw_person_boxes(frame, boxes)
        cv2.putText(output, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
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
        meta = (
            f"{runtime.status} | mode:{runtime.mode} | person:{runtime.person_count} "
            f"| src:{source_fps:.1f} | view:{view_fps:.1f} | ai:{runtime.infer_fps:.1f}"
        )

        tile = self.tiles.get(source_name)
        if tile is not None:
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
            self.fullscreen_window.set_frame(runtime.last_output, zoom=zoom, pan_x=pan_x, pan_y=pan_y)

    def _choose_sources_for_inference(
        self,
        source_names: list[str],
        infer_due: dict[str, bool],
    ) -> set[str]:
        if not source_names:
            return set()

        total = len(source_names)
        start = self._infer_rr_cursor % total
        ordered = [source_names[(start + idx) % total] for idx in range(total)]
        self._infer_rr_cursor = (start + 1) % total

        selected: list[str] = []
        for source_name in ordered:
            if not infer_due.get(source_name, False):
                continue
            selected.append(source_name)
            if len(selected) >= self.max_infer_per_tick:
                break
        return set(selected)

    def _tick_live(self) -> None:
        now_ts = time.perf_counter()
        self._maybe_adjust_live_timer_interval(now_ts)

        if not self._apply_async_inference_updates():
            return

        enabled_sources = self._get_enabled_sources()
        if not enabled_sources:
            return

        infer_interval_sec = 1.0 / max(0.1, self.model_target_fps)
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
                or self.focused_source == source_name
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
                alert=runtime.alert,
                boxes=boxes,
            )
            self._maybe_capture_event_snapshot(
                source_name=source_name,
                runtime=runtime,
                raw_frame=frame,
                decorated_frame=runtime.last_output,
            )
            runtime.last_decorated_capture_seq = current_capture_seq
            runtime.last_decorated_infer_ts = runtime.last_infer_ts

            self._refresh_tile(source_name)

    def start_live(self) -> None:
        if self.live_running:
            return

        self._apply_controls_to_runtime_state()
        self._rebuild_predict_kwargs()
        self._infer_rr_cursor = 0
        with self._infer_lock:
            self._infer_last_submit_ts.clear()
        self._start_inference_worker()
        self._live_timer_interval_ms = self._compute_live_timer_interval_ms()
        self._live_timer_last_adjust_ts = time.perf_counter()
        self.live_timer.start(self._live_timer_interval_ms)
        self.live_running = True
        self.start_live_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(True)
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
        self.start_live_btn.setEnabled(True)
        self.stop_live_btn.setEnabled(False)

        stop_ts = time.perf_counter()
        for source_name, runtime in self.runtimes.items():
            self._finalize_event_clip(source_name, runtime, stop_ts)
            runtime.release()
        with self._infer_lock:
            self.trackers.clear()
            self._infer_last_submit_ts.clear()

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
                        "visible_sec": 0.0,
                        "alert": False,
                        "file": _to_relative_or_abs(file_path),
                    }
                )
            if self._enforce_event_retention_limit():
                self._save_event_entries_index()
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
                    "visible_sec": float(raw.get("visible_sec", 0.0) or 0.0),
                    "alert": bool(raw.get("alert", False)),
                    "file": _to_relative_or_abs(file_path),
                }
            )

        loaded_entries.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        self.event_entries = loaded_entries
        if self._enforce_event_retention_limit():
            self._save_event_entries_index()
        if hasattr(self, "events_table"):
            self._refresh_events_table()

    def _save_event_entries_index(self) -> None:
        self.events_output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "events": self.event_entries}
        self.events_index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _refresh_events_table(self, select_newest: bool = False) -> None:
        if not hasattr(self, "events_table"):
            return

        self._event_table_updating = True
        try:
            ordered = list(enumerate(self.event_entries))
            ordered.reverse()
            self.events_table.setRowCount(len(ordered))
            for row, (entry_index, entry) in enumerate(ordered):
                timestamp = float(entry.get("timestamp", 0.0) or 0.0)
                if timestamp > 0:
                    time_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                else:
                    time_text = "-"
                source_text = str(entry.get("source", "source"))
                mode_text = str(entry.get("mode", "day"))
                persons_text = str(int(entry.get("persons", 0) or 0))
                visible_text = f"{float(entry.get('visible_sec', 0.0) or 0.0):.1f}"
                file_text = str(entry.get("file", ""))

                values = [time_text, source_text, mode_text, persons_text, visible_text, file_text]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, int(entry_index))
                    self.events_table.setItem(row, col, item)

            if hasattr(self, "events_status_label"):
                self.events_status_label.setText(f"Saved events: {len(self.event_entries)}")
        finally:
            self._event_table_updating = False

        if select_newest and self.events_table.rowCount() > 0:
            self.events_table.setCurrentCell(0, 0)
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
        self._refresh_events_table()
        self._log("All saved events were removed.")

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
        runtime.event_clip_started_wall_ts = 0.0
        return temp_path

    def _finalize_event_clip(self, source_name: str, runtime: SourceRuntime, now_ts: float) -> None:
        has_pending_clip = (
            runtime.event_clip_writer is not None
            or runtime.event_clip_temp_path is not None
            or runtime.event_clip_frames_written > 0
        )
        if not has_pending_clip:
            return

        frames_written = int(runtime.event_clip_frames_written)
        wall_ts = runtime.event_clip_started_wall_ts if runtime.event_clip_started_wall_ts > 0.0 else time.time()
        visible_sec = float(runtime.person_visible_duration_sec)
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
            self._queue_async_notice(f"[warn] failed to finalize event clip for '{source_name}'")
            return

        entry = {
            "timestamp": float(wall_ts),
            "source": source_name,
            "mode": runtime.mode,
            "persons": int(runtime.person_count),
            "visible_sec": visible_sec,
            "alert": bool(runtime.alert),
            "file": _to_relative_or_abs(output_path),
        }
        self.event_entries.append(entry)
        self._enforce_event_retention_limit()
        self._save_event_entries_index()

        runtime.last_event_capture_ts = now_ts
        runtime.event_saved_in_streak = True

        show_latest = bool(
            hasattr(self, "main_tabs")
            and hasattr(self, "events_tab_page")
            and self.main_tabs.currentWidget() is self.events_tab_page
        )
        self._refresh_events_table(select_newest=show_latest)
        self._log(
            f"Event saved: {source_name}, visible={visible_sec:.1f}s, "
            f"persons={runtime.person_count}, file={output_path}"
        )

    def _ensure_event_clip_writer(self, source_name: str, runtime: SourceRuntime, frame: np.ndarray) -> bool:
        if runtime.event_clip_writer is not None and runtime.event_clip_temp_path is not None:
            return True

        self.events_output_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        frame_size = (int(width), int(height))
        fps_candidates = [
            float(runtime.source_fps),
            float(runtime.fps),
            float(self.runtime_cfg.get("video_fps_fallback", 25.0)),
            float(self.model_target_fps),
            float(self.view_target_fps),
        ]
        clip_fps = next((fps for fps in fps_candidates if fps > 1e-3), 25.0)

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
            self._queue_async_notice(f"[warn] failed to initialize event clip writer for '{source_name}'")
            self._clear_event_clip_state(runtime, delete_temp_file=True)
            return False

        runtime.event_clip_writer = writer
        runtime.event_clip_temp_path = temp_path
        runtime.event_clip_frame_size = frame_size
        runtime.event_clip_frames_written = 0
        return True

    def _update_event_visibility_state(self, source_name: str, runtime: SourceRuntime, now_ts: float) -> None:
        person_present = runtime.person_count >= self.events_min_person_count and runtime.last_infer_ts > 0
        if person_present:
            if runtime.person_visible_since_ts <= 0.0:
                runtime.person_visible_since_ts = now_ts
                runtime.person_visible_duration_sec = 0.0
                runtime.event_saved_in_streak = False
                runtime.event_clip_started_wall_ts = time.time()
            else:
                runtime.person_visible_duration_sec = max(0.0, now_ts - runtime.person_visible_since_ts)
            return

        if runtime.person_visible_since_ts > 0.0:
            runtime.person_visible_duration_sec = max(0.0, now_ts - runtime.person_visible_since_ts)
            self._finalize_event_clip(source_name, runtime, now_ts)
        else:
            self._clear_event_clip_state(runtime, delete_temp_file=True)

        runtime.person_visible_since_ts = 0.0
        runtime.person_visible_duration_sec = 0.0
        runtime.event_saved_in_streak = False

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

        if not self._ensure_event_clip_writer(source_name, runtime, clip_frame):
            return

        target_size = runtime.event_clip_frame_size or (clip_frame.shape[1], clip_frame.shape[0])
        frame_to_write = clip_frame
        if frame_to_write.shape[1] != target_size[0] or frame_to_write.shape[0] != target_size[1]:
            frame_to_write = cv2.resize(frame_to_write, target_size, interpolation=cv2.INTER_LINEAR)

        if runtime.event_clip_writer is None:
            return

        try:
            runtime.event_clip_writer.write(frame_to_write)
            runtime.event_clip_frames_written += 1
        except Exception:  # noqa: BLE001
            self._queue_async_notice(f"[warn] failed to write event clip frame for '{source_name}'")
            self._clear_event_clip_state(runtime, delete_temp_file=True)

    # ---------- logs ----------
    def _clear_logs(self) -> None:
        self._log_entries.clear()
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

        self.inference_cfg["conf"] = float(self.conf_spin.value())
        self.inference_cfg["iou"] = float(self.iou_spin.value())
        self.inference_cfg["imgsz"] = int(self.imgsz_spin.value())
        self.inference_cfg["max_det"] = int(self.max_det_spin.value())

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
        self.runtime_cfg["model_target_fps"] = float(self.model_target_fps)
        self.runtime_cfg["max_infer_per_tick"] = int(self.max_infer_per_tick)
        self.runtime_cfg["live_tile_spacing"] = int(self.live_tile_spacing)
        self.runtime_cfg["console_logs"] = bool(self.console_logs_enabled)
        self.runtime_cfg["suppress_opencv_warnings"] = bool(self.suppress_opencv_warnings)
        self.runtime_cfg["auto_scan_cameras_on_startup"] = bool(self.auto_scan_cameras_on_startup)
        self.runtime_cfg["auto_start_live"] = bool(self.auto_start_live)

        previous_output_dir = self.events_output_dir
        previous_output_raw = self.events_output_dir_raw

        self.events_cfg["enabled"] = bool(self.events_enabled_checkbox.isChecked())
        self.events_cfg["min_visible_seconds"] = float(self.events_min_visible_spin.value())
        self.events_cfg["cooldown_seconds"] = float(self.events_cooldown_spin.value())
        self.events_cfg["min_person_count"] = int(self.events_min_person_spin.value())
        self.events_cfg["max_saved_events"] = int(self.events_max_saved_spin.value())
        self.events_cfg["save_annotated_frame"] = bool(self.events_save_annotated_checkbox.isChecked())
        self.events_cfg["once_per_streak"] = bool(self.events_once_per_streak_checkbox.isChecked())
        output_dir_raw = self.events_output_dir_edit.text().strip() or "logs/app/events"
        self.events_cfg["output_dir"] = output_dir_raw

        self.events_enabled = bool(self.events_cfg["enabled"])
        self.events_min_visible_seconds = max(0.1, float(self.events_cfg["min_visible_seconds"]))
        self.events_cooldown_seconds = max(0.0, float(self.events_cfg["cooldown_seconds"]))
        self.events_min_person_count = max(1, int(self.events_cfg["min_person_count"]))
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

        if self.current_model_path is not None and self.current_model_path.exists():
            self.model_cfg["selected_model_path"] = _to_relative_or_abs(self.current_model_path)

    def _persist_config(self, *, show_message: bool) -> None:
        self._apply_controls_to_runtime_state()

        self.config["model"] = dict(self.model_cfg)
        self.config["inference"] = dict(self.inference_cfg)
        self.config["tracker"] = dict(self.tracker_cfg)
        self.config["security"] = dict(self.security_cfg)
        self.config["events"] = dict(self.events_cfg)
        self.config["runtime"] = dict(self.runtime_cfg)
        self.config.pop("sources", None)

        save_yaml(self.config_path, self.config)
        try:
            self._save_sources_config()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[warn] Unable to save sources config: {exc}")

        if show_message:
            QMessageBox.information(
                self,
                "Config",
                f"Settings saved:\n{self.config_path}\nSources saved:\n{self.sources_settings_path}",
            )

    # ---------- Qt hooks ----------
    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: ANN401
        if event.type() == QEvent.Type.Resize:
            if watched is self.live_scroll.viewport():
                self._rebuild_live_layout()
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
        self._close_fullscreen_source()
        self.stop_live()
        self._stop_inference_worker()
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
        cameras = scan_available_cameras(max_index=max_index)
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
