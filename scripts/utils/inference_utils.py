from __future__ import annotations

import ctypes
from datetime import datetime
from math import ceil
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config_utils import resolve_path


def _to_windows_short_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
        if result and result < len(buffer):
            short_value = str(buffer.value).strip()
            if short_value:
                return Path(short_value)
    except Exception:  # noqa: BLE001
        pass
    return path


def _open_video_with_backend(video_path: Path, backend: int | None) -> cv2.VideoCapture | None:
    path_str = str(video_path)
    capture = cv2.VideoCapture(path_str) if backend is None else cv2.VideoCapture(path_str, backend)
    if not capture.isOpened():
        capture.release()
        return None

    # Validate that decoder can return a frame, not only "open".
    ok, _ = capture.read()
    if not ok:
        # Some codecs need a few reads before a valid frame.
        for _ in range(4):
            ok, _ = capture.read()
            if ok:
                break
    if not ok:
        capture.release()
        return None

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture


def _camera_backends() -> list[int | None]:
    return _camera_backend_candidates()


def _camera_backend_value(name: str) -> int | None:
    normalized = str(name or "").strip().lower()
    if normalized in {"", "auto", "default"}:
        return None

    aliases = {
        "dshow": "CAP_DSHOW",
        "directshow": "CAP_DSHOW",
        "msmf": "CAP_MSMF",
        "mediafoundation": "CAP_MSMF",
        "v4l2": "CAP_V4L2",
        "any": "CAP_ANY",
    }
    backend_attr = aliases.get(normalized, normalized.upper())
    if not backend_attr.startswith("CAP_"):
        backend_attr = f"CAP_{backend_attr}"
    if hasattr(cv2, backend_attr):
        return int(getattr(cv2, backend_attr))
    return None


def _camera_backend_candidates(preferred_backend: str | None = None) -> list[int | None]:
    preferred = str(preferred_backend or "").strip().lower()
    if preferred in {"any", "cap_any"}:
        return [None]

    candidates: list[int | None] = []
    preferred_value = _camera_backend_value(preferred)
    if preferred and preferred_value is not None:
        return [preferred_value]

    if os.name == "nt":
        # MSMF usually opens webcams quietly on Windows. Falling back to CAP_ANY can route
        # through FFmpeg/dshow and produce noisy "Failed list devices" warnings.
        for backend_name in ("CAP_MSMF", "CAP_DSHOW"):
            if hasattr(cv2, backend_name):
                backend_value = int(getattr(cv2, backend_name))
                if backend_value not in candidates:
                    candidates.append(backend_value)
        return candidates

    if hasattr(cv2, "CAP_V4L2"):
        v4l2_value = int(getattr(cv2, "CAP_V4L2"))
        if v4l2_value not in candidates:
            candidates.append(v4l2_value)
    candidates.append(None)
    return candidates


def camera_capture_can_read(capture: cv2.VideoCapture, attempts: int = 6) -> bool:
    for attempt in range(max(1, attempts)):
        try:
            ok, frame = capture.read()
        except cv2.error:
            return False
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True
        if attempt + 1 < attempts:
            time.sleep(0.05)
    return False


def _open_camera_capture(
    camera_index: int,
    *,
    validate_frame: bool = True,
    preferred_backend: str | None = None,
) -> cv2.VideoCapture:
    for backend in _camera_backend_candidates(preferred_backend):
        capture = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
        if not capture.isOpened():
            capture.release()
            continue
        if not validate_frame or camera_capture_can_read(capture):
            return capture
        capture.release()

    return cv2.VideoCapture()


def scan_available_cameras(max_index: int = 8, *, preferred_backend: str | None = None) -> list[int]:
    available: list[int] = []
    for camera_index in range(max_index + 1):
        capture = _open_camera_capture(camera_index, validate_frame=True, preferred_backend=preferred_backend)

        if capture.isOpened():
            available.append(camera_index)
        capture.release()

    return available


def open_video_file_capture(path_value: str | Path) -> cv2.VideoCapture:
    video_path = resolve_path(path_value)
    candidate_paths = [video_path]

    # Fallback for stale/encoded absolute paths: try local data/videos/<filename>.
    fallback_by_name = resolve_path(Path("data/videos") / video_path.name)
    if fallback_by_name != video_path and fallback_by_name.exists():
        candidate_paths.append(fallback_by_name)

    if os.name == "nt":
        short_path = _to_windows_short_path(video_path)
        if short_path != video_path:
            candidate_paths.append(short_path)
        short_fallback = _to_windows_short_path(fallback_by_name)
        if short_fallback not in candidate_paths and short_fallback.exists():
            candidate_paths.append(short_fallback)

    backends: list[int | None] = [None]
    if hasattr(cv2, "CAP_FFMPEG"):
        backends.append(cv2.CAP_FFMPEG)

    for candidate in candidate_paths:
        for backend in backends:
            capture = _open_video_with_backend(candidate, backend)
            if capture is not None:
                return capture

    return cv2.VideoCapture(str(video_path))


def open_capture(source: dict[str, Any]) -> cv2.VideoCapture:
    source_type = str(source.get("type", "video")).lower()
    raw_value = source.get("value")

    if source_type == "camera":
        camera_index = int(raw_value)
        preferred_backend = source.get("backend", source.get("camera_backend"))
        return _open_camera_capture(camera_index, preferred_backend=preferred_backend)

    if source_type == "video":
        capture = open_video_file_capture(str(raw_value))
        if bool(source.get("random_start", False)):
            _seek_random_start(capture)
        return capture

    # stream type (rtsp/http/etc.)
    return cv2.VideoCapture(str(raw_value))


def _seek_random_start(capture: cv2.VideoCapture) -> None:
    try:
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count > 1:
            target = random.randint(0, int(frame_count) - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(target))
            return

        # Fallback: try random ratio if frame count is missing.
        if capture.set(cv2.CAP_PROP_POS_AVI_RATIO, random.random()):
            return

        # Last resort: seek to random ms within first 30s if fps is known.
        if fps > 1.0:
            max_ms = int(30000)
            capture.set(cv2.CAP_PROP_POS_MSEC, float(random.randint(0, max_ms)))
    except Exception:  # noqa: BLE001
        return


def count_detections_for_class(result: Any, class_id: int = 0) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return 0
    return sum(1 for value in boxes.cls.tolist() if int(value) == class_id)


def resolve_security_mode(security_cfg: dict[str, Any]) -> str:
    mode = str(security_cfg.get("mode", "auto")).lower()
    if mode in {"day", "night"}:
        return mode

    start_hour = int(security_cfg.get("night_start_hour", 22))
    end_hour = int(security_cfg.get("night_end_hour", 6))
    current_hour = datetime.now().hour

    if start_hour <= end_hour:
        is_night = start_hour <= current_hour < end_hour
    else:
        is_night = current_hour >= start_hour or current_hour < end_hour

    return "night" if is_night else "day"


def should_raise_alert(person_count: int, mode: str, security_cfg: dict[str, Any]) -> bool:
    if mode == "night":
        threshold = int(security_cfg.get("night_person_threshold", 1))
    else:
        threshold = int(security_cfg.get("day_person_threshold", 1))
    return person_count >= threshold


def build_frame_grid(
    frames: list[np.ndarray],
    columns: int = 2,
    tile_width: int = 640,
    tile_height: int = 360,
) -> np.ndarray | None:
    if not frames:
        return None

    columns = max(1, columns)
    tiles = [cv2.resize(frame, (tile_width, tile_height)) for frame in frames]
    rows = ceil(len(tiles) / columns)

    blank_tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    while len(tiles) < rows * columns:
        tiles.append(blank_tile.copy())

    row_images: list[np.ndarray] = []
    for row_index in range(rows):
        row_start = row_index * columns
        row_end = row_start + columns
        row_images.append(np.hstack(tiles[row_start:row_end]))

    return np.vstack(row_images)
