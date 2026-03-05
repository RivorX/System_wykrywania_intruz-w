from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm
from ultralytics import YOLO

from .config_utils import ensure_dir, resolve_path


def download_file(
    url: str,
    destination: Path,
    timeout: int = 120,
    show_progress: bool = True,
    chunk_size: int = 1024 * 1024,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            if show_progress:
                total = int(response.headers.get("content-length", 0))
                progress_total = total if total > 0 else None
                with tqdm(
                    total=progress_total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {destination.name}",
                ) as progress:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        file.write(chunk)
                        progress.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    file.write(chunk)


def ensure_model_reference(model_cfg: dict[str, Any], base_dir: Path | None = None) -> str:
    model_name = str(model_cfg.get("name", "yolo26m.pt")).strip()
    if not model_name:
        raise ValueError("Model name cannot be empty.")

    weights_dir = ensure_dir(model_cfg.get("local_weights_dir", "models/base"), base_dir=base_dir)
    local_model_path = weights_dir / model_name

    if bool(model_cfg.get("force_redownload", False)) and local_model_path.exists():
        local_model_path.unlink()

    if local_model_path.exists():
        return str(local_model_path)

    auto_download = bool(model_cfg.get("auto_download", True))
    model_url = model_cfg.get("download_url")

    if auto_download and model_url:
        print(f"[model] Downloading {model_name} from configured URL.")
        download_file(str(model_url), local_model_path)
        return str(local_model_path)

    # Returning model_name allows Ultralytics to auto-download official weights.
    return model_name


def _candidate_checkpoint_paths(reference: str, model: YOLO) -> list[Path]:
    paths: list[Path] = []

    reference_path = Path(reference)
    if reference_path.exists() and reference_path.is_file():
        paths.append(reference_path.resolve())

    for attr in ("ckpt_path", "pt_path"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            candidate = Path(value)
            if candidate.exists() and candidate.is_file():
                paths.append(candidate.resolve())

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _persist_model_to_local_dir(
    model_cfg: dict[str, Any],
    reference: str,
    model: YOLO,
    base_dir: Path | None = None,
) -> str:
    model_name = str(model_cfg.get("name", "yolo26m.pt")).strip()
    if not model_name:
        return reference

    weights_dir = ensure_dir(model_cfg.get("local_weights_dir", "models/base"), base_dir=base_dir)
    target_path = (weights_dir / model_name).resolve()
    if target_path.exists():
        return str(target_path)

    for source_path in _candidate_checkpoint_paths(reference, model):
        if source_path == target_path:
            return str(target_path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            print(f"[model] Saved base model to {target_path}")
            return str(target_path)
        except Exception:  # noqa: BLE001
            continue

    return reference


def load_yolo_model(model_cfg: dict[str, Any], base_dir: Path | None = None) -> tuple[YOLO, str]:
    primary_reference = ensure_model_reference(model_cfg, base_dir=base_dir)
    fallback_name = str(model_cfg.get("fallback_name", "")).strip()
    use_fallback = bool(model_cfg.get("use_fallback", False))

    candidates = [primary_reference]
    if use_fallback and fallback_name:
        candidates.append(fallback_name)

    errors: list[str] = []
    for reference in candidates:
        try:
            model = YOLO(reference)
            persisted_reference = _persist_model_to_local_dir(
                model_cfg=model_cfg,
                reference=reference,
                model=model,
                base_dir=base_dir,
            )
            return model, persisted_reference
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{reference}: {exc}")

    detail = " | ".join(errors)
    raise RuntimeError(
        "Unable to load YOLO model. If YOLO26 weights are not available in Ultralytics yet, "
        "set model.download_url to direct .pt weights or provide a local file in models/base. "
        f"Details: {detail}"
    )
