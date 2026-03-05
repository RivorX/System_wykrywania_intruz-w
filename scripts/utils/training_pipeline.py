from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .batch_size_utils import resolve_smart_batch_size
from .config_utils import ensure_dir, load_yaml, resolve_path, save_json, utc_timestamp
from .dataset_utils import prepare_dataset_from_file
from .model_utils import load_yolo_model
from .runtime_env_utils import ensure_windows_compile_env


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:  # noqa: BLE001
        return None


def _read_run_quality_metrics(run_dir: Path) -> dict[str, Any]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {
            "map50": None,
            "map5095": None,
            "val_loss": None,
            "quality_metric": None,
            "quality_score": None,
        }

    try:
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        rows = []

    best_map50: float | None = None
    best_map5095: float | None = None
    best_val_loss: float | None = None

    for row in rows:
        map50 = _safe_float(row.get("metrics/mAP50(B)") or row.get("metrics/mAP50"))
        if map50 is not None:
            if best_map50 is None or map50 > best_map50:
                best_map50 = map50

        map5095 = _safe_float(row.get("metrics/mAP50-95(B)") or row.get("metrics/mAP50-95"))
        if map5095 is not None:
            if best_map5095 is None or map5095 > best_map5095:
                best_map5095 = map5095

        val_box = _safe_float(row.get("val/box_loss"))
        val_cls = _safe_float(row.get("val/cls_loss"))
        val_dfl = _safe_float(row.get("val/dfl_loss"))
        loss_parts = [value for value in (val_box, val_cls, val_dfl) if value is not None]
        if loss_parts:
            val_loss = float(sum(loss_parts))
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss

    quality_metric: str | None = None
    quality_score: float | None = None
    if best_map5095 is not None:
        quality_metric = "mAP50-95"
        quality_score = best_map5095
    elif best_map50 is not None:
        quality_metric = "mAP50"
        quality_score = best_map50
    elif best_val_loss is not None:
        quality_metric = "val_loss"
        quality_score = -best_val_loss

    return {
        "map50": best_map50,
        "map5095": best_map5095,
        "val_loss": best_val_loss,
        "quality_metric": quality_metric,
        "quality_score": quality_score,
    }


def _normalize_device(device_value: Any) -> Any:
    if device_value is None:
        return None
    if isinstance(device_value, str) and device_value.strip().lower() == "auto":
        return None
    return device_value


def _is_cuda_device_requested(device_value: Any) -> bool:
    if device_value is None:
        return False
    if isinstance(device_value, int):
        return device_value >= 0
    normalized = str(device_value).strip().lower()
    return normalized in {"0", "cuda", "cuda:0"}


def _resolve_run_prefix(
    *,
    project_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    model_reference: str,
) -> str:
    configured_prefix = str(project_cfg.get("runs_prefix", "auto")).strip()
    if configured_prefix and configured_prefix.lower() not in {"auto", "model", "from_model"}:
        return configured_prefix

    model_name = str(model_cfg.get("name", "")).strip()
    model_stem = Path(model_name).stem if model_name else Path(model_reference).stem
    if not model_stem:
        model_stem = "train"
    return f"{model_stem}_train"


def _resolve_batch_value(
    *,
    batch_cfg: Any,
    training_cfg: dict[str, Any],
    model_reference: str,
    dataset_reference: str,
    imgsz: int,
    device_value: Any,
    classes: list[int] | None,
    compile_value: Any,
) -> int:
    if isinstance(batch_cfg, str) and batch_cfg.strip().lower() in {"auto", "smart"}:
        if not torch.cuda.is_available():
            cpu_batch = int(training_cfg.get("cpu_batch_fallback", 2))
            print(f"[batch-auto] CUDA not available. Using cpu_batch_fallback={cpu_batch}")
            return max(1, cpu_batch)

        auto_batch_cfg = training_cfg.get("auto_batch", {})
        return resolve_smart_batch_size(
            model_reference=model_reference,
            dataset_reference=dataset_reference,
            imgsz=imgsz,
            device=device_value,
            classes=classes,
            compile_value=compile_value,
            auto_batch_cfg=auto_batch_cfg,
        )

    return int(batch_cfg)


def _apply_augmentation_overrides(
    *,
    train_args: dict[str, Any],
    training_cfg: dict[str, Any],
) -> None:
    augmentation_cfg = training_cfg.get("augmentation", {})
    if not isinstance(augmentation_cfg, dict):
        return

    allowed_keys = (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "fliplr",
        "flipud",
        "mosaic",
        "mixup",
        "copy_paste",
        "erasing",
    )
    for key in allowed_keys:
        if key in augmentation_cfg and augmentation_cfg[key] is not None:
            train_args[key] = augmentation_cfg[key]


def _configure_ultralytics_datasets_dir(*, training_cfg: dict[str, Any]) -> None:
    if not bool(training_cfg.get("force_project_datasets_dir", True)):
        return

    datasets_dir = ensure_dir(training_cfg.get("datasets_dir", "data/raw/ultralytics_datasets"))
    try:
        from ultralytics import settings as ultralytics_settings
    except Exception as exc:  # noqa: BLE001
        print(f"[env] Unable to import ultralytics settings ({exc}). Skipping datasets_dir override.")
        return

    try:
        current_value = str(ultralytics_settings.get("datasets_dir", "")).strip()
    except Exception:  # noqa: BLE001
        current_value = ""

    try:
        current_path = Path(current_value).resolve() if current_value else None
    except Exception:  # noqa: BLE001
        current_path = None

    if current_path is not None and current_path == datasets_dir.resolve():
        return

    try:
        ultralytics_settings.update({"datasets_dir": str(datasets_dir)})
        print(f"[env] Ultralytics datasets_dir set to: {datasets_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[env] Unable to update ultralytics datasets_dir ({exc}).")


def _count_images_in_dir(path: Path) -> int:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if not path.exists():
        return 0
    if path.is_file():
        return 1 if path.suffix.lower() in valid_exts else 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in valid_exts)


def _resolve_dataset_split_path(dataset_yaml_path: Path, dataset_cfg: dict[str, Any], split_name: str) -> Path | None:
    split_raw = str(dataset_cfg.get(split_name, "")).strip()
    if not split_raw:
        return None
    split_path = Path(split_raw)

    base_raw = str(dataset_cfg.get("path", "")).strip()
    if base_raw:
        base_path = Path(base_raw)
        if not base_path.is_absolute():
            base_path = (dataset_yaml_path.parent / base_path).resolve()
        if not split_path.is_absolute():
            split_path = base_path / split_path
    elif not split_path.is_absolute():
        split_path = (dataset_yaml_path.parent / split_path).resolve()

    return split_path.resolve()


def _print_dataset_split_summary(*, dataset_reference: str, training_cfg: dict[str, Any]) -> None:
    if not bool(training_cfg.get("print_dataset_split_summary", True)):
        return

    dataset_yaml_path = Path(dataset_reference)
    if not dataset_yaml_path.exists():
        return
    if dataset_yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        return

    try:
        dataset_cfg = load_yaml(dataset_yaml_path)
    except Exception:  # noqa: BLE001
        return
    if not isinstance(dataset_cfg, dict):
        return

    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_path = _resolve_dataset_split_path(dataset_yaml_path, dataset_cfg, split_name)
        if split_path is None:
            continue
        counts[split_name] = _count_images_in_dir(split_path)

    if counts:
        summary = ", ".join(f"{name}={value}" for name, value in counts.items())
        print(f"[data] Split image counts: {summary}")
        print("[data] Note: 'Image sizes ... train, ... val' means resolution (imgsz), not number of images.")


def _attach_results_plot_snapshot_callback(
    *,
    model: Any,
    logs_dir: Path,
    run_name: str,
    training_cfg: dict[str, Any],
) -> None:
    plot_cfg = training_cfg.get("results_plot", {})
    if not isinstance(plot_cfg, dict):
        return
    if not bool(plot_cfg.get("enabled", True)):
        return

    every_n_epochs = max(1, int(plot_cfg.get("every_n_epochs", 5)))
    snapshots_subdir = str(plot_cfg.get("snapshots_subdir", "plots")).strip() or "plots"
    max_snapshots = int(plot_cfg.get("max_snapshots", 0))
    run_dir = (logs_dir / run_name).resolve()
    snapshots_dir = run_dir / snapshots_subdir
    saved_epochs: set[int] = set()

    def _on_fit_epoch_end(trainer: Any) -> None:
        epoch_value = int(getattr(trainer, "epoch", -1)) + 1
        if epoch_value <= 0:
            return
        if epoch_value in saved_epochs:
            return

        total_epochs = int(getattr(trainer, "epochs", 0))
        is_final_epoch = total_epochs > 0 and epoch_value >= total_epochs
        if not is_final_epoch and (epoch_value % every_n_epochs != 0):
            return

        try:
            trainer.plot_metrics()
        except Exception as exc:  # noqa: BLE001
            print(f"[plots] Skipping snapshot for epoch {epoch_value}: {exc}")
            return

        results_png = Path(trainer.save_dir) / "results.png"
        if not results_png.exists():
            return

        snapshots_dir.mkdir(parents=True, exist_ok=True)
        destination = snapshots_dir / f"results_epoch{epoch_value:04d}.png"
        shutil.copy2(results_png, destination)
        saved_epochs.add(epoch_value)
        print(f"[plots] Snapshot saved: {destination}")

        if max_snapshots > 0:
            existing = sorted(snapshots_dir.glob("results_epoch*.png"))
            overflow = len(existing) - max_snapshots
            if overflow > 0:
                for old_file in existing[:overflow]:
                    old_file.unlink(missing_ok=True)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)


def _export_trained_weights(
    *,
    run_dir: Path,
    run_name: str,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    run_quality: dict[str, Any],
) -> list[Path]:
    export_cfg = training_cfg.get("export_weights", {})
    if not isinstance(export_cfg, dict):
        return []
    if not bool(export_cfg.get("enabled", True)):
        return []

    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        return []

    export_dir = ensure_dir(export_cfg.get("export_dir", "models/weights"))
    latest_subdir = str(export_cfg.get("latest_subdir", "latest")).strip()
    latest_dir = (export_dir / latest_subdir).resolve() if latest_subdir else export_dir.resolve()
    latest_dir.mkdir(parents=True, exist_ok=True)
    include_run_named = bool(export_cfg.get("include_run_named", False))

    copied: list[Path] = []
    model_name = str(model_cfg.get("name", "model_best.pt")).strip() or "model_best.pt"
    if not model_name.lower().endswith(".pt"):
        model_name = f"{model_name}.pt"
    canonical_target = (export_dir / model_name).resolve()
    canonical_meta_path = Path(f"{canonical_target}.meta.json")
    registry_path = (export_dir / "model_registry.json").resolve()

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    except Exception:  # noqa: BLE001
        registry = {}
    if not isinstance(registry, dict):
        registry = {}
    model_registry = registry.get("models")
    if not isinstance(model_registry, dict):
        model_registry = {}

    for weight_name in ("best.pt", "last.pt"):
        source = (weights_dir / weight_name).resolve()
        if not source.exists():
            continue

        latest_target = (latest_dir / weight_name).resolve()
        shutil.copy2(source, latest_target)
        copied.append(latest_target)

        latest_meta = {
            "run_name": run_name,
            "model_name": model_name,
            "weight_name": weight_name,
            "path": str(latest_target),
            "map50": run_quality.get("map50"),
            "map5095": run_quality.get("map5095"),
            "val_loss": run_quality.get("val_loss"),
            "quality_metric": run_quality.get("quality_metric"),
            "quality_score": run_quality.get("quality_score"),
            "updated_ts": latest_target.stat().st_mtime,
        }
        save_json(Path(f"{latest_target}.meta.json"), latest_meta)

        if include_run_named:
            run_target = (export_dir / f"{run_name}_{weight_name}").resolve()
            shutil.copy2(source, run_target)
            copied.append(run_target)

    best_source = (weights_dir / "best.pt").resolve()
    if best_source.exists():
        current_entry = model_registry.get(model_name)
        if not isinstance(current_entry, dict) and canonical_meta_path.exists():
            try:
                meta_payload = json.loads(canonical_meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta_payload = None
            if isinstance(meta_payload, dict):
                current_entry = meta_payload
        current_score = _safe_float(current_entry.get("quality_score")) if isinstance(current_entry, dict) else None
        new_score = _safe_float(run_quality.get("quality_score"))

        should_update = False
        reason = ""
        if not canonical_target.exists():
            should_update = True
            reason = "first-best"
        elif current_entry is None:
            should_update = False
            reason = "existing-unknown-keep"
        elif new_score is None:
            should_update = False
            reason = "missing-quality"
        elif current_score is None:
            should_update = True
            reason = "replace-unknown-score"
        elif new_score > current_score:
            should_update = True
            reason = "better-quality"
        else:
            should_update = False
            reason = "not-better"

        if should_update:
            shutil.copy2(best_source, canonical_target)
            copied.append(canonical_target)
            model_meta = {
                "run_name": run_name,
                "model_name": model_name,
                "weight_name": "best.pt",
                "path": str(canonical_target),
                "map50": run_quality.get("map50"),
                "map5095": run_quality.get("map5095"),
                "val_loss": run_quality.get("val_loss"),
                "quality_metric": run_quality.get("quality_metric"),
                "quality_score": new_score,
                "updated_ts": canonical_target.stat().st_mtime,
                "update_reason": reason,
            }
            save_json(canonical_meta_path, model_meta)
            model_registry[model_name] = {
                "run_name": run_name,
                "quality_metric": run_quality.get("quality_metric"),
                "quality_score": new_score,
                "map50": run_quality.get("map50"),
                "map5095": run_quality.get("map5095"),
                "val_loss": run_quality.get("val_loss"),
                "path": str(canonical_target),
                "updated_ts": canonical_target.stat().st_mtime,
            }
            print(
                f"[train] Updated canonical best model: {canonical_target} "
                f"(metric={run_quality.get('quality_metric')}, score={new_score})"
            )
        else:
            print(
                f"[train] Keeping existing canonical model: {canonical_target} "
                f"(new score={new_score}, current score={current_score})"
            )

    registry["models"] = model_registry
    save_json(registry_path, registry)

    if copied:
        latest_msg = ", ".join(str(path) for path in copied if latest_dir in path.parents or path.parent == latest_dir)
        if latest_msg:
            print(f"[train] Exported latest weights: {latest_msg}")

    return copied


def run_training(config_path: str | Path) -> Path:
    train_cfg = load_yaml(config_path)
    project_cfg = train_cfg.get("project", {})
    model_cfg = train_cfg.get("model", {})
    data_cfg = train_cfg.get("data", {})
    training_cfg = train_cfg.get("training", {})
    _configure_ultralytics_datasets_dir(training_cfg=training_cfg)

    if bool(data_cfg.get("auto_prepare", True)):
        dataset_cfg_path = data_cfg.get("dataset_config_path", "config/dataset.yaml")
        dataset_yaml_path = prepare_dataset_from_file(dataset_cfg_path)
        dataset_reference = str(dataset_yaml_path)
    else:
        dataset_yaml_raw = str(data_cfg.get("dataset_yaml_path", "")).strip()
        if not dataset_yaml_raw:
            raise ValueError("data.dataset_yaml_path must be set when data.auto_prepare is false.")

        # Allow both local YAML paths and Ultralytics built-ins like "coco.yaml".
        if dataset_yaml_raw.startswith(("http://", "https://")):
            dataset_reference = dataset_yaml_raw
        else:
            candidate_path = resolve_path(dataset_yaml_raw)
            if candidate_path.exists():
                dataset_reference = str(candidate_path)
            elif ("/" in dataset_yaml_raw) or ("\\" in dataset_yaml_raw):
                raise FileNotFoundError(f"Dataset YAML not found: {candidate_path}")
            else:
                dataset_reference = dataset_yaml_raw

    model, model_reference = load_yolo_model(model_cfg)

    logs_dir = ensure_dir(project_cfg.get("logs_dir", "logs/train"))
    run_prefix = _resolve_run_prefix(
        project_cfg=project_cfg,
        model_cfg=model_cfg,
        model_reference=model_reference,
    )
    run_name = str(training_cfg.get("run_name") or f"{run_prefix}_{utc_timestamp()}")
    classes_cfg = training_cfg.get("classes")
    classes = [int(class_id) for class_id in classes_cfg] if classes_cfg is not None else None
    compile_value = training_cfg.get("compile")
    ensure_windows_compile_env(training_cfg, compile_value=compile_value)
    device_value = _normalize_device(training_cfg.get("device", "auto"))

    if device_value is not None and _is_cuda_device_requested(device_value) and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests CUDA device, but torch.cuda.is_available() is False. "
            "Install CUDA PyTorch wheels (see README section 'Instalacja (Windows + CUDA)')."
        )

    imgsz = int(training_cfg.get("imgsz", 640))
    batch_value = _resolve_batch_value(
        batch_cfg=training_cfg.get("batch", 8),
        training_cfg=training_cfg,
        model_reference=model_reference,
        dataset_reference=dataset_reference,
        imgsz=imgsz,
        device_value=device_value,
        classes=classes,
        compile_value=compile_value,
    )

    train_args: dict[str, Any] = {
        "data": dataset_reference,
        "epochs": int(training_cfg.get("epochs", 50)),
        "imgsz": imgsz,
        "batch": batch_value,
        "workers": int(training_cfg.get("workers", 4)),
        "patience": int(training_cfg.get("patience", 20)),
        "optimizer": training_cfg.get("optimizer", "auto"),
        "lr0": float(training_cfg.get("lr0", 0.01)),
        "lrf": float(training_cfg.get("lrf", 0.01)),
        "cos_lr": bool(training_cfg.get("cos_lr", False)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0005)),
        "warmup_epochs": float(training_cfg.get("warmup_epochs", 3)),
        "cache": bool(training_cfg.get("cache", False)),
        "exist_ok": bool(training_cfg.get("exist_ok", False)),
        "save_period": int(training_cfg.get("save_period", -1)),
        "plots": bool(training_cfg.get("plots", True)),
        "project": str(logs_dir),
        "name": run_name,
    }

    if compile_value is not None:
        train_args["compile"] = compile_value

    if classes is not None:
        train_args["classes"] = classes

    if device_value is not None:
        train_args["device"] = device_value

    _apply_augmentation_overrides(train_args=train_args, training_cfg=training_cfg)
    _print_dataset_split_summary(dataset_reference=dataset_reference, training_cfg=training_cfg)
    _attach_results_plot_snapshot_callback(
        model=model,
        logs_dir=logs_dir,
        run_name=run_name,
        training_cfg=training_cfg,
    )

    print(f"[train] Starting run '{run_name}' with model '{model_reference}'")
    model.train(**train_args)

    run_dir = (logs_dir / run_name).resolve()
    run_quality = _read_run_quality_metrics(run_dir)
    exported_weights = _export_trained_weights(
        run_dir=run_dir,
        run_name=run_name,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        run_quality=run_quality,
    )
    run_summary = {
        "run_name": run_name,
        "model_reference": model_reference,
        "dataset_yaml": dataset_reference,
        "run_dir": str(run_dir),
        "train_args": train_args,
        "run_quality": run_quality,
        "exported_weights": [str(path) for path in exported_weights],
    }
    save_json(run_dir / "run_summary.json", run_summary)
    print(f"[train] Finished. Artifacts saved to {run_dir}")
    return run_dir
