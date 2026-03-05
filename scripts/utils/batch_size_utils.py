from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from .config_utils import ensure_dir, resolve_path, utc_timestamp

AUTO_BATCH_ALGO_VERSION = 1


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    keywords = (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cuda out of memory",
    )
    return any(keyword in message for keyword in keywords)


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()


def _is_compile_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized not in {"", "0", "false", "none", "off", "no"}


def _cache_enabled_compiled(auto_batch_cfg: dict[str, Any], compile_value: Any) -> bool:
    if not bool(auto_batch_cfg.get("cache_enabled", True)):
        return False
    # Enforce cache usage only for compiled runtime profile.
    return _is_compile_enabled(compile_value)


def _resolve_cache_path(auto_batch_cfg: dict[str, Any]) -> Path:
    cache_path_value = auto_batch_cfg.get("cache_path", "logs/train/auto_batch_compiled_cache.json")
    cache_path = resolve_path(cache_path_value)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def _load_cache(cache_path: Path) -> dict[str, Any]:
    default_cache = {"version": 1, "entries": {}}
    if not cache_path.exists():
        return default_cache
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default_cache
    if not isinstance(payload, dict):
        return default_cache
    payload.setdefault("version", 1)
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    return payload


def _save_cache(cache_path: Path, cache_payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")


def _device_fingerprint(device: Any) -> dict[str, Any]:
    torch_device = _resolve_torch_device(device)
    fp = {"requested_device": str(device), "resolved_device": str(torch_device)}
    if torch_device.type == "cuda" and torch.cuda.is_available():
        index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        fp.update(
            {
                "name": str(props.name),
                "index": int(index),
                "total_memory": int(getattr(props, "total_memory", 0)),
                "cc": f"{int(props.major)}.{int(props.minor)}",
            }
        )
    return fp


def _model_fingerprint(model_reference: str) -> dict[str, Any]:
    ref_path = Path(model_reference)
    if ref_path.exists() and ref_path.is_file():
        stats = ref_path.stat()
        return {
            "reference": str(ref_path.resolve()),
            "name": ref_path.name,
            "size": int(stats.st_size),
            "mtime_ns": int(stats.st_mtime_ns),
        }
    return {"reference": str(model_reference)}


def _batch_cache_key(
    *,
    model_reference: str,
    imgsz: int,
    device: Any,
    classes: list[int] | None,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
    min_batch: int,
    max_batch: int,
    multiple_of: int,
) -> tuple[str, dict[str, Any]]:
    classes_key = sorted(int(item) for item in classes) if classes is not None else None
    key_payload = {
        "algo_version": AUTO_BATCH_ALGO_VERSION,
        "model": _model_fingerprint(model_reference),
        "imgsz": int(imgsz),
        "device": _device_fingerprint(device),
        "classes": classes_key,
        "compile_enabled": _is_compile_enabled(compile_value),
        "compile_mode": str(auto_batch_cfg.get("compile_mode", "default")),
        "compile_dynamic": bool(auto_batch_cfg.get("compile_dynamic", True)),
        "probe_method": str(auto_batch_cfg.get("probe_method", "synthetic")).strip().lower(),
        "probe_amp": bool(auto_batch_cfg.get("probe_amp", True)),
        "max_vram_utilization": float(auto_batch_cfg.get("max_vram_utilization", 0.92)),
        "max_vram_metric": str(auto_batch_cfg.get("max_vram_metric", "allocated")).strip().lower(),
        "synthetic_warmup_steps": int(auto_batch_cfg.get("synthetic_warmup_steps", 1)),
        "synthetic_measure_steps": int(auto_batch_cfg.get("synthetic_measure_steps", 1)),
        "synthetic_reduce_tensors": int(auto_batch_cfg.get("synthetic_reduce_tensors", 8)),
        "min_batch": int(min_batch),
        "max_batch": int(max_batch),
        "multiple_of": int(multiple_of),
    }
    key_text = json.dumps(key_payload, sort_keys=True, ensure_ascii=True, default=str)
    cache_key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    return cache_key, key_payload


def _read_cached_batch(
    *,
    cache_payload: dict[str, Any],
    cache_key: str,
    min_batch: int,
    max_batch: int,
    auto_batch_cfg: dict[str, Any],
) -> int | None:
    entries = cache_payload.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(cache_key)
    if not isinstance(entry, dict):
        return None

    max_age_hours = float(auto_batch_cfg.get("cache_max_age_hours", 168))
    if max_age_hours > 0:
        created_at = entry.get("updated_at_utc")
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600.0
                if age_hours > max_age_hours:
                    return None
            except Exception:  # noqa: BLE001
                return None

    batch_value = int(entry.get("batch", 0) or 0)
    if batch_value < min_batch or batch_value > max_batch:
        return None
    return batch_value


def _write_cached_batch(
    *,
    cache_path: Path,
    cache_payload: dict[str, Any],
    cache_key: str,
    cache_key_payload: dict[str, Any],
    selected_batch: int,
    best_fit: int,
    attempts: int,
    auto_batch_cfg: dict[str, Any],
) -> None:
    entries = cache_payload.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        cache_payload["entries"] = entries

    entries[cache_key] = {
        "batch": int(selected_batch),
        "best_fit": int(best_fit),
        "attempts": int(attempts),
        "algo_version": AUTO_BATCH_ALGO_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "key_payload": cache_key_payload,
    }

    max_entries = max(10, int(auto_batch_cfg.get("cache_max_entries", 200)))
    if len(entries) > max_entries:
        sortable: list[tuple[str, float]] = []
        for key, item in entries.items():
            ts_value = 0.0
            if isinstance(item, dict):
                ts_raw = item.get("updated_at_utc")
                if isinstance(ts_raw, str):
                    try:
                        ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                        ts_value = ts_dt.timestamp()
                    except Exception:  # noqa: BLE001
                        ts_value = 0.0
            sortable.append((key, ts_value))
        sortable.sort(key=lambda pair: pair[1], reverse=True)
        keep_keys = {key for key, _ in sortable[:max_entries]}
        cache_payload["entries"] = {key: value for key, value in entries.items() if key in keep_keys}

    _save_cache(cache_path, cache_payload)


def _resolve_torch_device(device: Any) -> torch.device:
    if isinstance(device, int):
        if device >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")

    if isinstance(device, str):
        normalized = device.strip().lower()
        if normalized in {"", "auto"}:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if normalized.isdigit() and torch.cuda.is_available():
            return torch.device(f"cuda:{normalized}")
        try:
            return torch.device(device)
        except Exception:  # noqa: BLE001
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_tensors(value)


def _create_synthetic_probe_state(
    *,
    model_reference: str,
    device: Any,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
) -> dict[str, Any]:
    _clear_cuda_cache()
    torch_device = _resolve_torch_device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("Synthetic probe requires CUDA device.")

    respect_compile = bool(auto_batch_cfg.get("respect_compile", True))
    compile_mode = str(auto_batch_cfg.get("compile_mode", "default"))
    compile_dynamic = bool(auto_batch_cfg.get("compile_dynamic", True))
    probe_amp = bool(auto_batch_cfg.get("probe_amp", True))
    warmup_steps = max(0, int(auto_batch_cfg.get("synthetic_warmup_steps", 1)))
    measure_steps = max(1, int(auto_batch_cfg.get("synthetic_measure_steps", 1)))
    reduce_tensors = max(1, int(auto_batch_cfg.get("synthetic_reduce_tensors", 8)))
    max_vram_utilization = float(auto_batch_cfg.get("max_vram_utilization", 0.92))
    max_vram_utilization = min(max(max_vram_utilization, 0.5), 0.98)
    max_vram_metric = str(auto_batch_cfg.get("max_vram_metric", "allocated")).strip().lower()
    if max_vram_metric not in {"allocated", "reserved"}:
        max_vram_metric = "allocated"

    yolo_model = YOLO(model_reference)
    raw_model = yolo_model.model.to(torch_device)
    raw_model.train()
    for param in raw_model.parameters():
        param.requires_grad_(True)
    model = raw_model

    if respect_compile and _is_compile_enabled(compile_value):
        compile_error: BaseException | None = None
        if compile_dynamic:
            try:
                model = torch.compile(raw_model, mode=compile_mode, dynamic=True)
            except TypeError:
                model = torch.compile(raw_model, mode=compile_mode)
            except Exception as exc:  # noqa: BLE001
                compile_error = exc
                model = raw_model
                print(
                    "[batch-auto] torch.compile(dynamic=True) failed "
                    f"({exc.__class__.__name__}). Retrying without dynamic."
                )

        if model is raw_model:
            try:
                model = torch.compile(raw_model, mode=compile_mode)
            except Exception as exc:  # noqa: BLE001
                compile_error = exc
                if bool(auto_batch_cfg.get("fallback_disable_compile_on_error", True)):
                    model = raw_model
                    print(
                        "[batch-auto] torch.compile probe unavailable "
                        f"({exc.__class__.__name__}). Falling back to eager synthetic probe."
                    )
                else:
                    raise RuntimeError("torch.compile failed for synthetic auto-batch probe") from exc

        if model is raw_model and compile_error is not None:
            # Preserve a hint in state/debug path when compile was requested but unavailable.
            auto_batch_cfg["_compile_probe_error"] = f"{compile_error.__class__.__name__}: {compile_error}"

    autocast_enabled = probe_amp and torch_device.type == "cuda"
    total_vram_bytes = int(torch.cuda.get_device_properties(torch_device).total_memory)
    vram_hard_cap_bytes = int(float(total_vram_bytes) * max_vram_utilization)
    return {
        "torch_device": torch_device,
        "autocast_enabled": autocast_enabled,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "reduce_tensors": reduce_tensors,
        "total_vram_bytes": total_vram_bytes,
        "vram_hard_cap_bytes": vram_hard_cap_bytes,
        "max_vram_utilization": max_vram_utilization,
        "max_vram_metric": max_vram_metric,
        "warmup_done": False,
        "yolo_model": yolo_model,
        "raw_model": raw_model,
        "model": model,
    }


def _release_synthetic_probe_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    for key in ("model", "raw_model", "yolo_model"):
        value = state.get(key)
        if value is not None:
            del value
    state.clear()
    _clear_cuda_cache()


def _probe_batch_fit_mini_train(
    *,
    model_reference: str,
    dataset_reference: str,
    batch_size: int,
    imgsz: int,
    device: Any,
    classes: list[int] | None,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
) -> bool:
    keep_probe_artifacts = bool(auto_batch_cfg.get("keep_probe_artifacts", False))
    probe_fraction = float(auto_batch_cfg.get("probe_fraction", 0.02))
    probe_workers = int(auto_batch_cfg.get("probe_workers", 0))
    probe_project = ensure_dir(auto_batch_cfg.get("probe_logs_dir", "logs/train/_autobatch"))
    probe_name = f"probe_bs{batch_size}_{utc_timestamp()}"
    probe_run_dir = probe_project / probe_name
    respect_compile = bool(auto_batch_cfg.get("respect_compile", True))

    _clear_cuda_cache()
    probe_model = YOLO(model_reference)

    probe_args: dict[str, Any] = {
        "data": dataset_reference,
        "epochs": 1,
        "imgsz": imgsz,
        "batch": int(batch_size),
        "workers": probe_workers,
        "project": str(probe_project),
        "name": probe_name,
        "exist_ok": True,
        "save": False,
        "val": False,
        "plots": False,
        "cache": False,
        "verbose": False,
        "fraction": probe_fraction,
    }

    if classes is not None:
        probe_args["classes"] = classes
    if device is not None:
        probe_args["device"] = device
    if respect_compile and compile_value is not None:
        probe_args["compile"] = compile_value

    try:
        probe_model.train(**probe_args)
        return True
    except RuntimeError as exc:
        if _is_oom_error(exc):
            return False
        raise
    finally:
        _clear_cuda_cache()
        if not keep_probe_artifacts and probe_run_dir.exists():
            shutil.rmtree(probe_run_dir, ignore_errors=True)


def _probe_batch_fit_synthetic(
    *,
    model_reference: str,
    batch_size: int,
    imgsz: int,
    device: Any,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
    probe_state: dict[str, Any] | None = None,
) -> bool:
    own_state = probe_state is None
    state = probe_state
    if own_state:
        state = _create_synthetic_probe_state(
            model_reference=model_reference,
            device=device,
            compile_value=compile_value,
            auto_batch_cfg=auto_batch_cfg,
        )

    assert state is not None
    torch_device: torch.device = state["torch_device"]
    autocast_enabled = bool(state["autocast_enabled"])
    warmup_steps = int(state["warmup_steps"])
    measure_steps = int(state["measure_steps"])
    reduce_tensors = int(state["reduce_tensors"])
    vram_hard_cap_bytes = int(state.get("vram_hard_cap_bytes", 0) or 0)
    total_vram_bytes = int(state.get("total_vram_bytes", 0) or 0)
    max_vram_metric = str(state.get("max_vram_metric", "allocated"))
    model = state["model"]
    steps = measure_steps + (0 if bool(state.get("warmup_done", False)) else warmup_steps)

    try:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(torch_device)

        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            images = torch.randn(
                int(batch_size),
                3,
                int(imgsz),
                int(imgsz),
                device=torch_device,
                dtype=torch.float32,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                outputs = model(images)
                output_tensors = [
                    tensor
                    for tensor in _iter_tensors(outputs)
                    if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.requires_grad
                ]
                if not output_tensors:
                    raise RuntimeError(
                        "Synthetic probe failed: model outputs do not require grad."
                    )
                scalar_terms = [tensor.float().mean() for tensor in output_tensors[:reduce_tensors]]
                loss = torch.stack(scalar_terms).sum()
            loss.backward()
            torch.cuda.synchronize(torch_device)
            peak_bytes = int(torch.cuda.max_memory_allocated(torch_device))
            if max_vram_metric == "reserved":
                peak_bytes = int(torch.cuda.max_memory_reserved(torch_device))
            if vram_hard_cap_bytes > 0 and peak_bytes > vram_hard_cap_bytes:
                cap_gb = vram_hard_cap_bytes / (1024**3)
                peak_gb = peak_bytes / (1024**3)
                total_gb = total_vram_bytes / (1024**3) if total_vram_bytes > 0 else 0.0
                print(
                    "[batch-auto] VRAM cap exceeded "
                    f"(metric={max_vram_metric}, peak={peak_gb:.2f} GiB, "
                    f"cap={cap_gb:.2f} GiB, total={total_gb:.2f} GiB)"
                )
                return False

            del images
            del outputs
            del output_tensors
            del scalar_terms
            del loss
        state["warmup_done"] = True
        return True
    except RuntimeError as exc:
        if _is_oom_error(exc):
            return False
        raise
    finally:
        if own_state:
            _release_synthetic_probe_state(state)


def _probe_batch_fit(
    *,
    model_reference: str,
    dataset_reference: str,
    batch_size: int,
    imgsz: int,
    device: Any,
    classes: list[int] | None,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
    synthetic_probe_state: dict[str, Any] | None = None,
) -> bool:
    probe_method = str(auto_batch_cfg.get("probe_method", "synthetic")).strip().lower()
    fallback_to_mini_train = bool(auto_batch_cfg.get("fallback_to_mini_train", False))

    if probe_method in {"synthetic", "tensor", "fast"}:
        try:
            return _probe_batch_fit_synthetic(
                model_reference=model_reference,
                batch_size=batch_size,
                imgsz=imgsz,
                device=device,
                compile_value=compile_value,
                auto_batch_cfg=auto_batch_cfg,
                probe_state=synthetic_probe_state,
            )
        except RuntimeError as exc:
            if _is_oom_error(exc):
                return False
            if fallback_to_mini_train:
                print(
                    "[batch-auto] Synthetic probe failed "
                    f"({exc.__class__.__name__}). Falling back to mini-train probe."
                )
            else:
                raise

    return _probe_batch_fit_mini_train(
        model_reference=model_reference,
        dataset_reference=dataset_reference,
        batch_size=batch_size,
        imgsz=imgsz,
        device=device,
        classes=classes,
        compile_value=compile_value,
        auto_batch_cfg=auto_batch_cfg,
    )


def _round_down_to_multiple(value: int, multiple: int, minimum: int) -> int:
    if multiple <= 1:
        return max(minimum, value)
    rounded = (value // multiple) * multiple
    if rounded < minimum:
        return minimum
    return rounded


def resolve_smart_batch_size(
    *,
    model_reference: str,
    dataset_reference: str,
    imgsz: int,
    device: Any,
    classes: list[int] | None,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
) -> int:
    min_batch = max(1, int(auto_batch_cfg.get("min_batch", 2)))
    max_batch = max(min_batch, int(auto_batch_cfg.get("max_batch", 128)))
    start_batch = int(auto_batch_cfg.get("start_batch", 8))
    start_batch = max(min_batch, min(max_batch, start_batch))
    growth_factor = max(2, int(auto_batch_cfg.get("growth_factor", 2)))
    safety_factor = float(auto_batch_cfg.get("safety_factor", 0.9))
    safety_factor = min(max(safety_factor, 0.5), 1.0)
    multiple_of = max(1, int(auto_batch_cfg.get("multiple_of", 2)))
    max_probes = max(4, int(auto_batch_cfg.get("max_probes", 10)))
    use_cache = _cache_enabled_compiled(auto_batch_cfg, compile_value=compile_value)
    cache_path: Path | None = None
    cache_payload: dict[str, Any] | None = None
    cache_key: str | None = None
    cache_key_payload: dict[str, Any] | None = None

    if use_cache:
        cache_path = _resolve_cache_path(auto_batch_cfg)
        cache_payload = _load_cache(cache_path)
        cache_key, cache_key_payload = _batch_cache_key(
            model_reference=model_reference,
            imgsz=imgsz,
            device=device,
            classes=classes,
            compile_value=compile_value,
            auto_batch_cfg=auto_batch_cfg,
            min_batch=min_batch,
            max_batch=max_batch,
            multiple_of=multiple_of,
        )
        cached_batch = _read_cached_batch(
            cache_payload=cache_payload,
            cache_key=cache_key,
            min_batch=min_batch,
            max_batch=max_batch,
            auto_batch_cfg=auto_batch_cfg,
        )
        if cached_batch is not None:
            print(f"[batch-auto] Using cached batch={cached_batch} for compiled profile.")
            return cached_batch

    probe_method = str(auto_batch_cfg.get("probe_method", "synthetic")).strip().lower()
    fallback_to_mini_train = bool(auto_batch_cfg.get("fallback_to_mini_train", False))
    reuse_synthetic_state = bool(auto_batch_cfg.get("reuse_synthetic_state", True))
    synthetic_probe_state: dict[str, Any] | None = None
    if reuse_synthetic_state and probe_method in {"synthetic", "tensor", "fast"}:
        try:
            synthetic_probe_state = _create_synthetic_probe_state(
                model_reference=model_reference,
                device=device,
                compile_value=compile_value,
                auto_batch_cfg=auto_batch_cfg,
            )
            max_util = float(synthetic_probe_state.get("max_vram_utilization", 0.0))
            print(
                "[batch-auto] Synthetic probe session initialized "
                f"(reuse=true, max_vram_utilization={max_util:.2f})"
            )
        except RuntimeError as exc:
            if not fallback_to_mini_train:
                raise
            print(
                "[batch-auto] Synthetic probe init failed "
                f"({exc.__class__.__name__}). Falling back to mini-train probes."
            )
            synthetic_probe_state = None

    print(
        "[batch-auto] Smart batch search "
        f"(start={start_batch}, min={min_batch}, max={max_batch}, probes<={max_probes})"
    )

    attempts = 0
    lower_fit = 0
    upper_fail = max_batch + 1

    def probe(candidate: int) -> bool:
        nonlocal attempts
        attempts += 1
        print(f"[batch-auto] Probe {attempts}/{max_probes}: batch={candidate}")
        fits = _probe_batch_fit(
            model_reference=model_reference,
            dataset_reference=dataset_reference,
            batch_size=candidate,
            imgsz=imgsz,
            device=device,
            classes=classes,
            compile_value=compile_value,
            auto_batch_cfg=auto_batch_cfg,
            synthetic_probe_state=synthetic_probe_state,
        )
        print(f"[batch-auto] batch={candidate} -> {'fits' if fits else 'no-fit'}")
        return fits

    try:
        # Step 1: find a fitting lower bound.
        candidate = start_batch
        if probe(candidate):
            lower_fit = candidate
        else:
            upper_fail = candidate
            while attempts < max_probes and candidate > min_batch:
                candidate = max(min_batch, candidate // growth_factor)
                if probe(candidate):
                    lower_fit = candidate
                    break
                upper_fail = candidate

            if lower_fit == 0:
                print(f"[batch-auto] No fitting probe found above min. Using min_batch={min_batch}")
                return min_batch

        # Step 2: grow upper bound quickly.
        candidate = lower_fit
        while attempts < max_probes and candidate < max_batch and upper_fail == max_batch + 1:
            next_candidate = min(max_batch, candidate * growth_factor)
            if next_candidate == candidate:
                break
            if probe(next_candidate):
                lower_fit = next_candidate
                candidate = next_candidate
            else:
                upper_fail = next_candidate
                break

        if upper_fail == max_batch + 1:
            upper_fail = max_batch + 1

        # Step 3: binary search between known fit and known fail.
        while attempts < max_probes and upper_fail - lower_fit > 1:
            mid = (lower_fit + upper_fail) // 2
            if mid <= lower_fit:
                break
            if probe(mid):
                lower_fit = mid
            else:
                upper_fail = mid

        best_fit = max(lower_fit, min_batch)
        safe_batch = int(best_fit * safety_factor)
        safe_batch = _round_down_to_multiple(safe_batch, multiple_of, min_batch)
        safe_batch = min(safe_batch, best_fit)

        print(
            "[batch-auto] Selected "
            f"best_fit={best_fit}, safety_factor={safety_factor}, final_batch={safe_batch}"
        )

        if use_cache and cache_path is not None and cache_payload is not None and cache_key is not None and cache_key_payload is not None:
            _write_cached_batch(
                cache_path=cache_path,
                cache_payload=cache_payload,
                cache_key=cache_key,
                cache_key_payload=cache_key_payload,
                selected_batch=safe_batch,
                best_fit=best_fit,
                attempts=attempts,
                auto_batch_cfg=auto_batch_cfg,
            )
            print(f"[batch-auto] Cached compiled batch in {cache_path}")

        return safe_batch
    finally:
        _release_synthetic_probe_state(synthetic_probe_state)
