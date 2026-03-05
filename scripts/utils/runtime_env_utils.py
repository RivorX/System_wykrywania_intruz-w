from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _is_compile_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized not in {"", "0", "false", "none", "off", "no"}


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _default_cache_root() -> Path:
    cwd_drive = Path.cwd().anchor
    if not cwd_drive:
        cwd_drive = "C:\\"
    return Path(cwd_drive) / "torch_cache"


def ensure_windows_compile_env(config: dict[str, Any], compile_value: Any) -> None:
    if os.name != "nt":
        return
    if not _is_compile_enabled(compile_value):
        return

    enabled = bool(config.get("fix_unicode_cache_paths", True))
    if not enabled:
        return

    configured_root = str(config.get("cache_root_dir", "")).strip()
    base_dir = Path(configured_root) if configured_root else _default_cache_root()
    if not _is_ascii_path(base_dir):
        base_dir = _default_cache_root()

    triton_dir = (base_dir / "triton").resolve()
    inductor_dir = (base_dir / "inductor").resolve()
    temp_dir = (base_dir / "tmp").resolve()

    triton_dir.mkdir(parents=True, exist_ok=True)
    inductor_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)

    print(
        "[env] compile cache paths set: "
        f"TRITON_CACHE_DIR={triton_dir}, TORCHINDUCTOR_CACHE_DIR={inductor_dir}, TEMP={temp_dir}"
    )

