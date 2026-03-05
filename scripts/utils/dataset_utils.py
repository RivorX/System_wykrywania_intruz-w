from __future__ import annotations

import json
import random
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config_utils import ensure_dir, load_yaml, resolve_path, save_yaml
from .model_utils import download_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(images_root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return [
        path
        for path in images_root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _parse_limit_config(prepare_cfg: dict[str, Any]) -> tuple[bool, int | None]:
    limit_mode = str(prepare_cfg.get("limit_mode", "cap")).strip().lower()
    raw_max = prepare_cfg.get("max_images", None)

    if limit_mode in {"max", "all", "full", "unlimited"}:
        return False, None

    if raw_max in (None, "", "max", "all", "full", -1, 0):
        return False, None

    max_images = int(raw_max)
    if max_images <= 0:
        return False, None
    return True, max_images


def _apply_max_images_limit(
    pairs: list[tuple[Path, Path | None, Path]],
    prepare_cfg: dict[str, Any],
) -> list[tuple[Path, Path | None, Path]]:
    use_limit, max_images = _parse_limit_config(prepare_cfg)
    if not use_limit or max_images is None or len(pairs) <= max_images:
        return pairs

    sampling_mode = str(prepare_cfg.get("sampling_mode", "random")).strip().lower()
    seed = int(prepare_cfg.get("seed", 42))

    if sampling_mode == "head":
        limited = pairs[:max_images]
    else:
        shuffled = list(pairs)
        random.Random(seed).shuffle(shuffled)
        limited = shuffled[:max_images]

    print(f"[dataset] Image limit applied: {len(limited)}/{len(pairs)}")
    return limited


def _progress_flags(download_cfg: dict[str, Any]) -> tuple[bool, bool]:
    show_file_progress = bool(download_cfg.get("show_file_progress", download_cfg.get("show_progress", False)))
    show_overall_progress = bool(download_cfg.get("show_overall_progress", True))
    return show_file_progress, show_overall_progress


def _resolve_archive_source_root(download_cfg: dict[str, Any], base_dir: Path | None = None) -> Path:
    local_source_dir = download_cfg.get("local_source_dir")
    if local_source_dir:
        local_path = resolve_path(local_source_dir, base_dir=base_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Configured local_source_dir does not exist: {local_path}")
        return local_path

    source_url = download_cfg.get("source_url")
    archive_name = str(download_cfg.get("archive_name", "dataset.zip"))
    download_dir = ensure_dir(download_cfg.get("download_dir", "data/raw"), base_dir=base_dir)
    archive_path = download_dir / archive_name
    extract_dir = resolve_path(download_cfg.get("extract_dir", "data/raw/dataset"), base_dir=base_dir)
    force_extract = bool(download_cfg.get("force_extract", False))
    auto_download = bool(download_cfg.get("auto_download", True))
    show_file_progress, _ = _progress_flags(download_cfg)

    if auto_download and not archive_path.exists():
        if not source_url:
            raise ValueError("Dataset auto_download is enabled, but source_url is missing.")
        print("[dataset] Downloading dataset archive...")
        download_file(str(source_url), archive_path, show_progress=show_file_progress)

    if archive_path.exists() and (force_extract or not extract_dir.exists()):
        print("[dataset] Extracting dataset archive...")
        extract_dir.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(download_dir)

    if not extract_dir.exists():
        raise FileNotFoundError(
            f"Dataset source folder not found: {extract_dir}. "
            "Set download.extract_dir or download.local_source_dir."
        )

    return extract_dir


def _collect_pairs(
    images_root: Path,
    labels_root: Path,
    recursive: bool,
) -> list[tuple[Path, Path | None, Path]]:
    pairs: list[tuple[Path, Path | None, Path]] = []
    for image_path in _iter_images(images_root, recursive=recursive):
        relative_path = image_path.relative_to(images_root)
        label_path = labels_root / relative_path.with_suffix(".txt")
        pairs.append((image_path, label_path if label_path.exists() else None, relative_path))
    return pairs


def _label_has_target_class(label_path: Path | None, target_classes: set[int] | None) -> bool:
    if label_path is None or not label_path.exists():
        return False

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if target_classes is None or class_id in target_classes:
            return True
    return False


def _split_pairs(
    pairs: list[tuple[Path, Path | None, Path]],
    split_cfg: dict[str, Any],
    seed: int,
) -> dict[str, list[tuple[Path, Path | None, Path]]]:
    if not pairs:
        raise ValueError("Cannot split an empty dataset.")

    train_ratio = float(split_cfg.get("train", 0.7))
    val_ratio = float(split_cfg.get("val", 0.2))
    test_ratio = float(split_cfg.get("test", 0.1))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Invalid split ratios. Sum must be > 0.")

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if train_end == 0 and total > 0:
        train_end = 1
    if val_end <= train_end and total - train_end > 1 and val_ratio > 0:
        val_end = train_end + 1
    if val_end >= total and total > 1:
        val_end = total - 1

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _write_filtered_label(
    source_label_path: Path | None,
    destination_label_path: Path,
    class_mapping: dict[int, int],
) -> None:
    destination_label_path.parent.mkdir(parents=True, exist_ok=True)
    if source_label_path is None or not source_label_path.exists():
        destination_label_path.write_text("", encoding="utf-8")
        return

    output_lines: list[str] = []
    for line in source_label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))

        if class_mapping:
            if class_id not in class_mapping:
                continue
            parts[0] = str(class_mapping[class_id])

        output_lines.append(" ".join(parts))

    payload = "\n".join(output_lines)
    if payload:
        payload += "\n"
    destination_label_path.write_text(payload, encoding="utf-8")


def _copy_split(
    split_name: str,
    items: list[tuple[Path, Path | None, Path]],
    output_dir: Path,
    class_mapping: dict[int, int],
) -> None:
    split_images_dir = output_dir / "images" / split_name
    split_labels_dir = output_dir / "labels" / split_name
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path, relative_path in items:
        dest_image = split_images_dir / relative_path
        dest_label = split_labels_dir / relative_path.with_suffix(".txt")
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_image)
        _write_filtered_label(label_path, dest_label, class_mapping=class_mapping)


def _write_person_label_from_coco_bboxes(
    label_path: Path,
    bboxes: list[list[float]],
    width: int,
    height: int,
) -> None:
    lines: list[str] = []
    image_w = max(float(width), 1.0)
    image_h = max(float(height), 1.0)

    for bbox in bboxes:
        if len(bbox) < 4:
            continue
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if w <= 1.0 or h <= 1.0:
            continue

        x_center = (x + w / 2.0) / image_w
        y_center = (y + h / 2.0) / image_h
        norm_w = w / image_w
        norm_h = h / image_h

        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        norm_w = min(max(norm_w, 0.0), 1.0)
        norm_h = min(max(norm_h, 0.0), 1.0)

        lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(payload, encoding="utf-8")


def _download_coco_record(
    record: dict[str, Any],
    images_root: Path,
    labels_root: Path,
    images_base_url: str,
    show_file_progress: bool,
    retry_attempts: int,
) -> bool:
    split_name = record["split"]
    file_name = record["file_name"]
    relative_image_path = Path(split_name) / file_name
    image_path = images_root / relative_image_path
    label_path = labels_root / relative_image_path.with_suffix(".txt")
    image_url = record["coco_url"] or f"{images_base_url}/{split_name}/{file_name}"

    for attempt in range(retry_attempts + 1):
        try:
            if not image_path.exists():
                download_file(image_url, image_path, show_progress=show_file_progress)
            _write_person_label_from_coco_bboxes(
                label_path=label_path,
                bboxes=record["bboxes"],
                width=record["width"],
                height=record["height"],
            )
            return True
        except Exception:  # noqa: BLE001
            if attempt >= retry_attempts:
                return False
    return False


def _load_coco_person_records(annotation_file: Path, split_name: str, target_categories: set[int]) -> list[dict[str, Any]]:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in payload.get("images", [])}

    bboxes_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        category_id = int(ann.get("category_id", -1))
        if category_id not in target_categories:
            continue
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        image_id = int(ann.get("image_id", -1))
        bbox = ann.get("bbox")
        if image_id < 0 or not isinstance(bbox, list):
            continue
        bboxes_by_image[image_id].append([float(value) for value in bbox[:4]])

    records: list[dict[str, Any]] = []
    for image_id, bboxes in bboxes_by_image.items():
        image = images_by_id.get(image_id)
        if not image:
            continue
        records.append(
            {
                "split": split_name,
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "coco_url": str(image.get("coco_url", "")),
                "bboxes": bboxes,
            }
        )
    return records


def _resolve_coco_person_source_root(
    download_cfg: dict[str, Any],
    prepare_cfg: dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    auto_download = bool(download_cfg.get("auto_download", True))
    show_file_progress, show_overall_progress = _progress_flags(download_cfg)
    async_download = bool(download_cfg.get("async_download", True))
    max_download_workers = max(1, int(download_cfg.get("max_download_workers", 8)))
    retry_attempts = max(0, int(download_cfg.get("retry_attempts", 2)))
    force_extract = bool(download_cfg.get("force_extract", False))
    seed = int(prepare_cfg.get("seed", 42))

    download_dir = ensure_dir(download_cfg.get("download_dir", "data/raw/coco_subset"), base_dir=base_dir)
    source_root = resolve_path(download_cfg.get("extract_dir", "data/raw/coco_subset/source"), base_dir=base_dir)
    annotations_archive_name = str(download_cfg.get("annotations_archive_name", "annotations_trainval2017.zip"))
    annotations_archive_path = download_dir / annotations_archive_name
    annotations_extract_dir = resolve_path(
        download_cfg.get("annotations_extract_dir", "data/raw/coco_subset/annotations_trainval2017"),
        base_dir=base_dir,
    )
    annotation_url = str(
        download_cfg.get(
            "annotations_url",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        )
    )
    images_base_url = str(download_cfg.get("images_base_url", "http://images.cocodataset.org")).rstrip("/")
    include_splits = [str(item) for item in download_cfg.get("include_splits", ["train2017", "val2017"])]
    target_categories = {int(item) for item in download_cfg.get("target_category_ids", [1])}
    use_limit, max_images = _parse_limit_config(prepare_cfg)
    sampling_mode = str(prepare_cfg.get("sampling_mode", "random")).strip().lower()

    expected_meta = {
        "include_splits": include_splits,
        "target_category_ids": sorted(target_categories),
        "use_limit": use_limit,
        "max_images": max_images,
        "sampling_mode": sampling_mode,
        "seed": seed,
    }
    meta_path = source_root / "_subset_meta.json"

    if source_root.exists() and force_extract:
        shutil.rmtree(source_root, ignore_errors=True)

    images_root = source_root / "images"
    labels_root = source_root / "labels"
    if images_root.exists() and labels_root.exists() and any(images_root.rglob("*")) and not force_extract:
        if meta_path.exists():
            try:
                current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if current_meta == expected_meta:
                    print(f"[dataset] Reusing prepared COCO source: {source_root}")
                    return source_root
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(source_root, ignore_errors=True)

    if not auto_download and not annotations_archive_path.exists():
        raise FileNotFoundError(
            f"COCO annotations archive missing and auto_download is disabled: {annotations_archive_path}"
        )

    if auto_download and not annotations_archive_path.exists():
        print("[dataset] Downloading COCO annotations archive...")
        download_file(annotation_url, annotations_archive_path, show_progress=show_file_progress)

    annotation_root = annotations_extract_dir / "annotations"
    if force_extract and annotations_extract_dir.exists():
        shutil.rmtree(annotations_extract_dir, ignore_errors=True)
    if not annotation_root.exists():
        print("[dataset] Extracting COCO annotations...")
        annotations_extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(annotations_archive_path, "r") as archive:
            archive.extractall(annotations_extract_dir)

    records: list[dict[str, Any]] = []
    for split_name in include_splits:
        ann_file = annotation_root / f"instances_{split_name}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
        records.extend(_load_coco_person_records(ann_file, split_name=split_name, target_categories=target_categories))

    if not records:
        raise RuntimeError("No COCO records found for selected categories and splits.")

    if use_limit and max_images is not None and len(records) > max_images:
        if sampling_mode == "head":
            records = records[:max_images]
        else:
            random.Random(seed).shuffle(records)
            records = records[:max_images]

    if source_root.exists():
        shutil.rmtree(source_root, ignore_errors=True)
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    print(f"[dataset] Preparing COCO person subset ({len(records)} images)...")
    if async_download and show_file_progress:
        print("[dataset] show_file_progress=true with async_download can clutter output. Using false for per-file bars.")
        show_file_progress = False

    downloaded = 0
    failed = 0
    progress_bar = tqdm(
        total=len(records),
        desc="[dataset] Downloading COCO images",
        unit="img",
        disable=not show_overall_progress,
    )
    if async_download and max_download_workers > 1:
        print(f"[dataset] Async download enabled (workers={max_download_workers}, retries={retry_attempts}).")
        with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
            futures = [
                executor.submit(
                    _download_coco_record,
                    record,
                    images_root,
                    labels_root,
                    images_base_url,
                    show_file_progress,
                    retry_attempts,
                )
                for record in records
            ]

            for index, future in enumerate(as_completed(futures), start=1):
                success = future.result()
                if success:
                    downloaded += 1
                else:
                    failed += 1

                progress_bar.update(1)
                if show_overall_progress and (index == 1 or index % 25 == 0 or index == len(records)):
                    progress_bar.set_postfix({"ok": downloaded, "skip": failed})
    else:
        for index, record in enumerate(records, start=1):
            success = _download_coco_record(
                record=record,
                images_root=images_root,
                labels_root=labels_root,
                images_base_url=images_base_url,
                show_file_progress=show_file_progress,
                retry_attempts=retry_attempts,
            )
            if success:
                downloaded += 1
            else:
                failed += 1

            progress_bar.update(1)
            if show_overall_progress and (index == 1 or index % 25 == 0 or index == len(records)):
                progress_bar.set_postfix({"ok": downloaded, "skip": failed})

    progress_bar.close()

    if downloaded == 0:
        raise RuntimeError("Failed to download any COCO images for subset preparation.")

    if failed > 0:
        print(f"[dataset] COCO subset prepared with {downloaded} images ({failed} skipped).")
    else:
        print(f"[dataset] COCO subset prepared with {downloaded} images.")
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return source_root


def _resolve_source_root(
    download_cfg: dict[str, Any],
    prepare_cfg: dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    provider = str(download_cfg.get("provider", "archive")).strip().lower()
    if provider == "coco_person_subset":
        return _resolve_coco_person_source_root(download_cfg, prepare_cfg, base_dir=base_dir)
    return _resolve_archive_source_root(download_cfg, base_dir=base_dir)


def prepare_dataset_from_dict(dataset_cfg: dict[str, Any], base_dir: Path | None = None) -> Path:
    dataset_meta = dataset_cfg.get("dataset", {})
    download_cfg = dataset_cfg.get("download", {})
    source_layout = dataset_cfg.get("source_layout", {})
    prepare_cfg = dataset_cfg.get("prepare", {})

    output_dir = ensure_dir(dataset_meta.get("output_dir", "data/processed/intrusion_people"), base_dir=base_dir)
    dataset_yaml_name = str(dataset_meta.get("dataset_yaml_name", "dataset.yaml"))
    dataset_yaml_path = output_dir / dataset_yaml_name
    prepare_meta_path = output_dir / "_prepare_meta.json"
    force_prepare = bool(prepare_cfg.get("force_prepare", False))
    expected_prepare_meta = {
        "download": download_cfg,
        "source_layout": source_layout,
        "prepare": prepare_cfg,
    }

    if dataset_yaml_path.exists() and not force_prepare:
        if prepare_meta_path.exists():
            try:
                current_meta = json.loads(prepare_meta_path.read_text(encoding="utf-8"))
                if current_meta == expected_prepare_meta:
                    print(f"[dataset] Reusing prepared dataset: {dataset_yaml_path}")
                    return dataset_yaml_path
            except Exception:  # noqa: BLE001
                pass
        print("[dataset] Config changed - rebuilding prepared dataset.")

    source_root = _resolve_source_root(download_cfg, prepare_cfg, base_dir=base_dir)
    images_root = source_root / str(source_layout.get("images_dir", "images"))
    labels_root = source_root / str(source_layout.get("labels_dir", "labels"))
    recursive = bool(source_layout.get("recurse_images", True))

    if not images_root.exists():
        raise FileNotFoundError(f"Images folder not found: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_root}")

    pairs = _collect_pairs(images_root, labels_root, recursive=recursive)
    if not pairs:
        raise RuntimeError(f"No image files found in {images_root}")

    filter_classes = [int(item) for item in prepare_cfg.get("filter_classes", [])]
    class_names = [str(item) for item in prepare_cfg.get("class_names", [])]
    class_mapping = {old_class: new_idx for new_idx, old_class in enumerate(filter_classes)}
    keep_only_annotated = bool(prepare_cfg.get("keep_only_annotated", bool(filter_classes)))

    if keep_only_annotated:
        target_classes = set(filter_classes) if filter_classes else None
        filtered_pairs = [pair for pair in pairs if _label_has_target_class(pair[1], target_classes)]
        if not filtered_pairs:
            if filter_classes:
                raise RuntimeError(
                    "No images with requested classes were found. "
                    f"Requested classes: {filter_classes}"
                )
            raise RuntimeError("No annotated images were found in the dataset.")
        print(f"[dataset] Filtered images: {len(filtered_pairs)}/{len(pairs)} kept.")
        pairs = filtered_pairs

    pairs = _apply_max_images_limit(pairs, prepare_cfg=prepare_cfg)
    split_cfg = prepare_cfg.get("split", {})
    split_seed = int(prepare_cfg.get("seed", 42))
    split_map = _split_pairs(pairs, split_cfg=split_cfg, seed=split_seed)

    if class_mapping and not class_names:
        class_names = [f"class_{idx}" for idx in range(len(class_mapping))]
    if not class_names:
        class_names = ["person"]

    if force_prepare or dataset_yaml_path.exists():
        for folder_name in ("images", "labels"):
            folder_path = output_dir / folder_name
            if folder_path.exists():
                shutil.rmtree(folder_path)

    for split_name, split_items in split_map.items():
        _copy_split(split_name, split_items, output_dir=output_dir, class_mapping=class_mapping)

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    save_yaml(dataset_yaml_path, dataset_yaml)
    prepare_meta_path.write_text(json.dumps(expected_prepare_meta, indent=2), encoding="utf-8")

    print(
        "[dataset] Prepared dataset "
        f"(train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}) "
        f"at {dataset_yaml_path}"
    )
    return dataset_yaml_path


def prepare_dataset_from_file(dataset_config_path: str | Path, base_dir: Path | None = None) -> Path:
    dataset_cfg = load_yaml(dataset_config_path, base_dir=base_dir)
    return prepare_dataset_from_dict(dataset_cfg, base_dir=base_dir)
