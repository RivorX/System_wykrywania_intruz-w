from __future__ import annotations

import argparse

from utils.dataset_utils import prepare_dataset_from_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare dataset for YOLO training.")
    parser.add_argument(
        "--config",
        default="config/dataset.yaml",
        help="Path to dataset config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml = prepare_dataset_from_file(args.config)
    print(f"[dataset] Prepared dataset YAML: {dataset_yaml}")


if __name__ == "__main__":
    main()

