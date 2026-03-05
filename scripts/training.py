from __future__ import annotations

import argparse

from utils.training_pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for intrusion detection.")
    parser.add_argument(
        "--config",
        default="config/train.yaml",
        help="Path to training config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_training(args.config)
    print(f"[train] Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()

