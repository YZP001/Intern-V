"""Dataset registry/info override for Being-H training.

Being-H training code imports `configs.dataset_info` (namespace package).
We provide an XLerobot entry here without modifying upstream code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dataset.lerobot_v3_iterable_dataset import LeRobotV3IterableDataset


DATASET_REGISTRY = {
    # XLerobot (LeRobot v3.0 on-disk format)
    "xlerobot_posttrain": LeRobotV3IterableDataset,
}


def _default_datasets_root() -> Path:
    # Repo layout: <VLA_ROOT>/Being-H_xlerobot/configs/dataset_info.py
    vla_root = Path(__file__).resolve().parents[2]
    return vla_root / "datasets"

def _resolve_dataset_path(dataset_name: str) -> Path:
    """
    Resolve dataset path from env var.

    Supports two common setups:
    1) VLA_DATASETS_DIR points to the parent directory that contains datasets:
       e.g. /workspace/datasets
       -> /workspace/datasets/<dataset_name>
    2) VLA_DATASETS_DIR points directly to the dataset directory:
       e.g. /workspace/datasets/<dataset_name>
       -> use it as-is
    """
    base = Path(os.getenv("VLA_DATASETS_DIR", str(_default_datasets_root()))).expanduser()

    # If user already points to the dataset dir, accept it.
    if base.name == dataset_name and (base / "meta" / "info.json").exists():
        return base

    return base / dataset_name


DATASET_INFO = {
    "xlerobot_posttrain": {
        # Put the box into the robot basket (left arm + head cam).
        "local_xlerobot_box2basket_left_head": {
            "dataset_path": str(_resolve_dataset_path("local_xlerobot_box2basket_left_head")),
            "embodiment": "XLEROBOT",
            # Being-H only knows "new_embodiment" in its EmbodimentTag enum.
            "embodiment_tag": "new_embodiment",
            "subtask": "xlerobot.box2basket_left_head",
        }
    }
}
