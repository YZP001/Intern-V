# Copyright (c) 2026.
#
# NOTE:
# This file lives under `Being-H_xlerobot/configs/` to override `configs.data_config`
# used by Being-H, without editing the upstream repository.

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from BeingH.dataset.transform.base import ComposedModalityTransform, ModalityTransform
from BeingH.dataset.transform.state_action import StateActionToTensor, StateActionTransform
from BeingH.utils.schema import RotationType


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices sampled relative to the current index."""

    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class ModalityDef(BaseModel):
    source_column: str = Field(..., description="Original column name in the Parquet file")
    start: int = Field(..., description="Start dimension index in the column")
    end: int = Field(..., description="End dimension index in the column (exclusive)")
    absolute: bool = True

    rotation_type: Optional[RotationType] = Field(None, description="Rotation representation type, if applicable")
    continuous: bool = Field(True, description="Whether the data is continuous (floating point)")


class BaseDataConfig(ABC):
    def __init__(
        self,
        embodiment_tag,
        use_fixed_view: bool,
        max_view_num: int,
        obs_indices: list[int] | None = None,
        action_indices: list[int] | None = None,
    ):
        self.embodiment_tag = embodiment_tag
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.obs_indices = obs_indices if obs_indices is not None else [0]
        self.action_indices = action_indices if action_indices is not None else list(range(16))

    def modality_config(self) -> Dict[str, dict]:
        # Return plain dicts so it can be sent over the inference service safely.
        return {
            "video": {"delta_indices": list(self.obs_indices), "modality_keys": list(self.VIDEO_KEYS)},
            "state": {"delta_indices": list(self.obs_indices), "modality_keys": list(self.STATE_KEYS)},
            "action": {"delta_indices": list(self.action_indices), "modality_keys": list(self.ACTION_KEYS)},
            "language": {"delta_indices": [0], "modality_keys": list(self.LANGUAGE_KEYS)},
        }

    @abstractmethod
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Map final modality keys -> how to slice raw dataset columns."""

    def get_sampling_indices(self) -> Dict[str, List[int]]:
        sampling_map: Dict[str, List[int]] = {}
        for key in self.VIDEO_KEYS + self.STATE_KEYS:
            sampling_map[key] = self.obs_indices
        for key in self.ACTION_KEYS:
            sampling_map[key] = self.action_indices
        return sampling_map

    @abstractmethod
    def get_transforms(self) -> ModalityTransform:
        """Return a transformation pipeline (normalization, tensor conversion, ...)."""

    def add_video_modality(self, modalities: Dict[str, ModalityDef]) -> Dict[str, ModalityDef]:
        if self.use_fixed_view:
            video_keys = [next(iter(self.VIDEO_SOURCE_COLUMNS))]
        elif self.max_view_num == -1:
            video_keys = list(self.VIDEO_SOURCE_COLUMNS.keys())
        else:
            max_view_num = min(self.max_view_num, len(self.VIDEO_SOURCE_COLUMNS))
            video_keys = random.sample(list(self.VIDEO_SOURCE_COLUMNS.keys()), max_view_num)

        for video_key in video_keys:
            modalities[video_key] = ModalityDef(source_column=self.VIDEO_SOURCE_COLUMNS[video_key], start=0, end=0)
        return modalities


# ==============================================================================
# XLerobot (real) dataset config
# ==============================================================================


class XLeRobotBox2BasketLeftHeadDataConfig(BaseDataConfig):
    """
    XLerobot dataset recorded with:
    - head camera (OpenCV id 0)
    - left wrist camera (OpenCV id 1)
    - left arm joint pos only (6-DoF + gripper)

    Dataset columns (LeRobot v3.0):
    - observation.state: (6,)
    - action: (6,)
    - observation.images.head: video
    - observation.images.left_wrist: video
    """

    VIDEO_KEYS = ["video.head", "video.left_wrist"]
    VIDEO_SOURCE_COLUMNS = {
        "video.head": "observation.images.head",
        "video.left_wrist": "observation.images.left_wrist",
    }
    STATE_KEYS = ["state.left_arm"]
    ACTION_KEYS = ["action.left_arm"]
    LANGUAGE_KEYS = ["language.instruction"]

    # Unified 200-dim state/action space: we fill the first 6 dims.
    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        "state.left_arm": (0, 6),
        "action.left_arm": (0, 6),
    }

    state_normalization_modes = {"state.left_arm": "min_max"}
    action_normalization_modes = {"action.left_arm": "min_max"}

    def get_feature_meta(self):
        return {
            "state.left_arm": ("6-d left arm joint position (deg)", 6),
            "action.left_arm": ("6-d left arm joint position (deg)", 6),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities: Dict[str, ModalityDef] = {
            # task_index -> task string mapping is handled by dataset (tasks.parquet)
            "language.instruction": ModalityDef(source_column="task_index", start=0, end=0),
            "state.left_arm": ModalityDef(source_column="observation.state", start=0, end=6, absolute=True),
            "action.left_arm": ModalityDef(source_column="action", start=0, end=6, absolute=True),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(apply_to=self.STATE_KEYS, normalization_modes=self.state_normalization_modes),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(apply_to=self.ACTION_KEYS, normalization_modes=self.action_normalization_modes),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {
    "xlerobot_box2basket_left_head": XLeRobotBox2BasketLeftHeadDataConfig,
}

