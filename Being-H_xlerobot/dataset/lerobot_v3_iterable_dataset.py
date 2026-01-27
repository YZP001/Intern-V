"""LeRobot v3.0 dataset adapter for Being-H.

This yields "packet" dicts compatible with `BeingH.dataset.base_dataset.PackedDataset`,
without modifying upstream Being-H code (which assumes an older LeRobot dataset layout).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from BeingH.dataset.preprocess import build_vit_transform_base
from BeingH.utils.constants import (
    EMBODIMENT_TAG_MAPPING,
    INSTRUCTION_TEMPLATE,
    MULTI_DB_INSTRUCT_TEMPLATE,
    EmbodimentTag,
)
from BeingH.utils.conversation import get_conv_template
from BeingH.utils.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatistics,
    StateActionMetadata,
    VideoMetadata,
)
from BeingH.utils.video_utils import get_frames_by_timestamps

from configs.data_config import DATA_CONFIG_MAP, BaseDataConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EpisodeVideoRef:
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


class LeRobotV3IterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset compatible with Being-H PackedDataset."""

    def __init__(
        self,
        dataset_name: str,
        data_config_names: List[str],
        dataset_path_list: List[str],
        embodiment_tags: List[str],
        vit_transform_args: dict,
        num_used_episodes_per_dataset: Optional[List[int]] = None,
        num_used_episodes_per_task: Optional[List[int]] = None,
        num_used_frames_per_dataset: Optional[List[int]] = None,
        frame_step_size: Optional[List[int]] = None,
        is_train: bool = True,
        video_backend: str = "torchvision_av",
        video_backend_kwargs: Optional[dict] = None,
        logger: Any = None,
        # Tokenizer / prompt
        tokenizer: Any = None,
        template_name: Optional[str] = None,
        prompt_template: str = "long",
        # Visual
        force_image_size: int = 448,
        num_image_tokens: int = 0,
        max_view_num: int = -1,
        use_fixed_view: bool = False,
        # Action/state behavior
        is_relative: bool = False,
        is_abstract_action: bool = False,
        vit_dropout_prob: float = 0.0,
        state_dropout_prob: float = 0.0,
        sampling_strategy: str = "step",
        gen_action_type: str = "action_token",
        unified_state_dim: int = 200,
        unified_action_dim: int = 200,
        history_num: int = 1,
        action_chunk_length: int = 16,
        override_stats_path: Optional[str] = None,
        stats_level: str = "auto",
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.is_train = is_train
        self.logger = logger or logging.getLogger(__name__)
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_workers = num_workers
        self.initial_seed = seed

        self.dataset_name = dataset_name
        self.dataset_path_list = [str(p) for p in dataset_path_list]
        self.sub_dataset_names = [Path(p).name for p in dataset_path_list]

        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.force_image_size = force_image_size
        self.num_image_tokens = num_image_tokens
        self.vit_dropout_prob = vit_dropout_prob
        self.state_dropout_prob = state_dropout_prob

        _, self.vit_transform = build_vit_transform_base(
            is_train=self.is_train, force_image_size=force_image_size, **vit_transform_args
        )

        self.tokenizer = tokenizer
        self.template_name = template_name
        self.prompt_template = prompt_template
        self.instruction_template = INSTRUCTION_TEMPLATE
        self.multi_db_instruction_template = MULTI_DB_INSTRUCT_TEMPLATE

        conv = get_conv_template(self.template_name)
        self.system_prompt = conv.system_message

        self.gen_action_type = gen_action_type
        self.unified_state_dim = unified_state_dim
        self.unified_action_dim = unified_action_dim
        self.history_num = history_num
        self.action_chunk_length = action_chunk_length
        self.max_view_num = max_view_num
        self.use_fixed_view = use_fixed_view
        self.is_relative = is_relative
        self.is_abstract_action = is_abstract_action
        self.sampling_strategy = sampling_strategy

        self.num_used_episodes_per_dataset = num_used_episodes_per_dataset or [-1] * len(dataset_path_list)
        self.num_used_episodes_per_task = num_used_episodes_per_task or [-1] * len(dataset_path_list)
        self.num_used_frames_per_dataset = num_used_frames_per_dataset or [-1] * len(dataset_path_list)
        self.frame_step_size = frame_step_size or [1] * len(dataset_path_list)

        # Per-subdataset data configs (modalities + mappings).
        self.data_configs: Dict[str, BaseDataConfig] = {}
        for i, dataset_path in enumerate(self.dataset_path_list):
            sub_dataset_name = Path(dataset_path).name
            embodiment_tag = EmbodimentTag(embodiment_tags[i])
            dc_name = data_config_names[i]
            DataConfigClass = DATA_CONFIG_MAP[dc_name]
            self.data_configs[sub_dataset_name] = DataConfigClass(
                embodiment_tag=embodiment_tag,
                use_fixed_view=self.use_fixed_view,
                max_view_num=self.max_view_num,
                obs_indices=[0],
                action_indices=list(range(self.action_chunk_length)),
            )

        # On-disk metadata caches (v3.0)
        self._info: Dict[str, dict] = {}
        self._stats: Dict[str, dict] = {}
        self._tasks: Dict[str, Dict[int, str]] = {}
        self._episodes: Dict[str, list[dict]] = {}

        self._episode_to_data_file: Dict[tuple[str, int], tuple[int, int]] = {}
        self._episode_to_data_slice: Dict[tuple[str, int], tuple[int, int]] = {}
        self._episode_video_refs: Dict[str, Dict[int, Dict[str, _EpisodeVideoRef]]] = {}

        # In-process caches (per worker)
        self._episode_cache: Dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._datafile_cache: Dict[tuple[str, int, int], Any] = {}

        self._load_all_metadata()
        self._build_dataset_metadatas(override_stats_path=override_stats_path)

        self.all_steps: list[tuple[str, int, int]] = self._prepare_sample_index()
        self.rng = random.Random(self.initial_seed)
        self.set_epoch(self.initial_seed)

        self.logger.info(
            f"LeRobotV3IterableDataset '{self.dataset_name}' initialized with {len(self.all_steps)} step-units."
        )

    def __len__(self) -> int:
        return len(self.all_steps)

    # --------------------------------------------------------------------------
    # Metadata loading (LeRobot v3.0)
    # --------------------------------------------------------------------------

    def _load_all_metadata(self) -> None:
        """Load info/stats/tasks/episodes metadata from a local LeRobot v3 dataset."""
        import pandas as pd

        for dataset_path in self.dataset_path_list:
            ds_root = Path(dataset_path)
            sub_name = ds_root.name

            info_path = ds_root / "meta" / "info.json"
            stats_path = ds_root / "meta" / "stats.json"
            tasks_path = ds_root / "meta" / "tasks.parquet"

            if not info_path.exists():
                raise FileNotFoundError(f"Missing {info_path}")
            if not stats_path.exists():
                raise FileNotFoundError(f"Missing {stats_path}")
            if not tasks_path.exists():
                raise FileNotFoundError(f"Missing {tasks_path}")

            self._info[sub_name] = json.loads(info_path.read_text(encoding="utf-8"))
            self._stats[sub_name] = json.loads(stats_path.read_text(encoding="utf-8"))

            tasks_df = pd.read_parquet(tasks_path)
            # LeRobot v3 datasets can store the task string either as:
            # - an explicit "task" column, or
            # - the dataframe index (commonly saved as "__index_level_0__" in parquet metadata).
            if "task" not in tasks_df.columns:
                tasks_df = tasks_df.reset_index()

            # Try to locate the task-index and task-text columns with a few common variants.
            task_index_col = None
            for c in ("task_index", "task_id", "task_idx"):
                if c in tasks_df.columns:
                    task_index_col = c
                    break
            task_text_col = None
            for c in ("task", "instruction", "task_description", "description", "name", "index", "__index_level_0__", "level_0"):
                if c in tasks_df.columns:
                    task_text_col = c
                    break

            if task_index_col is None or task_text_col is None:
                raise ValueError(
                    f"Unexpected tasks.parquet schema: {tasks_path}\n"
                    f"Columns: {list(tasks_df.columns)}"
                )

            self._tasks[sub_name] = {
                int(idx): str(txt) for idx, txt in zip(tasks_df[task_index_col].tolist(), tasks_df[task_text_col].tolist())
            }

            # Episodes metadata is stored in parquet shards.
            episodes_files = sorted((ds_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            if not episodes_files:
                raise FileNotFoundError(f"No episodes parquet found under {ds_root / 'meta' / 'episodes'}")
            episodes_df = pd.concat([pd.read_parquet(p) for p in episodes_files], ignore_index=True)
            if "episode_index" not in episodes_df.columns:
                raise ValueError(f"episodes parquet missing 'episode_index': {ds_root / 'meta' / 'episodes'}")
            episodes_df = episodes_df.sort_values("episode_index")
            episodes = [r._asdict() for r in episodes_df.itertuples(index=False)]
            self._episodes[sub_name] = episodes

            # Precompute episode -> data slice / data file / video offset refs.
            self._episode_video_refs[sub_name] = {}
            video_keys = self._infer_video_keys_from_info(sub_name)
            for ep in episodes:
                ep_idx = int(ep["episode_index"])
                start = int(ep.get("dataset_from_index", 0))
                end = int(ep.get("dataset_to_index", 0))
                if end <= start:
                    raise ValueError(f"Bad dataset_from/to_index for episode {ep_idx} in {sub_name}: {start}..{end}")
                self._episode_to_data_slice[(sub_name, ep_idx)] = (start, end)

                data_chunk_idx = int(ep.get("data/chunk_index", 0))
                data_file_idx = int(ep.get("data/file_index", 0))
                self._episode_to_data_file[(sub_name, ep_idx)] = (data_chunk_idx, data_file_idx)

                self._episode_video_refs[sub_name][ep_idx] = {}
                for vid_key in video_keys:
                    v_chunk = int(ep.get(f"videos/{vid_key}/chunk_index", 0))
                    v_file = int(ep.get(f"videos/{vid_key}/file_index", 0))
                    v_from = float(ep.get(f"videos/{vid_key}/from_timestamp", 0.0))
                    v_to = float(ep.get(f"videos/{vid_key}/to_timestamp", 0.0))
                    self._episode_video_refs[sub_name][ep_idx][vid_key] = _EpisodeVideoRef(
                        chunk_index=v_chunk,
                        file_index=v_file,
                        from_timestamp=v_from,
                        to_timestamp=v_to,
                    )

    def _infer_video_keys_from_info(self, sub_name: str) -> list[str]:
        info = self._info[sub_name]
        features = info.get("features", {})
        return [k for k, ft in features.items() if ft.get("dtype") == "video"]

    # --------------------------------------------------------------------------
    # Metadata -> Being-H DatasetMetadata (normalization)
    # --------------------------------------------------------------------------

    def _build_dataset_metadatas(self, override_stats_path: Optional[str]) -> None:
        # Merge raw parquet-column stats into final modality-key stats using Being-H helper.
        from BeingH.dataset.datasets.vla_dataset import merge_statistics

        self.dataset_metadatas: Dict[str, DatasetMetadata] = {}
        self.transforms: Dict[str, Any] = {}
        self.dataset_fps: Dict[str, int] = {}

        for sub_name in self.sub_dataset_names:
            raw_stats = self._stats[sub_name]

            # Optional override (expects a dict with {"state":..., "action":...} at root).
            if override_stats_path and Path(override_stats_path).exists():
                override_full = json.loads(Path(override_stats_path).read_text(encoding="utf-8"))
                if "state" in override_full and "action" in override_full:
                    raw_stats = raw_stats | override_full

            data_config = self.data_configs[sub_name]
            structured = merge_statistics([raw_stats], [data_config], [1.0])
            statistics_obj = DatasetStatistics.model_validate(structured)

            modality_defs = data_config.define_modalities()
            state_meta: dict[str, StateActionMetadata] = {}
            action_meta: dict[str, StateActionMetadata] = {}
            video_meta: dict[str, VideoMetadata] = {}

            fps = int(self._info[sub_name].get("fps", 30))
            for key, defn in modality_defs.items():
                modality_type, modality_name = key.split(".", 1)
                if modality_type in ("state", "action"):
                    dim = defn.end - defn.start
                    meta_obj = StateActionMetadata(
                        absolute=defn.absolute,
                        rotation_type=defn.rotation_type,
                        shape=(dim,),
                        continuous=defn.continuous,
                    )
                    if modality_type == "state":
                        state_meta[modality_name] = meta_obj
                    else:
                        action_meta[modality_name] = meta_obj
                elif modality_type == "video":
                    vmeta = self._video_meta_from_info(sub_name, defn.source_column)
                    if vmeta is not None:
                        video_meta[modality_name] = VideoMetadata.model_validate(vmeta)

            modalities_obj = DatasetModalities(video=video_meta, state=state_meta, action=action_meta)
            md = DatasetMetadata(
                statistics=statistics_obj,
                modalities=modalities_obj,
                embodiment_tag=data_config.embodiment_tag.value,
            )

            self.dataset_metadatas[sub_name] = md
            self.dataset_fps[sub_name] = fps

            transform_pipeline = data_config.get_transforms()
            transform_pipeline.set_metadata(md)
            self.transforms[sub_name] = transform_pipeline

    def _video_meta_from_info(self, sub_name: str, source_column: str) -> Optional[dict]:
        info = self._info[sub_name]
        ft = info.get("features", {}).get(source_column)
        if not ft:
            return None

        names_list = ft["names"]
        shape_list = ft["shape"]
        height = shape_list[names_list.index("height")]
        width = shape_list[names_list.index("width")]

        info_dict = ft["info"] if "info" in ft else ft.get("video_info", {})
        channels = info_dict.get("video.channels", 3)
        fps = info_dict.get("video.fps", info.get("fps", 30))

        return {"resolution": (int(width), int(height)), "channels": int(channels), "fps": float(fps)}

    # --------------------------------------------------------------------------
    # Sampling index + DDP/worker sharding
    # --------------------------------------------------------------------------

    def set_epoch(self, seed: int) -> None:
        # Global shuffle must be identical across ranks.
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self))
        rank_splits = np.array_split(indices, self.world_size)
        self.episode_idxs_per_rank = rank_splits[self.local_rank]
        self.total_episode_idxs = indices
        self.num_files_per_rank = len(self.episode_idxs_per_rank)

    def _prepare_sample_index(self) -> list[tuple[str, int, int]]:
        if self.sampling_strategy != "step":
            raise NotImplementedError("Only sampling_strategy='step' is supported for this adapter.")

        index: list[tuple[str, int, int]] = []
        for ds_i, sub_name in enumerate(self.sub_dataset_names):
            episodes = list(self._episodes[sub_name])
            step = int(self.frame_step_size[ds_i]) if self.frame_step_size else 1
            if step <= 0:
                step = 1

            # Optional episode downsampling.
            num_ep = self.num_used_episodes_per_dataset[ds_i]
            if num_ep is not None and num_ep > 0 and len(episodes) > num_ep:
                rng = random.Random(self.initial_seed)
                episodes = rng.sample(episodes, num_ep)

            for ep in episodes:
                ep_idx = int(ep["episode_index"])
                start, end = self._episode_to_data_slice[(sub_name, ep_idx)]
                length = end - start
                if length <= 1:
                    continue

                # Random per-episode offset for frame stepping to avoid aliasing.
                start_frame_idx = random.randint(0, max(0, step - 1))
                for base_idx in range(start_frame_idx, length, step):
                    index.append((sub_name, ep_idx, int(base_idx)))

        if not index:
            self.logger.warning("No steps indexed; dataset may be empty.")
        return index

    # --------------------------------------------------------------------------
    # Data access helpers
    # --------------------------------------------------------------------------

    def _load_data_df_for_file(self, sub_name: str, chunk_index: int, file_index: int):
        import pandas as pd

        cache_key = (sub_name, chunk_index, file_index)
        if cache_key in self._datafile_cache:
            return self._datafile_cache[cache_key]

        ds_root = Path(next(p for p in self.dataset_path_list if Path(p).name == sub_name))
        info = self._info[sub_name]
        data_path_tmpl = info["data_path"]
        data_path = ds_root / data_path_tmpl.format(chunk_index=chunk_index, file_index=file_index)
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data parquet: {data_path}")

        df = pd.read_parquet(data_path)
        self._datafile_cache[cache_key] = df
        return df

    def _get_episode_arrays(self, sub_name: str, ep_idx: int) -> dict[str, np.ndarray]:
        """Return cached numpy arrays for an episode (timestamp, task_index, observation.state, action)."""
        cache_key = (sub_name, ep_idx)
        if cache_key in self._episode_cache:
            return self._episode_cache[cache_key]

        chunk_idx, file_idx = self._episode_to_data_file[(sub_name, ep_idx)]
        df = self._load_data_df_for_file(sub_name, chunk_idx, file_idx)

        start, end = self._episode_to_data_slice[(sub_name, ep_idx)]
        if "index" in df.columns:
            ep_df = df[(df["index"] >= start) & (df["index"] < end)]
        else:
            ep_df = df.iloc[start:end]
        if "frame_index" in ep_df.columns:
            ep_df = ep_df.sort_values("frame_index")

        arrays: dict[str, np.ndarray] = {}
        for col in ["timestamp", "task_index", "observation.state", "action"]:
            if col not in ep_df.columns:
                continue
            if col in ("observation.state", "action"):
                arrays[col] = np.stack(ep_df[col].values).astype(np.float32, copy=False)
            else:
                arrays[col] = ep_df[col].to_numpy()

        if "timestamp" not in arrays:
            raise ValueError(f"Episode data missing 'timestamp' for {sub_name} ep={ep_idx}")

        self._episode_cache[cache_key] = arrays
        return arrays

    # --------------------------------------------------------------------------
    # Main iteration
    # --------------------------------------------------------------------------

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch_count = 0

        while True:
            # Sharding seed must be identical across ranks/workers for consistent global shuffle.
            sharding_seed = self.initial_seed + epoch_count
            self.set_epoch(sharding_seed)

            # Operation seed differs across workers to avoid identical augmentation/dropout.
            MAX_NUMPY_SEED = 2**32 - 1
            op_seed = (sharding_seed * 1000) + (self.local_rank * 100) + worker_id
            op_seed = op_seed % MAX_NUMPY_SEED
            self.rng.seed(op_seed)
            random.seed(op_seed)
            np.random.seed(op_seed)
            epoch_count += 1

            worker_splits = np.array_split(self.episode_idxs_per_rank, num_workers)
            step_ids = worker_splits[worker_id]
            if len(step_ids) == 0:
                continue

            for step_id in step_ids:
                sub_name, ep_idx, base_index = self.all_steps[int(step_id)]
                arrays = self._get_episode_arrays(sub_name, int(ep_idx))
                traj_len = int(arrays["timestamp"].shape[0])
                if traj_len <= 1:
                    continue

                data_config = self.data_configs[sub_name]
                modality_defs = data_config.define_modalities()
                sampling_indices = data_config.get_sampling_indices()

                raw_data: Dict[str, Any] = {}
                for key, defn in modality_defs.items():
                    delta = np.array(sampling_indices.get(key, [0]), dtype=np.int64)
                    indices = int(base_index) + delta
                    padded = np.clip(indices, 0, traj_len - 1).astype(int)

                    modality_type, _ = key.split(".", 1)
                    if modality_type == "video":
                        # v3 stores per-episode offsets inside potentially shared mp4 files.
                        vid_key = defn.source_column
                        vref = self._episode_video_refs[sub_name][int(ep_idx)][vid_key]

                        ts = arrays["timestamp"][padded].astype(np.float32, copy=False)
                        ts_global = ts + np.float32(vref.from_timestamp)

                        ds_root = Path(next(p for p in self.dataset_path_list if Path(p).name == sub_name))
                        video_path_tmpl = self._info[sub_name]["video_path"]
                        video_path = ds_root / video_path_tmpl.format(
                            video_key=vid_key, chunk_index=vref.chunk_index, file_index=vref.file_index
                        )

                        frames = get_frames_by_timestamps(
                            video_path=str(video_path),
                            timestamps=ts_global,
                            video_backend=self.video_backend,
                            video_backend_kwargs=self.video_backend_kwargs,
                        )
                        raw_data[key] = frames
                    elif modality_type == "language":
                        # Task id -> task string handled later.
                        if defn.source_column in arrays:
                            task_idx = int(arrays[defn.source_column][padded[-1]])
                        else:
                            task_idx = int(arrays.get("task_index", np.array([0]))[0])
                        raw_data[key] = np.array([task_idx], dtype=np.int64)
                    else:
                        src = defn.source_column
                        if src not in arrays:
                            raise KeyError(f"Missing column '{src}' for {sub_name} ep={ep_idx}")
                        raw = arrays[src]
                        sliced = raw[:, defn.start:defn.end]
                        raw_data[key] = sliced[padded]

                language_data: Dict[str, Any] = {k: v for k, v in raw_data.items() if k.startswith("language.")}
                numerical_data: Dict[str, Any] = {
                    k: v for k, v in raw_data.items() if not k.startswith(("language.", "video."))
                }

                transformed_data = self.transforms[sub_name](numerical_data)
                mapping = data_config.UNIFIED_MAPPING

                T_state = self.history_num
                T_action = self.action_chunk_length
                state_data = torch.zeros(T_state, self.unified_state_dim, dtype=torch.float32)
                action_data = torch.zeros(T_action, self.unified_action_dim, dtype=torch.float32)
                action_mask = torch.zeros(T_action, self.unified_action_dim, dtype=torch.bool)

                drop_state_cond = (self.state_dropout_prob > 1e-9) and (random.random() < self.state_dropout_prob)
                for mod_key, (start, end) in mapping.items():
                    if mod_key not in transformed_data:
                        continue
                    src_t = transformed_data[mod_key]
                    if mod_key.startswith("state.") and not drop_state_cond:
                        state_data[:, start:end] = src_t
                    elif mod_key.startswith("action."):
                        action_data[:, start:end] = src_t
                        action_mask[:, start:end] = True

                instruction_key = next(iter(language_data.keys()), None)
                if instruction_key:
                    task_idx = int(language_data[instruction_key][0])
                    task_description = self._tasks[sub_name].get(task_idx, "Put the box into the robot basket.")
                else:
                    task_description = "Put the box into the robot basket."

                embodiment_tag = data_config.embodiment_tag
                tag_string = embodiment_tag.value
                default_id = EMBODIMENT_TAG_MAPPING[EmbodimentTag.NEW_EMBODIMENT.value]
                embodiment_id = EMBODIMENT_TAG_MAPPING.get(tag_string, default_id)

                packet = {
                    "sequence_plan": [],
                    "text_ids_list": [],
                    "image_tensor_list": [],
                    "state_tensor_list": [],
                    "action_tensor_list": [],
                    "num_tokens": 0,
                    "embodiment_id": embodiment_id,
                    "action_mask": action_mask,
                }

                # 1) system
                system_prompt = f"system\n{self.system_prompt}"
                text_ids = self.tokenizer.encode(system_prompt)
                packet["text_ids_list"].append(text_ids)
                packet["sequence_plan"].append(
                    {
                        "type": "text",
                        "has_loss": 0,
                        "enable_cfg": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "is_bos": True,
                        "is_eos": True,
                    }
                )
                packet["num_tokens"] += len(text_ids) + 3

                # 2) user
                text_ids = self.tokenizer.encode("user\n")
                packet["text_ids_list"].append(text_ids)
                packet["sequence_plan"].append(
                    {
                        "type": "text",
                        "has_loss": 0,
                        "enable_cfg": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "is_bos": True,
                        "is_eos": False,
                    }
                )
                packet["num_tokens"] += len(text_ids) + 1

                # 3) vision (stable order based on DataConfig.VIDEO_KEYS)
                video_groups: List[np.ndarray] = [raw_data[k] for k in data_config.VIDEO_KEYS if k in raw_data]
                num_views = len(video_groups)
                if self.vit_dropout_prob > 1e-9 and num_views > 1:
                    drop_decisions = [random.random() < self.vit_dropout_prob for _ in range(num_views)]
                    if all(drop_decisions):
                        drop_decisions[random.randint(0, num_views - 1)] = False
                else:
                    drop_decisions = [False] * num_views

                for v_idx, view_group in enumerate(video_groups):
                    drop_this_view = drop_decisions[v_idx]
                    for frame in view_group:
                        image_tensor = self.vit_transform(Image.fromarray(frame)).unsqueeze(0)
                        packet["sequence_plan"].append(
                            {
                                "type": "vit_image",
                                "has_loss": 0,
                                "enable_cfg": 0,
                                "special_token_loss": 0,
                                "special_token_label": None,
                                "num_image_tokens": self.num_image_tokens,
                                "drop_vit_cond": drop_this_view,
                                "is_bos": False,
                                "is_eos": False,
                            }
                        )
                        packet["num_tokens"] += self.num_image_tokens + 2
                        packet["image_tensor_list"].append(image_tensor)

                # 4) state
                packet["state_tensor_list"].append(state_data)
                packet["sequence_plan"].append(
                    {
                        "type": "state",
                        "has_loss": 0,
                        "enable_cfg": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "is_bos": False,
                        "is_eos": False,
                    }
                )
                packet["num_tokens"] += state_data.shape[0] + 2

                # 5) instruction text
                modality_feat_meta = data_config.get_feature_meta()
                view_list = [k.split(".")[-1] for k in data_config.VIDEO_KEYS]
                instruction = self._fill_instruction_template(
                    modality_feat_meta=modality_feat_meta,
                    view_list=view_list,
                    embodiment_tag=embodiment_tag.value,
                    task_description=task_description,
                    action_chunk_length=self.action_chunk_length,
                )
                text_ids = self.tokenizer.encode(instruction)
                packet["text_ids_list"].append(text_ids)
                packet["sequence_plan"].append(
                    {
                        "type": "text",
                        "has_loss": 0,
                        "enable_cfg": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "is_bos": False,
                        "is_eos": True,
                    }
                )
                packet["num_tokens"] += len(text_ids) + 2

                # 6) assistant
                text_ids = self.tokenizer.encode("assistant\n")
                packet["text_ids_list"].append(text_ids)
                packet["sequence_plan"].append(
                    {
                        "type": "text",
                        "has_loss": 0,
                        "enable_cfg": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "is_bos": True,
                        "is_eos": False,
                    }
                )
                packet["num_tokens"] += len(text_ids) + 1

                # 7) action
                packet["action_tensor_list"].append(action_data)
                packet["sequence_plan"].append(
                    {
                        "type": "action",
                        "has_loss": 1,
                        "enable_cfg": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "is_bos": False,
                        "is_eos": False,
                    }
                )
                packet["num_tokens"] += action_data.shape[0]

                # 8) end
                packet["text_ids_list"].append([])
                packet["sequence_plan"].append(
                    {
                        "type": "text",
                        "has_loss": 0,
                        "enable_cfg": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "is_bos": False,
                        "is_eos": True,
                        "is_end": True,
                    }
                )
                packet["num_tokens"] += 1

                yield packet

    def _fill_instruction_template(
        self,
        modality_feat_meta: dict,
        view_list: list[str],
        embodiment_tag: str,
        task_description: str,
        action_chunk_length: int,
    ) -> str:
        if self.prompt_template == "short":
            return task_description
        if self.prompt_template == "long":
            return self.instruction_template.format(task_description=task_description, k=action_chunk_length)
        if self.prompt_template == "detail":
            state_metas, state_dims = [], []
            action_metas, action_dims = [], []
            for k, d in modality_feat_meta.items():
                if k.startswith("state"):
                    state_metas.append(d[0])
                    state_dims.append(d[1])
                elif k.startswith("action"):
                    action_metas.append(d[0])
                    action_dims.append(d[1])

            parts = embodiment_tag.split("_")
            arm_type = parts[-2] if len(parts) >= 2 else embodiment_tag
            eef_type = parts[-1] if len(parts) >= 1 else embodiment_tag

            state_dim, state_desc = sum(state_dims), ", ".join(state_metas)
            action_dim, action_desc = sum(action_dims), ", ".join(action_metas)

            return self.multi_db_instruction_template.format(
                view_list=", ".join(view_list),
                arm_type=arm_type,
                eef_type=eef_type,
                max_state_dim=self.unified_state_dim,
                max_action_dim=self.unified_action_dim,
                state_dim=state_dim,
                state_desc=state_desc,
                action_dim=action_dim,
                action_desc=action_desc,
                task_description=task_description,
                k=action_chunk_length,
            )

        return self.instruction_template.format(task_description=task_description, k=action_chunk_length)
