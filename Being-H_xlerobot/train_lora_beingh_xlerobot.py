"""LoRA fine-tuning for Being-H on XLerobot LeRobot-v3 datasets.

This script is intentionally placed under `Being-H_xlerobot/` so we can:
- keep upstream Being-H code untouched
- override `configs.*` to register XLerobot dataset + modality config
"""

from __future__ import annotations

# Ensure upstream `BeingH` is importable while keeping `Being-H_xlerobot/` first on sys.path
# (so `configs.*` resolves to our overrides).
import sys
from pathlib import Path

_VLA_ROOT = Path(__file__).resolve().parents[1]
_BEINGH_ROOT = _VLA_ROOT / "Being-H"
if str(_BEINGH_ROOT) not in sys.path:
    sys.path.insert(1, str(_BEINGH_ROOT))

import argparse
import json
import hashlib
import logging
import os
import re
import traceback
import types
import inspect
import functools
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep, time
from typing import Any

import torch
import torch.distributed as dist
import yaml
from safetensors.torch import save_file
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from BeingH.dataset.base_dataset import PackedDataset, RobotDatasetConfig, collate_wrapper
from BeingH.inference.beingh_policy import VERSION_CONFIGS
from BeingH.model.beingvla import BeingH, BeingHConfig
from BeingH.model.layers import InternVLConnector
from BeingH.model.vit_model.internvit_navit import InternVisionConfig, InternVisionModel


logger = logging.getLogger("xlerobot_lora_train")

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _strip_state_dict_wrappers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Normalize parameter keys for saving/loading.

    With FSDP/DDP + activation checkpointing, the raw `state_dict()` keys may include wrapper prefixes like:
      - "module." (DDP)
      - "_fsdp_wrapped_module." (FSDP)
      - "_checkpoint_wrapped_module." (torch checkpoint_wrapper)

    PEFT's adapter helpers expect keys that match the original (unwrapped) module paths. If we don't strip
    these wrappers, saving LoRA weights can silently produce an empty adapter file.
    """
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module.") :]
        # Remove all occurrences to handle nested wrappers.
        k2 = k2.replace("_fsdp_wrapped_module.", "")
        k2 = k2.replace("_checkpoint_wrapped_module.", "")
        cleaned[k2] = v
    return cleaned


def _ensure_prepare_inputs_for_generation(model: torch.nn.Module) -> None:
    """
    PEFT's CAUSAL_LM wrapper expects `prepare_inputs_for_generation` on the base model.

    BeingH doesn't implement it (we don't do text generation here), so provide a small shim.
    """
    if hasattr(model, "prepare_inputs_for_generation"):
        return

    def _prepare_inputs_for_generation(self, input_ids, **kwargs):  # noqa: ANN001
        lm = getattr(self, "language_model", None)
        if lm is not None and hasattr(lm, "prepare_inputs_for_generation"):
            return lm.prepare_inputs_for_generation(input_ids, **kwargs)
        return {"input_ids": input_ids, **kwargs}

    model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)


@dataclass
class TrainCfg:
    # Paths
    base_model_path: str
    dataset_config_file: str = "Being-H_xlerobot/dataset_configs/xlerobot_box2basket_left_head.yaml"
    output_dir: str = "outputs/xlerobot_box2basket_lora"

    # Training
    max_steps: int = 2000
    save_adapter_at_end: bool = True
    log_every: int = 20

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    num_workers: int = 4
    seed: int = 42

    # Memory
    use_gradient_checkpoint: bool = True

    # Packing / prompt
    conv_style: str = "internlm2-chat"
    prompt_template: str = "long"
    expected_num_tokens: int = 32768
    # Mirror upstream Being-H defaults (these are important for memory usage).
    max_num_tokens_per_sample: int = 8192
    max_num_tokens: int = 32768
    max_buffer_size: int = 50
    prefer_buffer_before: int = 16384
    attn_mode: str = "causal"

    # Robot dataset config
    max_view_num: int = -1
    use_fixed_view: bool = False
    is_relative: bool = False
    is_abstract_action: bool = False
    history_num: int = 1

    vit_dropout_prob: float = 0.0
    state_dropout_prob: float = 0.0

    # FSDP
    sharding_strategy: str = "FULL_SHARD"
    backward_prefetch: str = "BACKWARD_PRE"
    cpu_offload: bool = False
    num_replicate: int = 1
    num_shard: int = 8

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

def _resolve_hf_or_local_model_dir(path_or_repo_id: str, *, cache_dir: str | None, revision: str | None) -> str:
    """
    Resolve a local checkpoint directory or download a Hugging Face repo snapshot.

    Accepts:
    - Local dir path (must exist)
    - Hugging Face repo id: e.g. "BeingBeyond/Being-H05-2B_libero_robocasa"
    - Explicit HF prefix: "hf:BeingBeyond/Being-H05-2B_libero_robocasa"
    """
    raw = path_or_repo_id.strip()
    repo_id = None
    if raw.startswith("hf:"):
        repo_id = raw[len("hf:") :].strip()
    else:
        p = Path(raw)
        if p.exists():
            return str(p.resolve())
        # Heuristic: only treat as HF repo id when it matches the standard "org/name" pattern.
        if _HF_REPO_ID_RE.match(raw):
            repo_id = raw

    if repo_id is None:
        raise FileNotFoundError(
            f"Base model path not found: {path_or_repo_id}\n"
            "Provide an existing local directory, or a Hugging Face repo id like:\n"
            "  BeingBeyond/Being-H05-2B_libero_robocasa\n"
            "(You can also prefix with 'hf:' to force HF download.)"
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download models from Hugging Face. "
            "Install it with: pip install huggingface_hub"
        ) from e

    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, revision=revision, resume_download=True)


def _resolve_hf_or_local_model_dir_distributed(path_or_repo_id: str, *, cache_dir: str | None, revision: str | None) -> str:
    """
    Same as _resolve_hf_or_local_model_dir, but only rank0 downloads.

    Important: do NOT use NCCL collectives here (broadcast_object_list), because if the HF download takes longer
    than PyTorch's default store timeout (often 10 minutes), non-rank0 workers can crash with a timeout.
    We use a filesystem-based rendezvous instead (shared on the same node).
    """
    # Use a stable sync file location across ranks; output_dir is guaranteed to be shared on-node.
    out_dir = Path(os.environ.get("XLE_ROBOT_OUT_DIR", os.getcwd()))
    sync_dir = out_dir / ".hf_resolve"
    sync_dir.mkdir(parents=True, exist_ok=True)

    # Key by request to avoid stale reads when users reuse the same output_dir for different base models/revisions.
    key = hashlib.sha1(f"{path_or_repo_id}|{cache_dir}|{revision}".encode("utf-8")).hexdigest()[:12]
    done_path = sync_dir / f"base_model_path.{key}.txt"
    err_path = sync_dir / f"base_model_path.{key}.error.txt"

    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    if rank == 0:
        try:
            resolved = _resolve_hf_or_local_model_dir(path_or_repo_id, cache_dir=cache_dir, revision=revision)
            tmp = done_path.with_suffix(".tmp")
            tmp.write_text(resolved, encoding="utf-8")
            tmp.replace(done_path)
            if err_path.exists():
                err_path.unlink()
            return resolved
        except Exception:
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            raise

    # Non-rank0: wait for rank0 to finish download/resolve.
    while True:
        if err_path.exists():
            raise RuntimeError(
                "Rank0 failed to resolve/download the base model from Hugging Face.\n"
                f"See: {err_path}\n\n{err_path.read_text(encoding='utf-8')}"
            )
        if done_path.exists():
            return done_path.read_text(encoding="utf-8").strip()
        sleep(2)


def _setup_distributed() -> int:
    dist.init_process_group("nccl")
    # torchrun sets LOCAL_RANK for per-process GPU assignment. If we don't set it explicitly,
    # every rank defaults to cuda:0, which breaks FSDP (device_id mismatch) and wastes GPUs.
    device_id = int(os.environ.get("LOCAL_RANK", dist.get_rank() % torch.cuda.device_count()))
    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return device_id


def _setup_logging(out_dir: Path, rank: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers: list[Any] = []
    if rank == 0:
        handlers.append(logging.StreamHandler())
        handlers.append(logging.FileHandler(out_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARN, format=fmt, handlers=handlers)


def _detect_llm_version(llm_config) -> str:
    # Mirror BeingHPolicy._detect_llm_version.
    if getattr(llm_config, "layer_module", None):
        if "Qwen3" in llm_config.layer_module:
            return "qwen3"
        if "Qwen2" in llm_config.layer_module:
            return "qwen2.5"
    arch = getattr(llm_config, "architectures", None) or []
    if "Qwen3ForCausalLM" in arch:
        return "qwen3"
    return "qwen2.5"


def _enable_activation_checkpointing(wrapped_model: torch.nn.Module, *, base_model: torch.nn.Module | None = None) -> None:
    """
    Enable activation/gradient checkpointing for memory savings.

    - Uses PyTorch's `apply_activation_checkpointing` (non-reentrant) like upstream Being-H training.
    - Extends the upstream check_fn to also checkpoint Qwen2/Qwen3 decoder layers (the largest activations).
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    try:
        from BeingH.train.fsdp_utils import grad_checkpoint_check_fn as _beingh_check_fn
    except Exception:
        _beingh_check_fn = None

    llm_layer_classes: tuple[type[torch.nn.Module], ...] = ()
    try:
        from BeingH.model.llm.qwen3_navit import Qwen3DecoderLayer, Qwen3MoTDecoderLayer  # type: ignore

        llm_layer_classes += (Qwen3DecoderLayer, Qwen3MoTDecoderLayer)
    except Exception:
        pass
    try:
        from BeingH.model.llm.qwen2_navit import Qwen2DecoderLayer, Qwen2MoTDecoderLayer, Qwen2MoEDecoderLayer  # type: ignore

        llm_layer_classes += (Qwen2DecoderLayer, Qwen2MoTDecoderLayer, Qwen2MoEDecoderLayer)
    except Exception:
        pass

    has_fsdp = any(isinstance(m, FSDP) for m in wrapped_model.modules())
    if has_fsdp:
        # IMPORTANT: When using FSDP, checkpoint the *FSDP-wrapped* submodules (children) instead of the raw
        # underlying modules. Otherwise, checkpoint recomputation can call a raw module directly while its
        # parameters are still sharded/flat (leading to empty LayerNorm weights or 1-D Linear weights).
        def _check_fn(m: torch.nn.Module) -> bool:
            if not isinstance(m, FSDP):
                return False
            inner = getattr(m, "_fsdp_wrapped_module", None)
            if inner is None:
                return False
            if _beingh_check_fn is not None and _beingh_check_fn(inner):
                return True
            return isinstance(inner, llm_layer_classes)
    else:
        # DDP / single-GPU: safe to checkpoint modules directly.
        def _check_fn(m: torch.nn.Module) -> bool:
            if _beingh_check_fn is not None and _beingh_check_fn(m):
                return True
            return isinstance(m, llm_layer_classes)

    apply_activation_checkpointing(
        wrapped_model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=_check_fn,
    )


def _load_beingh_model(base_model_path: str) -> BeingH:
    # Build model exactly like inference, then load all *.safetensors weights.
    from safetensors import safe_open

    cfg = BeingHConfig.from_pretrained(base_model_path)

    llm_version = _detect_llm_version(cfg.llm_config)
    QwenConfigClass, LanguageModelClass, _ = VERSION_CONFIGS[llm_version]

    llm_config_dict = cfg.llm_config.to_dict()
    llm_cfg = QwenConfigClass.from_dict(llm_config_dict)
    expert_config_dict = llm_config_dict.get("expert_config")
    if expert_config_dict:
        if not isinstance(expert_config_dict, dict):
            expert_config_dict = expert_config_dict.to_dict()
        llm_cfg.expert_config = QwenConfigClass.from_dict(expert_config_dict)

    vit_config_dict = cfg.vit_config.to_dict()
    vit_cfg = InternVisionConfig.from_dict(vit_config_dict)

    cfg.llm_config = llm_cfg
    cfg.vit_config = vit_cfg

    language_model = LanguageModelClass(cfg.llm_config)
    vit_model = InternVisionModel(cfg.vit_config)
    connector = InternVLConnector(
        llm_hidden_size=cfg.llm_config.hidden_size,
        vit_hidden_size=cfg.vit_config.hidden_size,
        downsample_ratio=cfg.downsample_ratio,
    )
    model = BeingH(language_model, vit_model, connector, cfg)
    # Keep the model weights in bf16 early to reduce GPU peak memory during FSDP init/sharding.
    model = model.to(dtype=torch.bfloat16)

    # Load all safetensors shards in directory.
    state_dict: dict[str, torch.Tensor] = {}
    for p in sorted(Path(base_model_path).glob("*.safetensors")):
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    # Official checkpoints can carry a few legacy modules that aren't used by the current codebase.
    # We load non-strictly but still fail hard on *missing* keys, and on unexpected keys we don't recognize.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing key(s) when loading BeingH checkpoint: {missing[:50]}")

    allowed_unexpected_prefixes = ("proprio_encoder.",)
    unexpected_unknown = [k for k in unexpected if not k.startswith(allowed_unexpected_prefixes)]
    if unexpected_unknown:
        raise RuntimeError(f"Unexpected key(s) when loading BeingH checkpoint: {unexpected_unknown[:50]}")
    if unexpected and dist.get_rank() == 0:
        logger.warning(f"Ignoring unexpected checkpoint keys: {unexpected}")
    return model


def _build_special_tokens(tokenizer) -> dict[str, int]:
    tokens = ["<|im_start|>", "<|im_end|>", "<img>", "</img>", "<|state_start|>", "<|state_end|>"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    newline = tokenizer.encode("\n")
    if len(newline) != 1:
        raise ValueError("Newline must be a single token for Being-H packing.")
    return {
        "bos_token_id": ids[0],
        "eos_token_id": ids[1],
        "start_of_image": ids[2],
        "end_of_image": ids[3],
        "start_of_state": ids[4],
        "end_of_state": ids[5],
        "newline_token_id": newline[0],
    }


def _save_dataset_metadata_minimal(train_dataset: PackedDataset, dataset_meta: dict, out_dir: Path) -> list[Path]:
    """
    Write `{group}_metadata.json` files compatible with BeingHPolicy._load_metadata.

    Returns list of written metadata file paths.
    """
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    if rank != 0:
        return []

    exp_cfg_dir = out_dir / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)

    grouped_names = list(dataset_meta.keys())
    written: list[Path] = []

    for i, grouped_ds in enumerate(train_dataset.grouped_datasets):
        group_name = grouped_names[i]
        if not hasattr(grouped_ds, "dataset_metadatas"):
            continue

        variants = {k: v.model_dump(mode="json") for k, v in grouped_ds.dataset_metadatas.items()}
        first_variant = next(iter(variants.values()))
        payload = {group_name: first_variant, f"{group_name}_variants": variants}

        path = exp_cfg_dir / f"{group_name}_metadata.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(path)

    return written


def _export_merged_checkpoint(
    base_model_path: str,
    adapter_dir: Path,
    metadata_files: list[Path],
    out_dir: Path,
) -> Path:
    """
    Create a self-contained checkpoint directory that BeingHPolicy can load directly:
      - config/tokenizer files
      - model.safetensors (LoRA merged into base weights)
      - {dataset_name}_metadata.json (normalization stats)
    """
    finetuned_dir = out_dir / "finetuned_model"
    finetuned_dir.mkdir(parents=True, exist_ok=True)

    # 1) Tokenizer (required by BeingHPolicy)
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    tok.save_pretrained(finetuned_dir)

    # 2) Config (BeingHConfig.json)
    cfg = BeingHConfig.from_pretrained(base_model_path)
    cfg.save_pretrained(finetuned_dir)

    # 3) Merge LoRA -> base weights and save a single safetensors
    from peft import PeftModel

    base = _load_beingh_model(base_model_path)
    _ensure_prepare_inputs_for_generation(base)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    save_file(merged.state_dict(), finetuned_dir / "model.safetensors")

    # 4) Copy metadata files into the checkpoint dir so inference can normalize.
    for m in metadata_files:
        (finetuned_dir / m.name).write_text(m.read_text(encoding="utf-8"), encoding="utf-8")

    return finetuned_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        required=True,
        help=(
            "Base model directory OR Hugging Face repo id.\n"
            "Examples:\n"
            "  --base_model_path /path/to/Being-H05-2B_libero_robocasa\n"
            "  --base_model_path BeingBeyond/Being-H05-2B_libero_robocasa"
        ),
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional Hugging Face cache dir (passed to snapshot_download). Useful on servers with small $HOME.",
    )
    parser.add_argument("--hf_revision", default=None, help="Optional Hugging Face revision/tag/commit.")
    parser.add_argument("--dataset_config_file", default=TrainCfg.dataset_config_file)
    parser.add_argument("--output_dir", default=TrainCfg.output_dir)
    parser.add_argument("--max_steps", type=int, default=TrainCfg.max_steps)
    parser.add_argument("--learning_rate", type=float, default=TrainCfg.learning_rate)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TrainCfg.gradient_accumulation_steps)
    parser.add_argument("--num_workers", type=int, default=TrainCfg.num_workers)
    parser.add_argument("--seed", type=int, default=TrainCfg.seed)
    parser.add_argument(
        "--use_gradient_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=TrainCfg.use_gradient_checkpoint,
        help="Enable activation/gradient checkpointing to reduce memory usage (slower).",
    )

    # Packing / prompt (memory-sensitive). Keep defaults aligned with Being-H/train.py.
    parser.add_argument(
        "--expected_num_tokens",
        type=int,
        default=TrainCfg.expected_num_tokens,
        help="Soft target token count for a packed batch (lower this if you hit OOM on 24GB GPUs).",
    )
    parser.add_argument(
        "--max_num_tokens_per_sample",
        type=int,
        default=TrainCfg.max_num_tokens_per_sample,
        help="Maximum tokens allowed in one raw sample; longer samples are skipped.",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=TrainCfg.max_num_tokens,
        help="Hard limit on tokens in a packed batch; flush if adding a sample would exceed it.",
    )
    parser.add_argument(
        "--max_buffer_size",
        type=int,
        default=TrainCfg.max_buffer_size,
        help="Maximum number of oversized samples kept in the overflow buffer.",
    )
    parser.add_argument(
        "--prefer_buffer_before",
        type=int,
        default=TrainCfg.prefer_buffer_before,
        help="While batch length is below this, pop from the overflow buffer before new sampling.",
    )

    # Robot dataset knobs (primarily memory/perf).
    parser.add_argument(
        "--max_view_num",
        type=int,
        default=TrainCfg.max_view_num,
        help="Max number of camera views to use (-1 = all). Lower this to reduce video tokens and VRAM.",
    )
    parser.add_argument(
        "--use_fixed_view",
        action=argparse.BooleanOptionalAction,
        default=TrainCfg.use_fixed_view,
        help="If enabled, always use the first camera view (reduces VRAM; may reduce performance).",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=TrainCfg.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=TrainCfg.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=TrainCfg.lora_dropout)
    parser.add_argument("--lora_target_modules", type=str, default=TrainCfg.lora_target_modules)

    # FSDP
    parser.add_argument("--sharding_strategy", type=str, default=TrainCfg.sharding_strategy)
    parser.add_argument("--backward_prefetch", type=str, default=TrainCfg.backward_prefetch)
    parser.add_argument("--cpu_offload", action="store_true", default=TrainCfg.cpu_offload)
    parser.add_argument("--num_replicate", type=int, default=TrainCfg.num_replicate)
    parser.add_argument("--num_shard", type=int, default=TrainCfg.num_shard)

    parser.add_argument(
        "--export_merged_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, export a self-contained finetuned checkpoint with LoRA merged (out_dir/finetuned_model).",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    # Used by _resolve_hf_or_local_model_dir_distributed() for a filesystem rendezvous across ranks.
    os.environ.setdefault("XLE_ROBOT_OUT_DIR", str(out_dir.resolve()))
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed as accelerate_set_seed
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "accelerate is required for this training script.\n"
            "Install it with: pip install accelerate\n"
            "Then launch with: accelerate launch --num_processes <N> ...\n"
            "(torchrun also works if you already have a distributed environment.)"
        ) from e

    # Prefer FSDP (saves memory vs DDP) when running multi-GPU.
    fsdp_plugin = None
    try:
        # accelerate renamed this plugin a few times; support common names.
        try:
            from accelerate.utils import FullyShardedDataParallelPlugin as _FSDPPlugin  # type: ignore
        except Exception:
            from accelerate.utils import FSDPPlugin as _FSDPPlugin  # type: ignore

        from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from BeingH.train.fsdp_utils import FSDP_WRAPPER_CLASSES

        # Add LLM decoder layer classes to the FSDP auto-wrap set so we shard the largest part of the model
        # (and can safely apply activation checkpointing on the FSDP-wrapped layers).
        wrapper_classes = set(FSDP_WRAPPER_CLASSES)
        try:
            from BeingH.model.llm.qwen3_navit import Qwen3DecoderLayer, Qwen3MoTDecoderLayer  # type: ignore

            wrapper_classes.update([Qwen3DecoderLayer, Qwen3MoTDecoderLayer])
        except Exception:
            pass
        try:
            from BeingH.model.llm.qwen2_navit import Qwen2DecoderLayer, Qwen2MoTDecoderLayer, Qwen2MoEDecoderLayer  # type: ignore

            wrapper_classes.update([Qwen2DecoderLayer, Qwen2MoTDecoderLayer, Qwen2MoEDecoderLayer])
        except Exception:
            pass

        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=wrapper_classes)

        plugin_kwargs = {
            "sharding_strategy": ShardingStrategy[args.sharding_strategy],
            "backward_prefetch": BackwardPrefetch[args.backward_prefetch],
            # accelerate plugins typically accept `cpu_offload: bool` and build CPUOffload internally.
            "cpu_offload": bool(args.cpu_offload),
            "auto_wrap_policy": auto_wrap_policy,
            "use_orig_params": True,
            "mixed_precision_policy": MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        }

        # Only pass kwargs supported by the installed accelerate version.
        sig = inspect.signature(_FSDPPlugin.__init__)
        plugin_kwargs = {k: v for k, v in plugin_kwargs.items() if k in sig.parameters}
        fsdp_plugin = _FSDPPlugin(**plugin_kwargs)
    except Exception:
        # FSDP plugin is optional; if unavailable, accelerate will fall back to DDP.
        # On large models this can OOM, so we log a warning to make it obvious.
        fsdp_plugin = None

    accel_kwargs = {
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "fsdp_plugin": fsdp_plugin,
    }
    sig = inspect.signature(Accelerator.__init__)
    accel_kwargs = {k: v for k, v in accel_kwargs.items() if k in sig.parameters}
    accelerator = Accelerator(**accel_kwargs)

    # Ensure each process uses its own GPU (important for FSDP).
    if accelerator.device.type == "cuda":
        torch.cuda.set_device(accelerator.local_process_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    _setup_logging(out_dir, rank)

    accelerate_set_seed(int(args.seed))

    if world_size > 1:
        dist_type = str(getattr(accelerator, "distributed_type", ""))
        if "FSDP" not in dist_type:
            logger.warning(
                f"accelerate distributed_type={dist_type}; expected FSDP for large models. "
                "If you hit OOM, ensure accelerate supports FSDP and/or run `accelerate config` and select FSDP."
            )

    if accelerator.is_main_process:
        logger.info(f"World size: {world_size}")
        logger.info(f"Config: {vars(args)}")

    # Resolve base model directory (local path or download from HF).
    resolved_base_model_path = _resolve_hf_or_local_model_dir_distributed(
        args.base_model_path, cache_dir=args.hf_cache_dir, revision=args.hf_revision
    )
    args.base_model_path = resolved_base_model_path
    if accelerator.is_main_process:
        logger.info(f"Resolved base_model_path: {resolved_base_model_path}")

    # Load tokenizer from base model.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left"
    special_tokens = _build_special_tokens(tokenizer)

    # Load model from base model directory.
    model = _load_beingh_model(args.base_model_path)

    # Derive packing params from model config to avoid mismatch.
    force_image_size = int(model.config.force_image_size)
    patch_size = int(model.config.vit_config.patch_size)
    downsample_ratio = float(model.config.downsample_ratio)
    num_image_tokens = int((force_image_size // patch_size) ** 2 * (downsample_ratio**2))

    # Apply LoRA adapters.
    from peft import LoraConfig, TaskType, get_peft_model

    for p in model.parameters():
        p.requires_grad = False

    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    # Helps PEFT locate base config/tokenizer when loading/merging the adapter later.
    lora_cfg.base_model_name_or_path = str(args.base_model_path)

    _ensure_prepare_inputs_for_generation(model)

    model = get_peft_model(model, lora_cfg)
    # FSDP flattening requires uniform dtype within each wrapped module. PEFT may create LoRA params
    # in fp32 even when the base weights are bf16, so we cast the full model to bf16.
    model = model.to(dtype=torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.4f}%)")

    # Dataset config YAML.
    dataset_meta = yaml.safe_load(Path(args.dataset_config_file).read_text(encoding="utf-8"))

    robot_cfg = RobotDatasetConfig(
        max_view_num=int(args.max_view_num),
        use_fixed_view=bool(args.use_fixed_view),
        is_relative=False,
        is_abstract_action=False,
        gen_action_type=model.config.gen_action_type,
        action_chunk_length=model.config.action_chunk_length,
        history_num=1,
        prompt_template="long",
        vit_dropout_prob=0.0,
        state_dropout_prob=0.0,
    )

    template_name = getattr(model.config, "template", "internlm2-chat")
    train_dataset = PackedDataset(
        tokenizer=tokenizer,
        template_name=template_name,
        grouped_dataset_meta=dataset_meta,
        robot_config=robot_cfg,
        special_tokens=special_tokens,
        force_image_size=force_image_size,
        img_patch_size=patch_size,
        img_downsample_ratio=downsample_ratio,
        expected_num_tokens=int(args.expected_num_tokens),
        max_num_tokens_per_sample=int(args.max_num_tokens_per_sample),
        max_num_tokens=int(args.max_num_tokens),
        max_buffer_size=int(args.max_buffer_size),
        prefer_buffer_before=int(args.prefer_buffer_before),
        attn_mode="causal",
        local_rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        is_train=True,
        logger=logger,
    )

    written_meta = _save_dataset_metadata_minimal(train_dataset, dataset_meta, out_dir)
    if accelerator.is_main_process:
        for p in written_meta:
            logger.info(f"Wrote metadata: {p}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(TrainCfg.beta1, TrainCfg.beta2),
        eps=TrainCfg.eps,
        weight_decay=TrainCfg.weight_decay,
    )

    # Let accelerate wrap the model/optimizer for DDP or FSDP.
    model, optim = accelerator.prepare(model, optim)
    model.train()

    # IMPORTANT: Apply activation checkpointing *after* FSDP wrapping.
    # If applied before FSDP, the auto-wrap policy may miss modules (they become CheckpointWrapper),
    # and during checkpoint recomputation some sharded parameters can appear as empty tensors on non-owning ranks.
    if bool(args.use_gradient_checkpoint):
        if accelerator.is_main_process:
            logger.info("Enabling activation/gradient checkpointing (memory savings, slower).")
        _enable_activation_checkpointing(model, base_model=accelerator.unwrap_model(model))

    start_t = time()
    optim.zero_grad(set_to_none=True)

    global_step = 0
    for batch in train_loader:
        if global_step >= args.max_steps:
            break

        with accelerator.accumulate(model):
            # Being-H's `collate_wrapper()` returns a `SimpleCustomBatch` with a `.cuda(device)` helper
            # (but no `.to(...)`). Keep this compatible with accelerate device placement.
            if hasattr(batch, "cuda"):
                batch = batch.cuda(accelerator.device)
            elif hasattr(batch, "to"):
                batch = batch.to(accelerator.device)
            data = batch.to_dict() if hasattr(batch, "to_dict") else batch
            with accelerator.autocast():
                loss_dict = model(**data)
                action_loss = loss_dict["action_loss"]
                und_loss = loss_dict["und_loss"]
                loss = action_loss + und_loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # FSDP exposes clip_grad_norm_ for sharded params.
                if hasattr(model, "clip_grad_norm_"):
                    model.clip_grad_norm_(TrainCfg.max_grad_norm)
                else:
                    accelerator.clip_grad_norm_(model.parameters(), TrainCfg.max_grad_norm)

                optim.step()
                optim.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if accelerator.is_main_process and global_step % TrainCfg.log_every == 0:
                elapsed = time() - start_t
                logger.info(
                    f"step {global_step}/{args.max_steps} "
                    f"act_loss={action_loss.item():.4f} und_loss={und_loss.item():.4f} "
                    f"lr={optim.param_groups[0]['lr']:.2e} elapsed_s={elapsed:.1f}"
                )
            global_step += 1

    # Save adapter (rank0 only).
    if TrainCfg.save_adapter_at_end:
        from peft import PeftModel

        adapter_dir = out_dir / "lora_adapter"
        if accelerator.is_main_process:
            adapter_dir.mkdir(parents=True, exist_ok=True)

        # Gather full state dict once; filter LoRA keys for adapter save.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                full_state = model.state_dict()
        else:
            full_state = accelerator.get_state_dict(model)

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            assert isinstance(unwrapped, PeftModel)
            from peft.utils import get_peft_model_state_dict

            # Normalize wrapper prefixes so PEFT can find LoRA keys reliably under FSDP/checkpointing.
            cleaned_full_state = _strip_state_dict_wrappers(full_state)
            adapter_state = get_peft_model_state_dict(unwrapped, state_dict=cleaned_full_state)
            if not adapter_state:
                # Fallback: keep any tensor that looks like LoRA weights.
                adapter_state = {k: v for k, v in cleaned_full_state.items() if "lora_" in k}

            if not adapter_state:
                raise RuntimeError(
                    "LoRA adapter state_dict is empty. This usually means the FSDP/checkpoint wrapper prefixes "
                    "weren't stripped correctly, or LoRA target modules didn't match the base model."
                )

            # Debug-friendly logging: confirm LoRA keys exist in the gathered state dict and show a few examples.
            lora_like_count = 0
            lora_like_examples: list[str] = []
            for k in cleaned_full_state.keys():
                if "lora_" in k:
                    lora_like_count += 1
                    if len(lora_like_examples) < 5:
                        lora_like_examples.append(k)
            logger.info(f"LoRA-like keys in full state dict: {lora_like_count}")
            if lora_like_examples:
                logger.info(f"Example LoRA keys: {lora_like_examples}")

            try:
                approx_mb = sum(int(v.numel()) * int(v.element_size()) for v in adapter_state.values()) / (1024 * 1024)
            except Exception:
                approx_mb = -1
            logger.info(
                f"LoRA adapter tensors: {len(adapter_state)} keys"
                + (f", ~{approx_mb:.1f} MiB" if approx_mb >= 0 else "")
            )
            adapter_examples = list(adapter_state.keys())[:5]
            if adapter_examples:
                logger.info(f"Example adapter keys: {adapter_examples}")

            # Safetensors doesn't like non-contiguous / non-CPU tensors; normalize once here.
            adapter_state_cpu: dict[str, torch.Tensor] = {}
            for k, v in adapter_state.items():
                adapter_state_cpu[k] = v.detach().cpu().contiguous()

            # Let PEFT write adapter_config.json + README.md. Under some wrapper combinations (FSDP + checkpointing),
            # PEFT can still produce an empty `adapter_model.safetensors` even when `state_dict` is populated.
            # We always overwrite the weights file with the exact adapter tensors we computed.
            unwrapped.save_pretrained(adapter_dir, state_dict=adapter_state_cpu, safe_serialization=True)

            adapter_weights_path = adapter_dir / "adapter_model.safetensors"
            if not adapter_weights_path.exists():
                raise RuntimeError("Expected PEFT to create adapter_model.safetensors, but it was not found.")

            # A real LoRA adapter for this model should be tens of MB. If we see a tiny file, something is wrong.
            save_file(adapter_state_cpu, str(adapter_weights_path), metadata={"format": "pt"})
            if adapter_weights_path.stat().st_size < 1_000_000:
                raise RuntimeError(
                    "Adapter weights file is unexpectedly small after rewrite. "
                    "Please inspect the logged adapter_state keys to confirm LoRA weights are present."
                )
            logger.info(f"Adapter weights file size: {adapter_weights_path.stat().st_size / (1024 * 1024):.1f} MiB")

            # Save a minimal run config for reproducibility.
            (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
            logger.info(f"Saved LoRA adapter to: {adapter_dir}")

            if args.export_merged_model:
                merged_dir = _export_merged_checkpoint(args.base_model_path, adapter_dir, written_meta, out_dir)
                logger.info(f"Exported merged finetuned checkpoint to: {merged_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
