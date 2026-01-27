"""ZMQ inference server for XLerobot using Being-H + optional LoRA adapter.

Server runs on the GPU machine (your 4x4090 server). Laptop/robot connects via ZMQ.
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
import re
from typing import Optional

import torch

from BeingH.inference.beingh_policy import BeingHPolicy
from BeingH.inference.beingh_service import BeingHInferenceServer
from BeingH.utils.schema import DatasetMetadata

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _resolve_hf_or_local_model_dir(path_or_repo_id: str, *, cache_dir: str | None, revision: str | None) -> str:
    """Resolve a local directory or download a Hugging Face repo snapshot to a local directory."""
    raw = path_or_repo_id.strip()
    repo_id = None
    if raw.startswith("hf:"):
        repo_id = raw[len("hf:") :].strip()
    else:
        p = Path(raw)
        if p.exists():
            return str(p.resolve())
        if _HF_REPO_ID_RE.match(raw):
            repo_id = raw

    if repo_id is None:
        raise FileNotFoundError(
            f"Model path not found: {path_or_repo_id}\n"
            "Provide an existing local directory, or a Hugging Face repo id like:\n"
            "  BeingBeyond/Being-H05-2B_libero_robocasa\n"
            "(You can also prefix with 'hf:' to force HF download.)"
        )

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, revision=revision, resume_download=True)


class BeingHPolicyWithExternalMetadata(BeingHPolicy):
    """Load Being-H weights from `model_path`, but dataset metadata from an external JSON file."""

    def __init__(self, *args, metadata_path: str, lora_path: Optional[str] = None, **kwargs):
        self._external_metadata_path = Path(metadata_path)
        self._lora_path = lora_path
        super().__init__(*args, **kwargs)

        if self._lora_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self._lora_path)
            self.model.to(self.device)
            self.model.eval()

    def _load_metadata(self, checkpoint_dir: Path):
        # Ignore checkpoint_dir; load from explicit path.
        metadata_path = self._external_metadata_path
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadatas = json.load(f)

        metadata_dict = all_metadatas.get(self.dataset_name)
        if metadata_dict is None:
            raise ValueError(f"No metadata found for dataset '{self.dataset_name}' in {metadata_path}")

        metadata = DatasetMetadata.model_validate(metadata_dict)
        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        required=True,
        help=(
            "Path to model directory. Recommended: out_dir/finetuned_model (exported by train_lora script). "
            "You can also pass the base Being-H model directory and provide --lora_path + --metadata_path."
        ),
    )
    p.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional Hugging Face cache dir (passed to snapshot_download) when --model_path is a repo id.",
    )
    p.add_argument("--hf_revision", default=None, help="Optional Hugging Face revision/tag/commit for --model_path.")
    p.add_argument("--lora_path", default=None, help="Optional LoRA adapter dir (output of train_lora script).")
    p.add_argument(
        "--metadata_path",
        default=None,
        help="Path to xlerobot_posttrain_metadata.json (output_dir/experiment_cfg/xlerobot_posttrain_metadata.json).",
    )

    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--api_token", default=None)

    p.add_argument("--data_config_name", default="xlerobot_box2basket_left_head")
    p.add_argument("--dataset_name", default="xlerobot_posttrain")
    p.add_argument("--embodiment_tag", default="new_embodiment")
    p.add_argument("--max_view_num", type=int, default=-1)
    p.add_argument("--use_fixed_view", action="store_true", default=False)
    p.add_argument("--enable_rtc", action="store_true", default=False)
    p.add_argument("--device", default=None, help="e.g. cuda:0. Defaults to 'cuda' if available else cpu.")

    args = p.parse_args()

    # Allow --model_path to be a HF repo id, e.g. BeingBeyond/Being-H05-2B_libero_robocasa.
    args.model_path = _resolve_hf_or_local_model_dir(args.model_path, cache_dir=args.hf_cache_dir, revision=args.hf_revision)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    instruction_template = (
        "According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?"
    )

    if args.metadata_path:
        policy = BeingHPolicyWithExternalMetadata(
            model_path=args.model_path,
            data_config_name=args.data_config_name,
            dataset_name=args.dataset_name,
            embodiment_tag=args.embodiment_tag,
            instruction_template=instruction_template,
            max_view_num=args.max_view_num,
            use_fixed_view=args.use_fixed_view,
            device=device,
            enable_rtc=args.enable_rtc,
            metadata_path=args.metadata_path,
            lora_path=args.lora_path,
        )
    else:
        # Use metadata shipped inside model_path (e.g. finetuned_model/xlerobot_posttrain_metadata.json).
        policy = BeingHPolicy(
            model_path=args.model_path,
            data_config_name=args.data_config_name,
            dataset_name=args.dataset_name,
            embodiment_tag=args.embodiment_tag,
            instruction_template=instruction_template,
            max_view_num=args.max_view_num,
            use_fixed_view=args.use_fixed_view,
            device=device,
            enable_rtc=args.enable_rtc,
        )

        if args.lora_path:
            from peft import PeftModel

            policy.model = PeftModel.from_pretrained(policy.model, args.lora_path)
            policy.model.to(policy.device)
            policy.model.eval()

    server = BeingHInferenceServer(policy=policy, host=args.host, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
