# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a TorchTitan HF export with pinned public Nemotron-H assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open

NEMOTRON_H_ASSET_REPOSITORY = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base"
NEMOTRON_H_ASSET_REVISION = "78dc93a79e2533922ac8ad2c16f79b7fb747970d"
CHECKPOINT_IDENTITY_SCHEMA = "ue5m3_fp4_hf_checkpoint_identity_v1"
ASSET_IDENTITY_SCHEMA = "ue5m3_fp4_nemotron_h_asset_identity_v1"

_ASSET_FILENAMES = {
    "config.json",
    "configuration_nemotron_h.py",
    "generation_config.json",
    "modeling_nemotron_h.py",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def checkpoint_identity(checkpoint: str | Path) -> dict[str, Any]:
    """Validate and identify a local standard-HF safetensors export."""

    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint must be a local directory: {root}")
    safetensors_files = sorted(root.glob("*.safetensors"), key=lambda path: path.name)
    if not safetensors_files:
        raise FileNotFoundError(f"checkpoint contains no .safetensors files: {root}")
    if list(root.glob("*.bin")):
        raise ValueError("PyTorch pickle checkpoints are not accepted; use safetensors")

    index_path = root / "model.safetensors.index.json"
    indexed_keys: list[str] | None = None
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Hugging Face safetensors index: {index_path}") from error
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json must contain a non-empty weight_map")
        referenced_shards: set[str] = set()
        for key, shard in weight_map.items():
            if not isinstance(key, str) or not key.startswith(("backbone.", "lm_head.")):
                raise ValueError(f"non-standard Nemotron-H checkpoint key: {key!r}")
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError(f"unsafe or invalid checkpoint shard name: {shard!r}")
            shard_path = root / shard
            if not shard_path.is_file() or shard_path.suffix != ".safetensors":
                raise FileNotFoundError(f"indexed checkpoint shard is missing: {shard_path}")
            referenced_shards.add(shard)
        actual_shards = {path.name for path in safetensors_files}
        if actual_shards != referenced_shards:
            raise ValueError(
                "checkpoint shard inventory disagrees with its index: "
                f"unreferenced={sorted(actual_shards - referenced_shards)}, "
                f"missing={sorted(referenced_shards - actual_shards)}"
            )
        indexed_keys = sorted(weight_map)
    elif len(safetensors_files) != 1 or safetensors_files[0].name != "model.safetensors":
        raise FileNotFoundError(
            "a multi-file export requires model.safetensors.index.json; an unindexed "
            "export must be named model.safetensors"
        )

    shard_keys: list[str] = []
    for shard_path in safetensors_files:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            shard_keys.extend(handle.keys())
    if len(shard_keys) != len(set(shard_keys)):
        raise ValueError("the checkpoint contains duplicate tensor keys across shards")
    for key in shard_keys:
        if not key.startswith(("backbone.", "lm_head.")):
            raise ValueError(f"non-standard Nemotron-H checkpoint key: {key!r}")
    shard_keys.sort()
    if indexed_keys is not None and shard_keys != indexed_keys:
        raise ValueError("safetensors tensor inventory disagrees with weight_map")

    files = [_file_record(path, root=root) for path in safetensors_files]
    if index_path.is_file():
        files.append(_file_record(index_path, root=root))
    files.sort(key=lambda record: record["path"])
    semantic = {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "format": "huggingface_safetensors",
        "files": files,
        "tensor_key_count": len(shard_keys),
        "tensor_keys_sha256": _canonical_sha256(shard_keys),
    }
    semantic["sha256"] = _canonical_sha256(semantic)
    return semantic


def _resolve_assets(
    assets: str | Path,
    *,
    revision: str,
    local_files_only: bool,
) -> tuple[Path, str]:
    path = Path(assets).expanduser()
    if path.is_dir():
        return path.resolve(), "local_directory"
    if not isinstance(assets, str) or not assets.strip():
        raise ValueError("hf_assets must be a non-empty local directory or repository ID")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - transformers installs this dependency.
        raise ImportError("remote assets require huggingface_hub") from error
    snapshot = snapshot_download(
        repo_id=assets,
        revision=revision,
        allow_patterns=sorted(_ASSET_FILENAMES),
        local_files_only=local_files_only,
    )
    return Path(snapshot).resolve(), "huggingface_snapshot"


def _asset_identity(root: Path, *, source: str, revision: str) -> dict[str, Any]:
    files = [
        _file_record(path, root=root)
        for path in sorted(root.iterdir(), key=lambda path: path.name)
        if path.is_file() and path.name in _ASSET_FILENAMES
    ]
    required = {"config.json", "configuration_nemotron_h.py", "modeling_nemotron_h.py"}
    observed = {record["path"] for record in files}
    if missing := required - observed:
        raise FileNotFoundError(f"Nemotron-H assets are missing {sorted(missing)} in {root}")
    semantic = {
        "schema": ASSET_IDENTITY_SCHEMA,
        "source": source,
        "requested_revision": revision,
        "files": files,
    }
    semantic["sha256"] = _canonical_sha256(semantic)
    return semantic


@contextmanager
def _checkpoint_view(
    checkpoint: Path,
    assets_root: Path,
    config: Any,
) -> Iterator[Path]:
    """Create a temporary complete HF directory without copying weight shards."""

    with tempfile.TemporaryDirectory(prefix="ue5m3-fp4-hf-") as raw_directory:
        directory = Path(raw_directory)
        for source in sorted(assets_root.iterdir(), key=lambda path: path.name):
            if source.is_file() and source.name in _ASSET_FILENAMES:
                shutil.copy2(source, directory / source.name)
        # Save after copying so the exact 8B architecture overrides the 12B
        # public source model's config while retaining its auto_map metadata.
        config.save_pretrained(directory)
        for source in sorted(checkpoint.glob("*.safetensors"), key=lambda path: path.name):
            (directory / source.name).symlink_to(source)
        index_path = checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            (directory / index_path.name).symlink_to(index_path)
        yield directory


def load_hf_nemotron_h_checkpoint(
    checkpoint: str | Path,
    *,
    hf_assets: str | Path = NEMOTRON_H_ASSET_REPOSITORY,
    hf_revision: str = NEMOTRON_H_ASSET_REVISION,
    device: str | torch.device = "cuda:0",
    local_files_only: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load BF16 learned weights into the reported public 8B architecture.

    ``checkpoint`` contains only standard Hugging Face safetensors.  Config,
    tokenizer metadata, and remote-code modules come from the separately
    pinned public Nemotron-H assets, so model checkpoints never need to be
    bundled with this repository.
    """

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    checkpoint_record = checkpoint_identity(checkpoint_path)
    assets_root, assets_source = _resolve_assets(
        hf_assets,
        revision=hf_revision,
        local_files_only=local_files_only,
    )
    assets_record = _asset_identity(
        assets_root,
        source=assets_source,
        revision=hf_revision,
    )
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise ImportError("checkpoint loading requires transformers") from error
    from ue5m3_fp4.integrations.torchtitan.nemotron_h import (
        NemotronH8BArgs,
        configure_nemotron_h_sdpa,
        verify_nemotron_h_sdpa,
    )

    config = AutoConfig.from_pretrained(
        assets_root,
        trust_remote_code=True,
        local_files_only=True,
    )
    NemotronH8BArgs(hf_assets_path=str(assets_root)).apply_to_config(config)
    config.torch_dtype = torch.bfloat16
    config.use_cache = False

    with _checkpoint_view(checkpoint_path, assets_root, config) as complete_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(
            complete_checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    model.eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("model.eval() did not put every loaded module in evaluation mode")
    sdpa_record = configure_nemotron_h_sdpa(model, cudnn_enabled=False)
    verified_mixers = verify_nemotron_h_sdpa(model)
    if tuple(sdpa_record["attention_mixers"]) != verified_mixers:
        raise RuntimeError("configured and verified Nemotron-H SDPA mixers disagree")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model.to(resolved_device)
    if any(module.training for module in model.modules()):
        raise RuntimeError("a loaded module left evaluation mode during device placement")
    provenance = {
        "checkpoint": checkpoint_record,
        "assets": assets_record,
        "hf_asset_repository": str(hf_assets),
        "hf_asset_revision": hf_revision,
        "loaded_dtype": "bfloat16",
        "device": str(resolved_device),
        "model_eval_before_quantized_conversion": True,
        "attention": sdpa_record,
        "attention_verified": True,
    }
    return model, provenance


__all__ = [
    "ASSET_IDENTITY_SCHEMA",
    "CHECKPOINT_IDENTITY_SCHEMA",
    "NEMOTRON_H_ASSET_REPOSITORY",
    "NEMOTRON_H_ASSET_REVISION",
    "checkpoint_identity",
    "load_hf_nemotron_h_checkpoint",
]
