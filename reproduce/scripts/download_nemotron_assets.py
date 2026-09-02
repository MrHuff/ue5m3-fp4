#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download pinned Nemotron-H configuration, tokenizer, and remote code only.

The allowlist intentionally excludes model parameters.  The resulting content
inventory can be archived beside a run's resolved configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base"
REVISION = "78dc93a79e2533922ac8ad2c16f79b7fb747970d"
MANIFEST_NAME = "UE5M3_ASSET_MANIFEST.json"
MODEL_SOURCE = "modeling_nemotron_h.py"
MODEL_SOURCE_ORIGINAL_SHA256 = (
    "8fed3b30c627bc5c58f1f17f5941fa2641d1ea69bf52c40bac31ec0dd67dd4a9"
)
MODEL_SOURCE_PATCHED_SHA256 = "9498e7b4b28592fc03d9b00e74ae5484672a842fd8e322b69eabe1edfa14689a"
ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "*.model",
    "*.py",
    "**/*.py",
)
IGNORE_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
    "*.h5",
    "*.msgpack",
)
FORBIDDEN_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack"}
)

_FUSED_OUT_PROJ_ORIGINAL = """            if self.training and cache_params is None:
                out = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )
"""
_FUSED_OUT_PROJ_PATCHED = """            if self.training and cache_params is None:
                fused_kwargs = {
                    "D": self.D,
                    "chunk_size": self.chunk_size,
                    "seq_idx": None,  # was seq_idx
                    "activation": self.activation,
                    "rmsnorm_weight": self.norm.weight,
                    "rmsnorm_eps": self.norm.variance_epsilon,
                    "headdim": self.head_dim,
                    "ngroups": self.n_groups,
                    "norm_before_gate": False,
                    "return_final_states": False,
                    **dt_limit_kwargs,
                }
                if type(self.out_proj) is nn.Linear:
                    out = mamba_split_conv1d_scan_combined(
                        projected_states,
                        self.conv1d.weight.squeeze(1),
                        self.conv1d.bias,
                        self.dt_bias,
                        A,
                        outproj_weight=self.out_proj.weight,
                        outproj_bias=self.out_proj.bias,
                        **fused_kwargs,
                    )
                else:
                    scan_output = mamba_split_conv1d_scan_combined(
                        projected_states,
                        self.conv1d.weight.squeeze(1),
                        self.conv1d.bias,
                        self.dt_bias,
                        A,
                        **fused_kwargs,
                    )
                    out = self.out_proj(scan_output)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify a previously downloaded directory without network access",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        if path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith("pytorch_model"):
            raise RuntimeError(
                f"refusing parameter/weight artifact in asset directory: {relative}"
            )
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not any(record["path"] == "config.json" for record in records):
        raise RuntimeError("downloaded assets do not contain config.json")
    if not any(str(record["path"]).startswith("tokenizer") for record in records):
        raise RuntimeError("downloaded assets do not contain tokenizer files")
    if not any(str(record["path"]).endswith(".py") for record in records):
        raise RuntimeError("downloaded assets do not contain Nemotron-H remote code")
    return records


def _manifest(records: list[dict[str, object]]) -> dict[str, object]:
    canonical_inventory = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schema": "ue5m3_nemotron_h_assets_v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "weights_included": False,
        "remote_code_patch": {
            "purpose": "dispatch converted Mamba out_proj through module.forward",
            "original_sha256": MODEL_SOURCE_ORIGINAL_SHA256,
            "patched_sha256": MODEL_SOURCE_PATCHED_SHA256,
            "fused_only_for_exact_dense_linear": True,
        },
        "file_count": len(records),
        "inventory_sha256": hashlib.sha256(canonical_inventory).hexdigest(),
        "files": records,
    }


def _verify(root: Path) -> dict[str, object]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"asset manifest does not exist: {manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_source = root / MODEL_SOURCE
    if not model_source.is_file() or _sha256(model_source) != MODEL_SOURCE_PATCHED_SHA256:
        raise RuntimeError("Nemotron-H remote-code patch is absent or has been modified")
    actual = _manifest(_inventory(root))
    if recorded != actual:
        raise RuntimeError("asset content differs from UE5M3_ASSET_MANIFEST.json")
    return actual


def _apply_quantized_out_proj_patch(root: Path) -> None:
    path = root / MODEL_SOURCE
    if not path.is_file():
        raise FileNotFoundError(f"pinned remote model source is missing: {path}")
    original_hash = _sha256(path)
    if original_hash != MODEL_SOURCE_ORIGINAL_SHA256:
        raise RuntimeError(
            "pinned modeling_nemotron_h.py differs before patching: "
            f"expected {MODEL_SOURCE_ORIGINAL_SHA256}, got {original_hash}"
        )
    source = path.read_text(encoding="utf-8")
    if source.count(_FUSED_OUT_PROJ_ORIGINAL) != 1:
        raise RuntimeError("Nemotron-H patch anchors are not unique")
    patched = source.replace(
        _FUSED_OUT_PROJ_ORIGINAL,
        _FUSED_OUT_PROJ_PATCHED,
    )
    if hashlib.sha256(patched.encode("utf-8")).hexdigest() != MODEL_SOURCE_PATCHED_SHA256:
        raise RuntimeError("internal error: patched Nemotron-H source digest differs")
    path.write_text(patched, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.verify_only:
        result = _verify(output)
        print(json.dumps(result, sort_keys=True))
        return

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to merge into non-empty output directory: {output}; "
            "use --verify-only for an existing download"
        )
    output.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install 'huggingface-hub' to download the pinned assets") from error

    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=str(output),
        cache_dir=str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None,
        allow_patterns=list(ALLOW_PATTERNS),
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    _apply_quantized_out_proj_patch(output)
    result = _manifest(_inventory(output))
    (output / MANIFEST_NAME).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
