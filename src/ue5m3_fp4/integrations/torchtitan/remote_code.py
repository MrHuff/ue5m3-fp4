# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-locked fix for quantized Mamba ``out_proj`` dispatch.

The pinned Nemotron-H remote code fuses its training-time output projection by
passing ``out_proj.weight`` directly to the Mamba kernel.  That is correct for
an exact ``nn.Linear`` but bypasses the ``forward`` method of a converted FP4
linear.  This narrow source patch retains the fused projection for dense
linears and requests the scan output followed by module dispatch otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

NEMOTRON_H_HF_REVISION = "78dc93a79e2533922ac8ad2c16f79b7fb747970d"
ORIGINAL_MODELING_SHA256 = "8fed3b30c627bc5c58f1f17f5941fa2641d1ea69bf52c40bac31ec0dd67dd4a9"
PATCHED_MODELING_SHA256 = "9498e7b4b28592fc03d9b00e74ae5484672a842fd8e322b69eabe1edfa14689a"

_ORIGINAL_FUSED_BLOCK = """\
            if self.training and cache_params is None:
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

_PATCHED_FUSED_BLOCK = """\
            if self.training and cache_params is None:
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def patch_nemotron_h_remote_code(path: str | Path) -> str:
    """Patch one pinned ``modeling_nemotron_h.py`` atomically.

    Unknown source revisions fail before any write.  Reapplying the operation
    to the exact patched file is idempotent.
    """

    target = Path(path)
    payload = target.read_bytes()
    digest = _sha256(payload)
    if digest == PATCHED_MODELING_SHA256:
        return digest
    if digest != ORIGINAL_MODELING_SHA256:
        raise RuntimeError(
            "refusing to patch unrecognized Nemotron-H remote code: "
            f"expected {ORIGINAL_MODELING_SHA256}, got {digest}"
        )
    text = payload.decode("utf-8")
    if text.count(_ORIGINAL_FUSED_BLOCK) != 1:
        raise RuntimeError("the hash-matched source does not contain one expected fused block")
    patched = text.replace(_ORIGINAL_FUSED_BLOCK, _PATCHED_FUSED_BLOCK).encode("utf-8")
    patched_digest = _sha256(patched)
    if patched_digest != PATCHED_MODELING_SHA256:
        raise RuntimeError(
            "internal patched-source hash mismatch: "
            f"expected {PATCHED_MODELING_SHA256}, got {patched_digest}"
        )
    with NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(patched)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return patched_digest


def require_patched_nemotron_h_assets(path: str | Path) -> Path:
    """Require a local asset directory containing the exact dispatch patch."""

    root = Path(path)
    if not root.is_dir():
        raise RuntimeError(
            "exact public training requires a local, hash-locked Hugging Face "
            "asset directory; run the public asset preparation step first"
        )
    modeling = root / "modeling_nemotron_h.py"
    if not modeling.is_file():
        raise RuntimeError(f"Nemotron-H assets are missing {modeling.name}")
    digest = _sha256(modeling.read_bytes())
    if digest != PATCHED_MODELING_SHA256:
        raise RuntimeError(
            "Nemotron-H training assets do not contain the required quantized "
            f"out_proj dispatch patch (got {digest})"
        )
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modeling_file", type=Path)
    arguments = parser.parse_args(argv)
    print(patch_nemotron_h_remote_code(arguments.modeling_file))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NEMOTRON_H_HF_REVISION",
    "ORIGINAL_MODELING_SHA256",
    "PATCHED_MODELING_SHA256",
    "patch_nemotron_h_remote_code",
    "require_patched_nemotron_h_assets",
]
