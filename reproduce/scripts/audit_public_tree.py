#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed on checkpoints, private locations, or likely secrets.

The default audit covers every tracked file plus every non-ignored untracked
file, so it is useful before committing as well as in CI.  ``--history`` also
scans every reachable Git blob for credential-shaped content.  Third-party
submodule contents are excluded: their pinned gitlinks and URLs are audited,
while their source trees remain the responsibility of the upstream projects.
"""

from __future__ import annotations

import argparse
import configparser
import os
import re
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
MAX_PROJECT_FILE_BYTES = 50 * 1024 * 1024

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
FORBIDDEN_NAMES = {
    ".env",
    "credentials.json",
    "pytorch_model.bin",
}
FORBIDDEN_PATH_PARTS = {"checkpoint", "checkpoints"}

# Build sentinels from fragments so this source file does not flag itself.
PRIVATE_TEXT = (
    "s3" + "://us-east" + "-1-large-model-training",
    "/" + "volt/",
    "/" + "workspace/",
    "PET_" + "MASTER_ADDR",
    "WANDB_" + "API_KEY",
    "AWS_" + "SECRET_ACCESS_KEY",
    "github.com/" + "graphcore-research/",
)
SECRET_PATTERNS = (
    ("PEM private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("AWS access-key identifier", re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("GitHub personal token", re.compile(rb"ghp_[A-Za-z0-9]{30,}")),
    ("GitHub fine-grained token", re.compile(rb"github_pat_[A-Za-z0-9_]{40,}")),
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{30,}")),
)
EXPECTED_SUBMODULES = {
    "third_party/TransformerEngine": (
        "https://github.com/NVIDIA/TransformerEngine.git",
        "01aef4fc721bd12fd09cd56d53a314aee1b953d6",
    ),
    "third_party/olmes": (
        "https://github.com/allenai/olmes.git",
        "8e2743734066b073c5d8498d1b8220f67a21a2d6",
    ),
    "third_party/torchtitan": (
        "https://github.com/pytorch/torchtitan.git",
        "e37f83f58b35fdbceed9a5916b3490c16247ac9c",
    ),
}


def _git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.check_output(
        ("git", "-C", str(REPOSITORY), *arguments),
        input=input_bytes,
    )


def _project_paths() -> tuple[Path, ...]:
    payload = _git(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    paths = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if relative.parts and relative.parts[0] == "third_party":
            continue
        paths.append(relative)
    return tuple(sorted(set(paths), key=lambda path: path.as_posix()))


def _scan_payload(payload: bytes, *, label: str) -> list[str]:
    errors = []
    for sentinel in PRIVATE_TEXT:
        if sentinel.encode() in payload:
            errors.append(f"{label}: contains private location or environment marker")
    for kind, pattern in SECRET_PATTERNS:
        if pattern.search(payload):
            errors.append(f"{label}: contains a likely {kind}")
    return errors


def _scan_secrets(payload: bytes, *, label: str) -> list[str]:
    errors = []
    for kind, pattern in SECRET_PATTERNS:
        if pattern.search(payload):
            errors.append(f"{label}: contains a likely {kind}")
    return errors


def _audit_path(relative: Path) -> list[str]:
    errors = []
    path = REPOSITORY / relative
    label = relative.as_posix()
    lowered_parts = {part.lower() for part in relative.parts}
    lowered_name = relative.name.lower()
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append(f"{label}: model/checkpoint file type is forbidden")
    if lowered_name in FORBIDDEN_NAMES or (
        lowered_name.startswith("pytorch_model") and lowered_name.endswith(".bin")
    ):
        errors.append(f"{label}: model weight or environment file is forbidden")
    if lowered_parts & FORBIDDEN_PATH_PARTS:
        errors.append(f"{label}: checkpoint directories are forbidden")
    if path.is_symlink():
        target = (path.parent / os.readlink(path)).resolve()
        try:
            target.relative_to(REPOSITORY.resolve())
        except ValueError:
            errors.append(f"{label}: symlink escapes the repository")
        return errors
    if not path.is_file():
        # Gitlinks are directories in a recursive checkout and were excluded
        # above; anything else in the project allowlist must be a regular file.
        errors.append(f"{label}: is not a regular file")
        return errors
    size = path.stat().st_size
    if size > MAX_PROJECT_FILE_BYTES:
        errors.append(f"{label}: {size} bytes exceeds the public-file limit")
        return errors
    errors.extend(_scan_payload(path.read_bytes(), label=label))
    return errors


def _audit_history() -> list[str]:
    errors = []
    observed: set[str] = set()
    for line in _git("rev-list", "--objects", "--all").decode().splitlines():
        object_id, _, path = line.partition(" ")
        if not path or object_id in observed or path.startswith("third_party/"):
            continue
        if _git("cat-file", "-t", object_id).strip() != b"blob":
            continue
        observed.add(object_id)
        size = int(_git("cat-file", "-s", object_id))
        if size > MAX_PROJECT_FILE_BYTES:
            errors.append(f"history:{object_id}:{path}: blob is unexpectedly large")
            continue
        payload = _git("cat-file", "blob", object_id)
        # Earlier public test fixtures may intentionally spell private-location
        # sentinels. The history gate is credential-focused; the complete
        # current tree receives both location and credential checks above.
        errors.extend(_scan_secrets(payload, label=f"history:{object_id}:{path}"))
    return errors


def _audit_submodules() -> list[str]:
    """Verify the public URLs and exact gitlinks, including the staged index."""

    errors = []
    gitmodules = REPOSITORY / ".gitmodules"
    if not gitmodules.is_file():
        return [".gitmodules: missing submodule configuration"]
    try:
        indexed = _git("show", ":.gitmodules")
    except subprocess.CalledProcessError:
        return [".gitmodules: is not staged or tracked"]
    working = gitmodules.read_bytes()
    if indexed != working:
        errors.append(".gitmodules: staged index differs from the working tree")

    parser = configparser.ConfigParser()
    parser.read_string(working.decode("utf-8"))
    observed: dict[str, str] = {}
    for section in parser.sections():
        if not section.startswith('submodule "'):
            errors.append(f".gitmodules: unexpected section {section!r}")
            continue
        path = parser.get(section, "path", fallback="")
        url = parser.get(section, "url", fallback="")
        if not path or not url or path in observed:
            errors.append(f".gitmodules: invalid or duplicate entry {section!r}")
            continue
        observed[path] = url
    expected_urls = {path: value[0] for path, value in EXPECTED_SUBMODULES.items()}
    if observed != expected_urls:
        errors.append(".gitmodules: URL/path inventory differs from the public release")

    for path, (_, expected_commit) in EXPECTED_SUBMODULES.items():
        fields = _git("ls-files", "--stage", "--", path).decode().split()
        if len(fields) < 4 or fields[0] != "160000" or fields[1] != expected_commit:
            errors.append(f"{path}: staged gitlink differs from {expected_commit}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also scan every reachable project Git blob for likely credentials",
    )
    args = parser.parse_args()

    paths = _project_paths()
    errors = [error for path in paths for error in _audit_path(path)]
    errors.extend(_audit_submodules())
    if args.history:
        errors.extend(_audit_history())
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(f"public-tree audit failed with {len(errors)} finding(s)")
    scope = "working tree and Git history" if args.history else "working tree"
    print(f"audited {len(paths)} project files across the {scope}; no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
