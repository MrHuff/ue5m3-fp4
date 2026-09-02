#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify, safely stage, and re-check the frozen OLMES request bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SHA256 = "b7cd708300b7b63edd45e4d973de7195b2c98384f1a9b0773f49c5a8d0e47898"
ARCHIVE_SHA256 = "0bf27af57eb1bb1b98872c4af12d419498652d935a6b745cc7ec4ecdb32d7483"
PAYLOAD_INVENTORY_SHA256 = "5af34a3cc3ee849ee6a0e4b13597f782cd23881eb63799fcb11270d47c9e953a"
REQUEST_INVENTORY_SHA256 = "613955cc0ca3a1798f85c3ce60edf422eee66ed4c0566242f0b59c7211a95271"
ORIGINAL_RUN_EVAL_SHA256 = "72baeb5324dfe798fd4a10aa741c6720b8b426c7d0a83c6e706d1b06f16a0c97"
PATCHED_RUN_EVAL_SHA256 = "fde74f9c7aab5efd0ed22ddaf776d1375eef3dea55293bfb36ff3ec5f23ab80a"
_ORIGINAL_CONDITION = (
    b"            if os.path.exists(requests_output_file) and cached_predictions is not None:\n"
)
_PATCHED_CONDITION = (
    b"            if os.path.exists(requests_output_file) and (\n"
    b"                cached_predictions is not None\n"
    b'                or os.environ.get("OLMES_REUSE_RAW_REQUESTS") == "1"\n'
    b"            ):\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe request-bundle path: {value!r}")
    return path


def _safe_symlink(member: PurePosixPath, target: str) -> None:
    target_path = PurePosixPath(target)
    resolved = posixpath.normpath(
        posixpath.join(member.parent.as_posix(), target_path.as_posix())
    )
    if (
        not target
        or target_path.is_absolute()
        or resolved == ".."
        or resolved.startswith("../")
        or not resolved.startswith("payload/")
    ):
        raise ValueError(f"unsafe archive symlink {member}: {target!r}")


def _load_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if _sha256(path) != MANIFEST_SHA256:
        raise ValueError("frozen OLMES manifest SHA-256 does not match the paper")
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise TypeError("frozen OLMES manifest must be an object")
    if document.get("schema") != "ue5m3_fp4_olmes_likelihood_request_bundle_v1":
        raise ValueError("unexpected frozen OLMES manifest schema")
    requests = document.get("request_inventory")
    payload = document.get("payload_inventory")
    if not isinstance(requests, dict) or not isinstance(payload, dict):
        raise TypeError("frozen OLMES manifest lacks inventory objects")
    requests_without_hash = dict(requests)
    request_hash = requests_without_hash.pop("inventory_sha256", None)
    payload_without_hash = dict(payload)
    payload_hash = payload_without_hash.pop("inventory_sha256", None)
    if request_hash != REQUEST_INVENTORY_SHA256 or request_hash != _canonical_sha256(
        requests_without_hash
    ):
        raise ValueError("frozen OLMES request inventory hash is invalid")
    if payload_hash != PAYLOAD_INVENTORY_SHA256 or payload_hash != _canonical_sha256(
        payload_without_hash
    ):
        raise ValueError("frozen OLMES payload inventory hash is invalid")
    expected = {
        "file_count": 146,
        "request_count": 368_932,
        "task_document_count": 79_371,
        "request_types": ["loglikelihood"],
    }
    for field, value in expected.items():
        if requests.get(field) != value:
            raise ValueError(f"frozen OLMES request inventory changed {field}")
    if len(requests.get("files", [])) != 146:
        raise ValueError("frozen OLMES request inventory does not contain 146 files")
    if document.get("archive", {}).get("sha256") != ARCHIVE_SHA256:
        raise ValueError("frozen OLMES manifest records a different archive")
    return document


def _extract(
    archive_path: Path,
    destination: Path,
    inventory: dict[str, Any],
) -> None:
    if _sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("frozen OLMES archive SHA-256 does not match the paper")
    if destination.exists():
        raise FileExistsError(destination)
    records = {str(record["path"]): dict(record) for record in inventory["files"]}
    if len(records) != inventory["file_count"]:
        raise ValueError("payload inventory contains duplicate paths")
    directories: set[str] = set()
    for name in records:
        parent = _safe_path(name).parent
        while parent.parts:
            directories.add(parent.as_posix())
            parent = parent.parent
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            name = _safe_path(member.name).as_posix()
            if name in members:
                raise ValueError(f"duplicate archive member: {name}")
            members[name] = member
            if member.isdir():
                if name not in directories:
                    raise ValueError(f"undeclared archive directory: {name}")
            elif member.isreg():
                record = records.get(name)
                if record is None or record.get("kind") != "file":
                    raise ValueError(f"undeclared archive file: {name}")
                if member.size != record.get("size_bytes"):
                    raise ValueError(f"archive size differs for {name}")
            elif member.issym():
                record = records.get(name)
                if record is None or record.get("kind") != "symlink":
                    raise ValueError(f"undeclared archive symlink: {name}")
                if member.linkname != record.get("target"):
                    raise ValueError(f"archive symlink target differs for {name}")
                _safe_symlink(PurePosixPath(name), member.linkname)
            else:
                raise ValueError(f"unsupported archive member type: {name}")
        if set(members) != directories | set(records):
            raise ValueError("archive member inventory differs from its manifest")
        for name in sorted(directories, key=lambda item: (item.count("/"), item)):
            (destination / name).mkdir()
        for name, record in sorted(records.items()):
            target = destination / name
            if record["kind"] == "file":
                source = archive.extractfile(members[name])
                if source is None:
                    raise RuntimeError(f"could not read archived file {name}")
                with target.open("xb") as handle:
                    shutil.copyfileobj(source, handle, length=1 << 20)
                if (
                    target.stat().st_size != record["size_bytes"]
                    or _sha256(target) != record["sha256"]
                ):
                    raise RuntimeError(f"extracted file failed verification: {name}")
                target.chmod(int(record["mode"]))
            else:
                os.symlink(str(record["target"]), target)


def _verify_requests(directory: Path, manifest: dict[str, Any]) -> None:
    inventory = manifest["request_inventory"]
    for record in inventory["files"]:
        relative = _safe_path(str(record["path"]))
        path = directory / relative.as_posix()
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"staged OLMES request changed: {relative}")


def prepare(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    work_root = args.work_root.expanduser().resolve()
    evaluation = args.evaluation_dir.expanduser().resolve()
    extracted = work_root / "extracted"
    _extract(args.archive.expanduser().resolve(), extracted, manifest["payload_inventory"])
    requests = extracted / "payload" / "requests"
    cache = extracted / "payload" / "hf-cache"
    if not requests.is_dir() or not cache.is_dir():
        raise RuntimeError("verified bundle lacks requests or Hugging Face cache")
    _verify_requests(requests, manifest)
    for record in manifest["request_inventory"]["files"]:
        relative = _safe_path(str(record["path"]))
        destination = evaluation / relative.as_posix()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(requests / relative.as_posix(), destination)
    _verify_requests(evaluation, manifest)


def verify(args: argparse.Namespace) -> None:
    _verify_requests(
        args.evaluation_dir.expanduser().resolve(),
        _load_manifest(args.manifest),
    )


def patch_olmes(args: argparse.Namespace) -> None:
    path = args.source.expanduser().resolve()
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != ORIGINAL_RUN_EVAL_SHA256:
        raise ValueError("OLMES run_eval.py does not match the pinned source")
    if original.count(_ORIGINAL_CONDITION) != 1:
        raise ValueError("OLMES replay gate did not match exactly once")
    patched = original.replace(_ORIGINAL_CONDITION, _PATCHED_CONDITION, 1)
    if hashlib.sha256(patched).hexdigest() != PATCHED_RUN_EVAL_SHA256:
        raise RuntimeError("OLMES replay patch does not match the reviewed source")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(patched)
    temporary.chmod(stat.S_IMODE(path.stat().st_mode))
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True, type=Path)
    prepare_parser.add_argument("--archive", required=True, type=Path)
    prepare_parser.add_argument("--work-root", required=True, type=Path)
    prepare_parser.add_argument("--evaluation-dir", required=True, type=Path)
    prepare_parser.set_defaults(function=prepare)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True, type=Path)
    verify_parser.add_argument("--evaluation-dir", required=True, type=Path)
    verify_parser.set_defaults(function=verify)
    patch_parser = commands.add_parser("patch-olmes")
    patch_parser.add_argument("--source", required=True, type=Path)
    patch_parser.set_defaults(function=patch_olmes)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
