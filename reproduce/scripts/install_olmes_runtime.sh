#!/usr/bin/env bash
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the paper-compatible OLMES environment without changing the training
# environment in the parent container.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 NEW_VIRTUAL_ENV_DIRECTORY" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
runtime_dir=$1
lock=$repo_root/reproduce/environment/olmes-cp312-linux-aarch64.lock
expected_lock=ccb078f4c1c87766931e2dafa0c6002640c2d35d5c7e41965ea3531f3bc16da5
expected_olmes_commit=8e2743734066b073c5d8498d1b8220f67a21a2d6
expected_olmes_tree=991940b9a2b37f8491ff29d1d22487b209fe750f
ai2_olmo_commit=090253dac6688f2532509daa7aa2eb5fae50e956
alpaca_eval_commit=db85f8065408b842100436a45f56c65d3a0dd6a6

if [[ -e "$runtime_dir" ]]; then
  echo "refusing to overwrite existing runtime directory: $runtime_dir" >&2
  exit 2
fi
if [[ "$(python -c 'import platform; print(platform.machine())')" != "aarch64" ]]; then
  echo "the published OLMES lock is qualified only for Linux aarch64" >&2
  exit 2
fi
if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "the published OLMES lock requires CPython 3.12" >&2
  exit 2
fi

echo "$expected_lock  $lock" | sha256sum --check --strict -
observed_olmes_commit=$(git -C "$repo_root/third_party/olmes" rev-parse HEAD)
observed_olmes_tree=$(git -C "$repo_root/third_party/olmes" rev-parse 'HEAD^{tree}')
if [[ "$observed_olmes_commit" != "$expected_olmes_commit" || \
      "$observed_olmes_tree" != "$expected_olmes_tree" ]]; then
  echo "the OLMES submodule does not match the paper runtime" >&2
  exit 2
fi

python -m venv --system-site-packages "$runtime_dir"
runtime_python=$runtime_dir/bin/python
"$runtime_python" -m pip install --upgrade \
  "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"$runtime_python" -m pip install \
  --disable-pip-version-check --no-cache-dir --no-deps \
  --require-hashes --requirement "$lock"
"$runtime_python" -m pip install \
  --disable-pip-version-check --no-cache-dir --no-deps --no-build-isolation \
  "git+https://github.com/allenai/OLMo.git@$ai2_olmo_commit" \
  "git+https://github.com/natolambert/alpaca_eval.git@$alpaca_eval_commit"
"$runtime_python" -m pip install \
  --disable-pip-version-check --no-cache-dir --no-deps --no-build-isolation \
  --editable "$repo_root/third_party/olmes"

"$runtime_python" - "$ai2_olmo_commit" "$alpaca_eval_commit" <<'PY'
import json
import sys
from importlib.metadata import distribution, version


def direct_commit(distribution_name: str) -> str | None:
    metadata = distribution(distribution_name)
    direct_url = metadata.read_text("direct_url.json")
    if direct_url is None:
        return None
    document = json.loads(direct_url)
    return document.get("vcs_info", {}).get("commit_id")


expected = {
    "ai2-olmo": sys.argv[1],
    "alpaca-eval": sys.argv[2],
}
observed = {name: direct_commit(name) for name in expected}
if observed != expected:
    raise SystemExit(f"VCS dependency revisions differ: expected {expected}, got {observed}")
if version("ai2-olmes") != "0.1.0":
    raise SystemExit(f"unexpected OLMES version: {version('ai2-olmes')}")
print(
    json.dumps(
        {
            "schema": "ue5m3_fp4_public_olmes_install_v1",
            "ai2_olmes": version("ai2-olmes"),
            "ai2_olmo_commit": observed["ai2-olmo"],
            "alpaca_eval_commit": observed["alpaca-eval"],
        },
        indent=2,
        sort_keys=True,
    )
)
PY
