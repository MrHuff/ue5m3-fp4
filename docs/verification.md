# Release-candidate verification

This page records checks run against the standalone extraction on 2026-09-01.
It is evidence for the portable CPU reference only; it is not a supported
environment matrix or a GPU/native-kernel qualification.

## Environment

- Linux aarch64, kernel 6.12.77
- Python 3.12.3
- PyTorch `2.9.0a0+145a3a7bda.nv25.10`
- PyYAML 6.0.3
- safetensors 0.6.2
- Ruff 0.12.11

The PyTorch build is a development environment, not the minimum supported
version declared by package metadata. Python 3.11/3.12 and released PyTorch
versions still require a clean CI matrix before publication.

## Source checks

```bash
uvx --from ruff==0.12.11 ruff check .
uvx --from ruff==0.12.11 ruff format --check .
PYTHONPATH=src python -m pytest -q
```

Results: lint passed, all 21 Python files were formatted, and 77 tests passed.
The test count includes a check that every release-candidate digest in
`SOURCE_PROVENANCE.md` matches its file.

The synthetic checkpoint-reload example completed under all three activation
policies:

```bash
PYTHONPATH=src python examples/tiny_mlp.py --activation-mode current_tensor
PYTHONPATH=src python examples/tiny_mlp.py --activation-mode training_replay
PYTHONPATH=src python examples/tiny_mlp.py --activation-mode calibrated_frozen
```

Each run reported the explicit
`quantized_ue5m3_fp4_decoded_torch` numerical path.

## Distribution checks

`python -m build` completed in an isolated build environment. The resulting
sdist and wheel both contained all four canonical YAML recipes. The wheel was
installed with `--no-deps` into a fresh virtual environment that inherited the
host's tested PyTorch stack. From that environment:

- the package-resource API discovered and loaded all four recipes;
- all 77 tests passed against the installed wheel; and
- all three synthetic inference-policy examples completed.

Final artifact digests are intentionally not frozen here. Release artifacts
must be rebuilt from the reviewed tag and their hashes published outside the
artifact whose hash they describe.

## Security-oriented checks

`detect-secrets==1.5.0` scanned every tracked and unignored file and reported
zero findings. A separate pattern audit found no private storage/workspace
paths or common literal token forms outside the negative tests that contain
those strings intentionally.

This does not close the release gate: Gitleaks or TruffleHog must still scan the
complete final Git history, and the destination repository must have a tested
private vulnerability-reporting channel.

## Checks still required

- GPU numerical checks on documented hardware;
- supported released Python/PyTorch CPU and CUDA matrices;
- a complete-history Gitleaks or TruffleHog report;
- organizational release and licensing/attribution approval; and
- model-scale integration and reproduction tests after those assets are added.
