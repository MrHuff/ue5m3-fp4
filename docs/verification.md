# Public-alpha reference verification

This page records checks run against the standalone extraction on 2026-09-01
and the complete-history scan repeated on 2026-09-02.
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
version declared by package metadata. A matrix covering released Python and
PyTorch versions remains useful follow-up qualification; it is not evidence
claimed by this public alpha.

## Source checks

```bash
uvx --from ruff==0.12.11 ruff check .
uvx --from ruff==0.12.11 ruff format --check .
PYTHONPATH=src python -m pytest -q
```

Results: lint passed, all 21 Python files were formatted, and 77 tests passed.
The test count includes a check that every extraction-snapshot digest in
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

Gitleaks 8.29.1 was downloaded from its official GitHub release, its Linux
arm64 archive was checked against the release checksum, and `gitleaks git`
reported no leaks across all five commits through `f43d29d` on 2 September
2026. Version 8.29.1 was used instead of 8.30.1 because a public upstream
report says the latter can fail to apply its default rules
([gitleaks#2170](https://github.com/gitleaks/gitleaks/issues/2170)). This does
not cover later uncommitted documentation: the publication operator must repeat
the complete-history scan after committing the final public tree and before
changing repository visibility. GitHub private vulnerability reporting must
then be enabled and verified immediately after the repository becomes public.

## Scope and follow-up validation

Graphcore approved public release under Apache-2.0 with the current `NOTICE`.
The remaining publication operations are the post-commit history scan and
enabling and verifying GitHub private vulnerability reporting after the
visibility change.

GPU numerical checks, released Python/PyTorch matrices, native kernels, and
model-scale integration remain future qualification targets. They are not
required to publish this portable CPU fake-quantization reference, and this
repository does not use their absence to imply paper-scale reproduction.
