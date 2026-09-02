# Release checklist for 0.2.0a0

This checklist applies to the expanded release containing the CUDA/Triton
numerical path, public TorchTitan/Nemotron-H method integration, and sanitized
paper-result artifacts. It does not certify a byte-identical rerun of the
historical jobs.

Record commands, environment, results, skips, and artifact hashes in
[`verification.md`](verification.md). Do not mark a gate complete because an
older release passed it.

## Reproduction claims

- [ ] Confirm the README separately describes:
  - the exact released numerical/kernel implementation;
  - the runnable public TorchTitan method reconstruction;
  - audited, sanitized paper outputs; and
  - unavailable byte-identical historical replay.
- [ ] Confirm every runnable config is under `reproduce/configs/` and accepted
  by the launcher.
- [ ] Confirm every record-only config is under `reproduce/historical_specs/`
  and rejected by the launcher.
- [ ] Confirm native NVFP4 documentation states the exact Transformer Engine
  and hardware requirements and reports only the effective runtime behavior.
- [ ] Confirm no text implies that the public deterministic data loader
  recreates the unavailable historical Mosaic object order or cursor.
- [ ] Confirm no text implies that audited CSV outputs were recomputed from
  publicly distributed checkpoints or evaluation inputs.
- [ ] Confirm single-run evidence and evaluation-example bootstrap intervals
  are described accurately.

## Source, privacy, and licensing

- [ ] Review the final source allowlist; exclude credentials, private storage
  locations, internal-only paths, private orchestration, checkpoints, and
  model weights.
- [ ] `python reproduce/scripts/audit_public_tree.py --history`
- [ ] Run a working-tree and complete-history secret scan against the final
  commit; record tool versions and results.
- [ ] Verify downloaders exclude model-weight formats and fail closed on an
  unexpected asset hash.
- [ ] Confirm Apache-2.0 headers and `LICENSE` are present for project files.
- [ ] Review `NOTICE` against the final expanded extraction and retain
  applicable NVIDIA attribution.
- [ ] Confirm third-party submodule licenses and immutable Git revisions.
- [ ] Confirm the release owner has approved the final expanded source and
  notice boundary for publication.
- [ ] Confirm `.env.example` contains placeholders only.

## Static, CPU, and package checks

- [ ] `ruff check src tests examples reproduce`
- [ ] `ruff format --check src tests examples reproduce`
- [ ] `PYTHONPATH=src:third_party/torchtitan python -m pytest`
- [ ] `PYTHONPATH=src python examples/tiny_mlp.py --activation-mode current_tensor`
- [ ] `PYTHONPATH=src python examples/tiny_mlp.py --activation-mode training_replay`
- [ ] `PYTHONPATH=src python examples/tiny_mlp.py --activation-mode calibrated_frozen`
- [ ] `python reproduce/scripts/validate_bundle.py`
- [ ] Parse each runnable TOML through the pinned TorchTitan config manager.
- [ ] Validate `CITATION.cff` as CFF 1.2.0 and confirm version `0.2.0a0` matches
  `pyproject.toml`.
- [ ] `python -m build`
- [ ] Inspect wheel and sdist inventories for required code, recipes,
  reproduction metadata, and prohibited generated/private files.
- [ ] Install the wheel into a clean environment and test package resource and
  CLI entry points.
- [ ] Run `reproduce/scripts/install_olmes_runtime.sh` in a fresh CPython 3.12
  Linux aarch64 environment and verify the lock plus all three source pins.
- [ ] Confirm documentation says a complete Git checkout is required for
  submodule-backed training; wheel/sdist installation alone is insufficient.

## CUDA/Triton and model integration

- [ ] Build `reproduce/Dockerfile` from the final commit and record the image
  digest.
- [ ] Run `reproduce/scripts/verify_runtime.py` inside the image.
- [ ] Run all CUDA/Triton tests on a recorded supported NVIDIA GPU.
- [ ] Verify E2M1/UE5M3 quantization against the portable reference at edge,
  tie, underflow, saturation, B=16, and B=32 cases.
- [ ] Verify the K=64 issue-RZ GEMM and selected output-grid behavior on the
  documented CUDA/Triton runtime.
- [ ] Construct the reported 8B Nemotron-H model from the pinned, patched
  no-weight assets and verify parameter count, 52-layer pattern, SDPA mixers,
  and state-dict roots.
- [ ] Verify B16 and B32 conversion selects the exact expected 112 internal
  linears and leaves only the vocabulary head excluded from FP4, with its BF16
  checkpoint parameter retained and its matrix multiplication computed in FP32.
- [ ] Verify the Mamba fused training path cannot bypass a converted
  `out_proj`.
- [ ] Run a minimal TorchTitan optimizer-step smoke and prove D=50 advances
  once per optimizer step rather than per accumulation microbatch.
- [ ] For native NVFP4 configs, verify or explicitly record that the pinned
  Transformer Engine build and suitable Blackwell hardware were unavailable.

## Checkpoints, evaluation, and reference artifacts

- [ ] Exercise DCP-to-HF conversion on a public smoke checkpoint and verify the
  strict state-dict adapter and safetensors inventory.
- [ ] Run BF16 and quantized validation smoke inputs through the public CLI;
  verify provenance differentiates BF16, D=1, cold D=50, and
  calibrated/frozen paths.
- [ ] Verify calibration/validation overlap, duplicate records, changed files,
  and invalid checkpoint layouts fail closed.
- [ ] Verify the public OLMES wrapper runs or dry-runs all seven numeric paths,
  records a new identity for public task reconstruction, and accepts frozen
  replay only after the supplied archive and manifest pass every hash and
  inventory check.
- [ ] Verify all sanitized result-table hashes and invariants in
  `reproduce/reference_results/provenance.json`.
- [ ] Regenerate reference figures/tables and compare their hashes with
  `generated/artifact_manifest.json`.
- [ ] Scan sanitized CSV, JSON, and generated files for private storage or job
  identifiers.

## Publication operations

- [ ] Review the final diff and ensure `SOURCE_PROVENANCE.md` hashes were
  refreshed only after source changes stopped.
- [ ] Commit the exact reviewed tree and rerun all commit-dependent checks.
- [ ] Tag the reviewed commit as `v0.2.0a0`.
- [ ] Build release artifacts from that tag and publish their SHA-256 digests.
- [ ] Verify the public repository renders README, license, citation, security,
  and reproduction links correctly.
- [ ] Verify GitHub private vulnerability reporting remains enabled.

## Explicit non-claims for this release

These are known boundaries, not failures to disclose:

- no historical model checkpoints or private Mosaic MDS inventory/cursor are
  distributed;
- no frozen quantized-OLMES request archive is distributed;
- the public data order is a new deterministic identity;
- one independent training trajectory exists per reported configuration;
- the probe-matched path is software fake quantization, not native UE5M3
  hardware; and
- audited paper metrics can be inspected and re-rendered but cannot be
  recomputed from their original inputs using this repository alone.
