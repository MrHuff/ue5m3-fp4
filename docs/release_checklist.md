# Public-alpha release checklist

This checklist covers the portable decoded-PyTorch fake-quantization reference
that is present in this repository. Graphcore approved public release of this
standalone extraction under Apache-2.0 with the current `NOTICE` on 2 September
2026. The approval does not broaden the repository's technical claims: this
alpha is neither a native UE5M3 implementation nor a complete reproduction of
the paper's 8B experiments.

## Public-alpha release basis

- [x] Obtain Graphcore approval for public release under Apache-2.0.
- [x] Review and approve the current `NOTICE` for this extraction.
- [x] Build the extraction on fresh Git history without importing source Git
  objects.
- [x] Keep credentials, non-public checkpoint locations, private storage names,
  and cluster manifests out of source, tests, and examples.
- [x] Run Gitleaks 8.29.1 over all five commits through `f43d29d`; the
  2 September 2026 scan reported no leaks.
- [ ] After committing the final public-release documentation, rerun Gitleaks
  over the complete resulting history before changing repository visibility.
- [ ] Immediately after changing repository visibility, enable GitHub private
  vulnerability reporting and verify that GitHub reports it as enabled.
- [x] Provide a private fallback contact in `SECURITY.md`.
- [x] Exclude corporate logos and internal report templates.
- [x] Add author, report, contact, license, and repository metadata in
  `CITATION.cff` and `pyproject.toml`.

## Portable reference verification

- [x] Provide a synthetic end-to-end training, checkpoint-reload, and
  quantized-inference example.
- [x] Document that the decoded-Torch path is software fake quantization, not
  native UE5M3 hardware or the paper's probe-matched comparator.
- [x] Require explicit model-conversion coverage rather than silently choosing
  eligible linears.
- [x] Keep the canonical training and inference recipes in source
  distributions.
- [x] Verify that those recipes are installed and discoverable from a wheel.
- [x] Build the wheel and sdist, install the wheel into a fresh environment,
  and rerun all tests and examples in the recorded reference environment.
- [x] Run lint, formatting checks, 77 CPU tests, and all three synthetic
  inference-policy examples.
- [ ] From the final tagged commit, rebuild any published wheel and sdist and
  publish their SHA-256 digests with the release artifacts.
- [ ] Verify the README from a clean public clone after the visibility change.

## Publication operations

- [ ] Commit the exact reviewed public-alpha tree.
- [ ] Confirm that `SOURCE_PROVENANCE.md` hashes match the final extracted
  sources.
- [ ] Complete the final full-history secret scan described above.
- [ ] Change `MrHuff/ue5m3-fp4` to public visibility and verify the rendered
  README, license, citation, and security links.
- [ ] Enable private vulnerability reporting and verify its enabled state.
- [ ] Tag the reviewed commit before publishing package artifacts.

## Non-blocking integration roadmap

The following work would support additional environments or stronger
reproduction claims. It is intentionally not a gate for publishing the current
portable CPU fake-quantization reference:

- Test released Python/PyTorch combinations and document a supported CPU/CUDA
  matrix.
- Run numerical checks on documented CUDA hardware with exact device, CUDA,
  PyTorch, and matmul-policy provenance.
- Add an optional TorchTitan/Nemotron-H adapter and the architecture's reviewed
  eligible-linear allowlist without vendored submodules.
- Add the probe-matched GEMM backend before claiming reproduction of the
  report's corresponding comparisons.
- Add independently runnable Transformer Engine/NVFP4 and BF16 reference paths
  before presenting cross-path reproduction results from this repository.
- Add sanitized public data manifests and model-checkpoint acquisition
  instructions, subject to the relevant model and dataset licenses.
- Add an OLMES runner and paper-scale launch configurations before advertising
  one-command reproduction of the 8B experiments.
- Add CI after reviewing workflow permissions and pinning third-party actions
  to immutable revisions.
