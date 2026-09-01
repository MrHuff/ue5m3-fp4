# Public-release checklist

This repository is a private alpha release candidate. Do not make it public
until every item marked **blocking** is complete. Checked boxes describe work
verified in this standalone repository; they do not waive organizational
approval.

## Credentials, security, and history

- [ ] **Blocking:** obtain security approval for the clean extraction and
  complete any source-repository credential remediation out of band.
- [x] Build the extraction on fresh Git history without importing source Git
  objects.
- [ ] **Blocking:** run Gitleaks or TruffleHog over the complete standalone Git
  history immediately before the public push and retain the report.
- [x] Run Gitleaks 8.29.1 over the complete current release-candidate history;
  rerun it after the final reviewed tag as required above.
- [x] Keep credentials, private checkpoint locations, bucket names, and
  cluster manifests out of source, tests, and examples.
- [x] Provide private correspondence/security email addresses in `SECURITY.md`.
- [ ] **Blocking:** enable and test GitHub private vulnerability reporting
  before changing the repository to public visibility.
- [ ] Add CI only after reviewing workflow permissions and pinning third-party
  actions to immutable revisions.

## Rights and attribution

- [ ] **Blocking:** confirm organizational approval for an Apache-2.0 code
  release.
- [ ] **Blocking:** review `NOTICE` against the exact Transformer Engine source
  revision and confirm that the extracted treatment and attribution are
  sufficient.
- [ ] Confirm the author list and paper metadata, then add `CITATION.cff`.
- [x] Exclude corporate logos and internal report templates from this initial
  extraction.
- [ ] Complete a final trademark/name review for the project description.

## Packaging and CPU reference

- [x] Provide a synthetic end-to-end training, checkpoint-reload, and
  quantized-inference example.
- [x] Document that the decoded-Torch path is software fake quantization, not
  native UE5M3 hardware or the paper's probe-matched comparator.
- [x] Require explicit model-conversion coverage rather than silently choosing
  eligible linears.
- [x] Keep the canonical training and inference recipes in source distributions.
- [x] Verify that those recipes are also installed and discoverable from a
  wheel.
- [x] Rebuild the wheel and sdist, install the wheel into a fresh environment,
  and rerun all tests and examples for the current release candidate.
- [ ] Record the final source-distribution and wheel SHA-256 digests.
- [ ] Freeze and test a supported Python/PyTorch CPU matrix.

## GPU and model integration

- [ ] **Blocking:** run the documented CUDA numerical tests on supported
  hardware and retain exact device, CUDA, PyTorch, and matmul-policy provenance.
- [ ] Add an optional TorchTitan/Nemotron-H adapter without vendored submodules.
- [ ] Publish the architecture's exact eligible-linear allowlist.
- [ ] Add the probe-matched GEMM backend before claiming reproduction of the
  corresponding paper comparisons.
- [ ] Add independently runnable Transformer Engine/NVFP4 and BF16 reference
  paths before presenting cross-path reproduction results.
- [ ] Add sanitized, immutable public data/checkpoint manifests and an OLMES
  runner before advertising one-command paper-scale reproduction.

## Final publication audit

- [ ] Confirm all public metric tables and artifact identifiers are sanitized
  and backed by releasable data.
- [ ] Verify README commands from a clean clone with no pre-existing caches or
  credentials.
- [ ] Confirm `SOURCE_PROVENANCE.md` hashes match the final committed sources.
- [ ] Tag the exact reviewed commit; build release artifacts from that tag.
- [ ] Re-run the complete-history secret scan on the tagged history.
- [ ] Have a maintainer who did not prepare the release verify every blocking
  item before changing repository visibility.
